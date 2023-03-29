#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include<vector>  // 包含下声明
#include<cmath>
namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

// 类似c中，是cuda底层内存。 gpu上内存，大小是size
struct CudaArray {
  CudaArray(const size_t size) {
    // stuck
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    //printf("CUDA error: %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

// 分别指定kernels的block和grid
struct CudaDims {
  dim3 block, grid;
};

// （一维） 根据size和线程数，计算需要的grid,block大小. 
// 总的全局线程数，等于size. 每个线程处理一个数据 
// 每个块，包含线程数目固定是256
CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;  // 256个线程一个块。 共需size/256个块。
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);   // 每个块256个线程
  dim.grid = dim3(num_blocks, 1, 1);         // 共size/256个块
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

// 把host上的vector,转成cuda上可操作的数组。（比如shape/stride,原来的一个n维的int数组）
// size是原vec实际大小，放到data数组里
CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;  // 放一个数组。包含
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions"); // 不超过8维
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////
// 每个线程处理out中一个数据（与线程id对应位置处的）。将该数据置成val
__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;     // 线程id(全局)
  if (gid < size) out[gid] = val;                         // 将out数据中对应数据置成val
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);  // 根据out计算所需线程。每个线程处理一个数据
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);  // 发布，在gpu端计算。调用FillKernel函数，传入out的首地址等（out已经是gpu上内存了）
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

// 把逻辑矩阵中行主序的第i个值，转成该元素在原始矩阵中的内存offset（按shape,strides）
// 无论int还是unsigned都很可能小于size_t需要的大小，所以必须有个size_t.可移植性好，用来表示数组索引等。以及内存大小。替代unsigned int
// https://www.zhihu.com/question/24773728
// __device__ :只能从设备端调用。在设备端执行
 __device__  size_t convert_gid_2_moffset(size_t gid,size_t size, CudaVec shape,CudaVec strides, size_t offset){
  // 首先根据i,确定索引：
  // 加入索引是[i,j,k,z],    i*shape[1]*...*shape[-1] + j* shape[2]*...*shape[-1] + ... + z ==gid
  // 每一维的索引id,是gid减去高维元素后，除以剩余大小的商。而余数用于计算剩下的维度
  // 首维是size/shape[0]. 直到最后一维
  uint32_t n_dim=shape.size;  // 共n维
  CudaVec index;              // 索引
  index.size=n_dim;
  size_t remain_n=gid;                       // 依次减. 被除数
  size_t remain_chushu=size;                 // 除数
  uint32_t i=0;
  while(i<n_dim){
      remain_chushu=remain_chushu/shape.data[i];  // 除数一开始是shape[1]*shape[2]*...*shape[-1]. 最后是1,对应的z是remain_n/1
      index.data[i]=remain_n/remain_chushu;   // 商是索引  2*3*4  size=24   位置18,18/12 是1，位置12,12/12=1  位置0-11是0，12-23是1
      remain_n=remain_n%remain_chushu;        // 余数，是剩余维度的元素数目。 (如果是0，是该维某个首元素。更低维索引都是0)
      i++;
      if (remain_n==0){
        break;                                // i以后的维度，索引都是0. 
      }   
  };
  if (i!=n_dim){           // 正常i==n_dim. break以后， i以后的维度，索引都是0. 
      while(i<n_dim){
        index.data[i]=0;
        i++;
      }
  }
  // 再根据索引+strides，确定offset
  size_t mem_offset=offset;     // 带初始offset
  for (int i=0;i<n_dim;i++){
     mem_offset+=strides.data[i] * index.data[i];
  }

  return mem_offset;

}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of a array
   *   offset: offset of a array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  // 给out[i]赋值。赋按行主序得到的逻辑矩阵a的第i个值。每个线程处理逻辑矩阵第i个值
  if (gid<size){    // 防止多分配了线程。
     size_t mem_offset=convert_gid_2_moffset(gid,size,shape,strides,offset);
     *(out+gid)=*(a+mem_offset);
  }
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  // 将sliceArray a,按行主序遍历，写入连续内存out中。 a的shape和strides,转成array传进去
  // 每个线程处理一个数据（对应out的第i个数据。也是a中逻辑矩阵的第i个数据。不需要搞遍历了）
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

// 每个线程设置一个值.  按行主序得到的逻辑矩阵out的第i个值。给out赋上
__global__ void setitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset){

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid<size){    // 防止多分配了线程。
     size_t mem_offset=convert_gid_2_moffset(gid,size,shape,strides,offset);
     *(out+mem_offset)=*(a+gid);  // out 是逻辑矩阵
  }
}

__global__ void setitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset){

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid<size){    // 防止多分配了线程。
     size_t mem_offset=convert_gid_2_moffset(gid,size,shape,strides,offset);
     *(out+mem_offset)=val;  // out 是逻辑矩阵
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  // 给定连续矩阵a，按照行主序设置到对应逻辑矩阵out上。 （矩阵都默认在device内存中）
  CudaDims dim = CudaOneDim(a.size); // 每个线程处理一个数据。共size个线程
  setitemKernel<<<dim.grid,dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), // out分配的原始size比较大。不是连续内存
                                         VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  size_t real_size=1;
  for (int i=0;i<shape.size();i++){
    real_size*=real_size*shape[i];
  }
  CudaDims dim = CudaOneDim(real_size); // 每个线程处理一个数据。共size个线程
  setitemKernel<<<dim.grid,dim.block>>>(val, out->ptr, real_size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION

// element-wise的模板
#define myMUL(a,b) (a)*(b)
#define myDIV(a,b) (a)/(b)
#define myEQL(a,b) (a)==(b)
#define myGE(a,b)  (a)>=(b)
#define myMAX(a,b) (a)>(b)? (a):(b)
#define myPOW(a,b) pow((a),(b))
#define myLOG(a) log((a))
#define myEXP(a) exp((a))
#define myTANH(a) tanh((a))

enum class MyOp{ // enum class，类型更强。用来封装不同op，作为模板参数
  mul,
  div,
  eql,
  ge,
  max,
  pow,
  log,
  exp,
  tanh
};

// 用作参数的内层函数。核函数内部只能访问设备内存 核函数内部不能使用静态变量(__device__ float a;)
__global__ void kernel_func(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size,MyOp op,int mod) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    switch (op)
    {
      case MyOp::mul:
        if (mod==0){
          out[gid] = myMUL(a[gid],b[gid]);
        }else{
          // b:原来传进来主机端的一个指针。需要把值复制到gpu内存上，对应gpu内存。否则不能访问/ (或者分配成gpu端可访问的主机内存/统一内存uvm)
          out[gid] = myMUL(a[gid],*b);
        }
        break;

      case MyOp::div:
        if (mod==0){
          out[gid] = myDIV(a[gid],b[gid]);
        }else{
          out[gid] = myDIV(a[gid],*b);
        }
        break;

      case MyOp::eql:
        if (mod==0){
          out[gid] = myEQL(a[gid],b[gid]);
        }else{
          out[gid] = myEQL(a[gid],*b);
        }
        break;

      case MyOp::ge:
        if (mod==0){
          out[gid] = myGE(a[gid],b[gid]);
        }else{
          out[gid] = myGE(a[gid],*b);
        }
        break;
 
      case MyOp::max:
        if (mod==0){
          out[gid] = myMAX(a[gid],b[gid]);
        }else{
          out[gid] = myMAX(a[gid],*b);
        }
        break;

      case MyOp::pow:
        if (mod==0){
          out[gid] = myPOW(a[gid],b[gid]);
        }else{
          out[gid] = myPOW(a[gid],*b);
        }
        break;

      case MyOp::log:
        out[gid] = myLOG(a[gid]);
        break;

      case MyOp::exp:
        out[gid] = myEXP(a[gid]);
        break;

      case MyOp::tanh:
        out[gid] = myTANH(a[gid]);
        break;

      default:
        break;
    }
  }
}

template<MyOp op>  // 模板参数op
void elewiseTmp(const CudaArray& a, const CudaArray& b, CudaArray* out){  // 模板函数本身。 里边用到模板参数p
    CudaDims dim = CudaOneDim(out->size);
    kernel_func<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size,op,0);        // 使用模板参数op
}

// scalar-wise的模板
template<MyOp op>  // 模板参数op
void scalarTmp(const CudaArray& a, scalar_t val, CudaArray* out){  // 模板函数本身。 里边用到模板参数p
    CudaDims dim = CudaOneDim(out->size);
    // val对应指针直接传过去，作为主机内存，没法在__global__函数内被设备访问
    // 需要分配设备内存 / 设备可访问的主机内存（零拷贝固定内存，统一内存等）
    scalar_t *devPtr;
    cudaError_t err =cudaMalloc(&devPtr,1*ELEM_SIZE);  // 分配一个设备端的全局内存.才能在device端访问。 nbytes
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    cudaMemcpy(devPtr,&val,1*ELEM_SIZE,cudaMemcpyHostToDevice);  // 把值复制到设备指针

    // 把设备指针传递给核函数
    kernel_func<<<dim.grid, dim.block>>>(a.ptr, devPtr, out->ptr, out->size,op,1);        // 使用模板参数op

    cudaFree(devPtr);  // 释放设备内存
}

// 单参模板
template<MyOp op>  
void elewiseTmp2(const CudaArray& a, CudaArray* out){  // 模板函数本身。 里边用到模板参数p
    CudaDims dim = CudaOneDim(out->size);
    kernel_func<<<dim.grid, dim.block>>>(a.ptr, a.ptr, out->ptr, out->size,op,0);        // b:用a占位
}


// 实例化：
void (*EwiseMul) (const CudaArray& ,const CudaArray& , CudaArray*)=elewiseTmp<MyOp::mul>;  // 初始化。指向实例化出来的函数
void (*EwiseDiv) (const CudaArray& ,const CudaArray& , CudaArray*)=elewiseTmp<MyOp::div>;  // 初始化。指向实例化出来的函数
void (*EwiseEq) (const CudaArray& ,const CudaArray& , CudaArray*)=elewiseTmp<MyOp::eql>;
void (*EwiseGe) (const CudaArray& ,const CudaArray& , CudaArray*)=elewiseTmp<MyOp::ge>;
void (*EwiseMaximum) (const CudaArray& ,const CudaArray& , CudaArray*)=elewiseTmp<MyOp::max>;
void (*EwiseLog) (const CudaArray& , CudaArray*)=elewiseTmp2<MyOp::log>;  // 单参数
void (*EwiseExp) (const CudaArray& , CudaArray*)=elewiseTmp2<MyOp::exp>;
void (*EwiseTanh) (const CudaArray& , CudaArray*)=elewiseTmp2<MyOp::tanh>;



void (*ScalarMul) (const CudaArray& ,scalar_t , CudaArray*)=scalarTmp<MyOp::mul>;  // 初始化。指向实例化出来的函数
void (*ScalarDiv) (const CudaArray& ,scalar_t , CudaArray*)=scalarTmp<MyOp::div>;  
void (*ScalarEq) (const CudaArray& ,scalar_t , CudaArray*)=scalarTmp<MyOp::eql>;
void (*ScalarGe) (const CudaArray& ,scalar_t , CudaArray*)=scalarTmp<MyOp::ge>;
void (*ScalarMaximum) (const CudaArray& ,scalar_t , CudaArray*)=scalarTmp<MyOp::max>;
void (*ScalarPower) (const CudaArray& ,scalar_t , CudaArray*)=scalarTmp<MyOp::pow>;
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// 全是一维
__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out,uint32_t M, uint32_t N,uint32_t P){

  // 1：simplest 每个线程处理一个输出c[i][j]
  size_t gid=blockDim.x * blockIdx.x + threadIdx.x;  
  size_t row_id= gid/P;                                    // c[i][j]对应的行列号
  size_t col_id=gid%P;
  scalar_t c_ij=0;
  for (size_t k=0;k<N;k++){                                // A[i:]  * B[:j]
      c_ij+=  *(a+ row_id*N +k) * *(b+ k*P+col_id);        // A[i,k]  B[k,j]
  }
  *(out+row_id*P+col_id)=c_ij;

}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION

  // 1:simplest. 每个线程处理一个输出c[i,j]. 有很多重复访问。共m,p个线程，每个线程读取一行一列数据：n*n. 不同线程读取了很多重复数据
  CudaDims dim = CudaOneDim(out->size);
  MatmulKernel<<<dim.grid,dim.block>>>(a.ptr,  b.ptr, out->ptr,M, N,P);
  
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

// 每个线程处理最多reduce_size个元素： i*reduce_size -(i+1)*reduce_size
// 但如果reduce_size比较大。最好分几个步骤优化。比如out_size=1,只有一个线程。处理所有数据。
// 可以分几次reduce到想要的output_size
__global__ void reduce_sum_kernel(const scalar_t* a, scalar_t* out,size_t out_size,size_t reduce_size){
   size_t gid= blockIdx.x*blockDim.x+ threadIdx.x;
   if (gid<out_size){
      scalar_t sum=0;
      size_t offset=gid*reduce_size;
      for (size_t i=0;i<reduce_size;i++){                   //  i*reduce_size -(i+1)*reduce_size
          sum+=*(a+offset+i);
      }
      *(out+gid)=  sum;
   }
}

__global__ void reduce_max_kernel(const scalar_t* a, scalar_t* out,size_t out_size,size_t reduce_size){
   size_t gid= blockIdx.x*blockDim.x+ threadIdx.x;          // 处理out[i],对应原始a[i*reduce_size]-a[(i+1)*reduce_size]
   if (gid<out_size){
      size_t offset=gid*reduce_size;
      scalar_t max=*(a+offset);

      scalar_t current;
      for (size_t i=0;i<reduce_size;i++){                   //  i*reduce_size -(i+1)*reduce_size
          current=*(a+offset+i);
          if(current>max){
             max=current;
          }
      }
      *(out+gid)=  max;                                      // 每个线程处理一个输出out[i]
   }
}


void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over.是a的shape[-1]
   */
  /// BEGIN YOUR SOLUTION
  // 每n个一组。每个输出对应一个线程. 要reduce维度已经放到了最后一维。每reduce_size个在内存中连续
  CudaDims dim = CudaOneDim(out->size);  // 共out个线程。每个线程处理erduce_size个数据
  reduce_max_kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr,out->size,reduce_size);
  /// END YOUR SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);  // 共out个线程。每个线程处理erduce_size个数据
  reduce_sum_kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size,reduce_size);
  /// END YOUR SOLUTION
}

}  // namespace cuda
}  // namespace needle


// 测试：
using namespace needle::cuda;
int main(){
  //void(*p)(const AlignedArray&, const AlignedArray& , AlignedArray*);   // 模板函数本身的指针.可以像函数名一样使用。  (声明)
  //p= elewiseTmp<&ewise_mat_f>;       
  
  //fptr1 = square;   // 这里可以赋值。函数内

  CudaArray a(24);

  int m=1;
  int n=1;
  int p=1;

  CudaVec shape;
  shape.size=3;
  shape.data[0]=m;
  shape.data[1]=n;
  shape.data[2]=p;

  CudaVec strides;
  shape.size=3;
  shape.data[0]=m;
  shape.data[1]=n;
  shape.data[2]=p;
  // for (uint32_t i=0;i<m*n*p;i++){
  //   *(a.ptr+i)=i;
  // }

  // size_t gid=18;
  // //convert_gid_2_moffset<<<1, 1>>>(gid,24,shape,shape,0);
  // convert_gid_2_moffset(gid,24,shape,shape,0);

  return 0;
}

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  // cuda内存
  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // 把底层gpu中内存数据，根据strides,shape等，复制成cpu上的np.array
  // 对应python端的函数:device.to_numpy(self._handle, self.shape, self.strides, self._offset)
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });// 原始stirdes，改成np的字节strides

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);   // 分配cpu内存。用来存转好的np
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);// copy到host内存(指针处）
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);// 根据host内存，对应strides等，生成numpy数组，返回（copy）
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        // 把a（原始np）的内容，复制到out的底层连续内存中。 从CPU到device。 其中a是python端传入的numpy数组
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
