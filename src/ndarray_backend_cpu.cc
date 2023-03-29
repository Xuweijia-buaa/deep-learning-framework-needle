#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <boost/coroutine2/all.hpp>

#include<vector>  // 包含下声明
#include<cmath>
namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);  // 4字节


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  // 构造方法,用pybind_class_的pybind:init函数绑定到类名Array上。在python端直接按modles.Array(size)调用,分配连续内存。返回指针
  // 这里按256字节对齐的。相当于8个float元素。也就是每8个元素都是对齐的。
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);  // 按8个float去对齐。 返回的ptr指向的内存地址，是256的倍数。共分配size个元素的内存大小
    if (ret != 0) throw std::bad_alloc(); // 结果flag是0
    this->size = size;                    // 该结构的size设置一下。
  }
  ~AlignedArray() { free(ptr); }         // pybind在绑定时，把管理权转移给了python。因此python创建后管理整个生命周期。用完删除时，也会调用析构方法。

  size_t ptr_as_int() {return (size_t)ptr; }   // 直接返回该结构体对应的地址
  scalar_t* ptr;                               // 给该结构体分配的地址
  size_t size;                                 // 占用的内存（float数目）
};


// 把out全填充成val
void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

// TODO:搞一个函数模板。这3个里，前2个参数数目类似：const AlignedArray& sliceArray,AlignedArray* other
// 公共部分：遍历不连续的slice。只输出索引（i,j,k,z,...）
// compact: 遍历  const AlignedArray& sliceArray  赋值
// setitem: AlignedArray*sliceArray

template < void(*op)(const AlignedArray&, AlignedArray* , int, int ,scalar_t) >   // 底层函数参数,作为模板参数。 op是一个函数指针
// 最外层。单纯输出不连续逻辑矩阵sliceArray，行主序每个元素对应的索引
void common_index(const AlignedArray& a, AlignedArray* out, scalar_t val,std::vector<int32_t> shape,std::vector<int32_t> strides, size_t offset){
    // 因为shape的维度不确定。不知道改用几个for循环。就定义一个len(shape)长的索引
    // 根据索引索引[i,j,k,z,,],，strides,offset,找到实际位置
    // 本模板函数负责按行主序迭代索引。假如有4维：
    // 从[0,0,0,0]到[0,0,0,1],再到[0,0,0,max].遍历完了[0,0,0,*]
    // 再到[0,0,1,0], [0,0,1,1],再到[0,0,1,max]，遍历完[0,0,1,*] 
    // 最终记录到[shape[i],shape[j],...shape[z]]. 所有维到达shape[i]. 遍历完该sliceArray每个元素的索引

    // 之后可以根据索引+strides(+offset),定位矩阵中该元素在内存中位置

    int n_dim=shape.size();      // 共n维
    int size=1;                  // 矩阵中总的元素数目
  
    // 索引和对应的进位标志   
    int carry=0;         // 进位标志。. 0111 -> 进位到1000
    std::vector<int32_t> current_ijz; // 用一个数组记录每个维度当前遍历到哪里了. 当做索引[i,j,k,z,,]本身 . 相当于n位。慢慢进位到满
                                      // 从[0,0,0,0]到[0,0,0,1],再到[0,0,0,max].遍历完了[0,0,0,*]
                                      // 再到[0,0,1,0], [0,0,1,1],再到[0,0,1,max]，遍历完[0,0,1,*] 
                                      // 最终记录到[shape[i],shape[j],...shape[z]]. 所有维到达shape[i]

    // 初始化索引为0000
    for (int i=0;i<n_dim;i++){     // 按维度数目，初始化索引
        current_ijz.push_back(0);
        size=size*shape[i];        // 逻辑矩阵元素数目
    }

    // 开始输出索引：
    int offset_in_slice;
    for (int n=0;n<size;n++){
         
        // 先使用[i,j,k...z]
        offset_in_slice=offset;
        for (int i=0;i<n_dim;i++){
          offset_in_slice+=current_ijz[i]* strides[i];  // i*s[i]， ijkz对应的在内存中的offset
        }

        // 不同模板，这里传入的底层参数不同.一些参数用不上
         op(a,out,n,offset_in_slice,val);

        // 再修改索引：从后往前判断哪一位满了，进位。自己归0 [0 0 0 0 0]。n位修改成新索引
        for (int i=n_dim-1;i>=0;i--){         // 从最后一维看起。到shape了，进位。高维也一直进。低维idx置0            
            if (i==n_dim-1){                  // 最后一维，默认每次进位，加1。(如果加1满了，置0，让高维进位，carry是1）， （加1不满，进位到这里截止。carry=0）
                if (current_ijz[i]<shape[i]-1){ // 加1后不满，进位到这里截止。carry=0.其他更高维保持不变
                    current_ijz[i]+=1;
                    carry=0; 
                    break;
                }else{
                    carry=1;                    // 已经满了(到了shape[i]-1)，再+1需进位。自己设置成0，进位
                    current_ijz[i]=0;
                }                
            // 其他每一维。如果有进位，加1。（如果加1满了，置0，让高维进位 carry是1）  （加1不满，进位到这里截止。carry=0。 break） 否则跳过                               
            }else{                                   // 更高维
                if (carry==1){                       // 如果需要进位
                  if (current_ijz[i]<shape[i]-1){    // 不满足进位条件. 增加后直接用，直到s-1。只加1
                    current_ijz[i]+=1;
                    carry=0; 
                    break;
                  }else{
                    carry=1;                     // 上次到shape-1了。 这次满足进位条件了。自己设置成0，更高维进位
                    current_ijz[i]=0;
                  }               
                }                                 // 如果不需要进位，跳过
            }
        } // 本次索引使用完毕
   }    // 遍历完了逻辑矩阵的所有元素
};      // 模板函数结束

// 底层函数,作为模板参数传入
// 遍历切片得到的逻辑矩阵，并写入新矩阵a[i]（compact）/ 被设成值val（ScalarSetitem）/ 被设成值a[i]（EwiseSetitem）
// 根据strides,offset,shape来索引元素 （行主序）
// 对对象属性的访问只能用./ 对指针访问，.和->都可以.
void compact_f(const AlignedArray& a, AlignedArray* out, int n, int offset_in_slice,scalar_t val){
        *(out->ptr+n)= *(a.ptr+offset_in_slice);  // compact: a是逻辑矩阵，out是内存连续新矩阵
};

void setitem_f(const AlignedArray& a, AlignedArray* out, int n, int offset_in_slice,scalar_t val){
      *(out->ptr+offset_in_slice)=*(a.ptr+n);  // setitem  a是内存连续的底层数据。out是__getitem__得到的逻辑矩阵。把a按行主序，set到逻辑矩阵对应内存上
};
void setitem_vf(const AlignedArray& a, AlignedArray* out,int n, int offset_in_slice,scalar_t val){
      *(out->ptr+offset_in_slice)=val;  // setitem    out是__getitem__得到的逻辑矩阵。把val按行主序，set到逻辑矩阵对应内存上
};


// 实例化：
void (*comp_ptr) ( const AlignedArray& , AlignedArray* , scalar_t, std::vector<int32_t> ,std::vector<int32_t> , size_t) =common_index<compact_f>;
void (*setit_ptr) ( const AlignedArray& , AlignedArray* , scalar_t, std::vector<int32_t> ,std::vector<int32_t> , size_t) =common_index<setitem_f>;
void (*setval_ptr) ( const AlignedArray& , AlignedArray* , scalar_t, std::vector<int32_t> ,std::vector<int32_t> , size_t) =common_index<setitem_vf>;


// a: 原始的逻辑矩阵。（可能不连续，通过strides,offset,shape等来索引.原始输入shape是list）
// out:已经在python端，通过调用Array类的构造方法，分配好了连续内存
// 把a中的逻辑矩阵，按行主序的顺序，写入到out对应的连续数组中。
// 需要遍历逻辑矩阵a
void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out       shape是不变的
   *   strides: strides of the *a* array (not out, which has compact strides)    输入矩阵的strides。不是连续的
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)   输入矩阵的offset
   *
   * Returns:
   *  只返回这个内存（设置对应的size）
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN YOUR SOLUTION
  comp_ptr(a,out,0,shape,strides,offset);
  /// END YOUR SOLUTION
}

// 把底层数据a，写入_getitem_后得到的切片out中（按照out展开的顺序）。
// 需要遍历out,把a的底层数组按out行主序的顺序，依次写入这个逻辑矩阵的对应位置
// out: 已经根据slice得到的切片（逻辑矩阵），内存并不一定连续。 有strides,shape,offset等参数，控制这个逻辑矩阵对应的真实内存
//      尽管不连续，但一定是按行主序排布的。shape中的最后一维是基础。
// a: 把a中的值，设置到这个切片上。a是一个紧密排布的矩阵，大小等同于out(逻辑矩阵)。 串行设置到out的每个实际(i,j,k)位置，行主序展开的
void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  setit_ptr(a,out,0,shape,strides,offset);
  /// END YOUR SOLUTION
}

// 把val写到 getitem_后得到的切片out中.  同逻辑矩阵的shape个
// 需要遍历逻辑矩阵
void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN YOUR SOLUTION
  AlignedArray a(0);  //不会用到
  setval_ptr(a,out,val,shape,strides,offset);
  /// END YOUR SOLUTION
}



void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   * a,b大小相同
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   * a,out大小相同
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

/**
 * In the code the follows, use the above template to create analogous element-wise
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
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION

// element-wise的模板
template<scalar_t (*op)(const scalar_t &,const scalar_t &)>     // 模板参数op，是一个函数(指针).对应内部函数
void elewiseTmp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){  // 模板函数本身。 里边用到模板参数p
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i],b.ptr[i]);                        // 使用模板参数（p）
  }
}

// scalar-wise的模板
template<scalar_t (*op)(const scalar_t &,const scalar_t &)>     // 模板参数op，是一个函数(指针).对应内部函数
void scalarTmp(const AlignedArray& a, scalar_t val, AlignedArray* out){  // 模板函数本身。 里边用到模板参数p
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], val);           // 使用模板参数（p）
  }
}

// 底层函数：只传一个参数
template<scalar_t (*op)(const scalar_t &)>     // 模板参数op，是一个函数(指针).对应内部函数
void elewiseTmp2(const AlignedArray& a, AlignedArray* out){  // 模板函数本身。 里边用到模板参数p
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i]);           // 使用模板参数（p）
  }
}

// 用作参数的内层函数
scalar_t mul_f(const scalar_t & a, const scalar_t & b){
   return a*b;
};
scalar_t div_f(const scalar_t & a, const scalar_t & b){
   return a/b;
};
scalar_t equal_f(const scalar_t & a, const scalar_t & b){
   return a==b;
};
scalar_t ge_f(const scalar_t & a, const scalar_t & b){
   return a>=b;
};
scalar_t max_f(const scalar_t & a, const scalar_t & b){
   return a>b? a:b;
};
scalar_t power_f(const scalar_t & a, const scalar_t & b){
   //return a**b;
   return pow(a,b);
};

scalar_t log_f(const scalar_t & a){
   return log(a);
};
scalar_t exp_f(const scalar_t & a){
   //return a**b;
   return exp(a);
};
scalar_t tanh_f(const scalar_t & a){
   return tanh(a);
};


// 用底层函数实例化：

// 函数指针p. 对应的函数参数是const AlignedArray& , const AlignedArray& , AlignedArray* ， 作为该指针的类型
//           右侧是形参列表，表示p指向一个函数。左侧是函数的返回值。
//void (*p) (const AlignedArray&, const AlignedArray& , AlignedArray*);      // 模板函数本身的指针.可以像函数名一样使用。  (声明)
//p = elewiseTmp<&ewise_mat_f>;                                              // 指向实例化出来的函数. 其中模板参数是函数（指针）. 不能用赋值语句

// 实例化element-wise对应的模板. 指针EwiseMul, 本身相当于函数名(namespace内的)。可以返回
void (*EwiseMul) (const AlignedArray&, const AlignedArray& , AlignedArray*)=elewiseTmp<&mul_f>;  // 初始化。指向实例化出来的函数，模板参数是一个底层函数
void (*EwiseDiv) (const AlignedArray&, const AlignedArray& , AlignedArray*)=elewiseTmp<&div_f>;  
void (*EwiseEq) (const AlignedArray&, const AlignedArray& , AlignedArray*)=elewiseTmp<&equal_f>;  
void (*EwiseGe) (const AlignedArray&, const AlignedArray& , AlignedArray*)=elewiseTmp<&ge_f>;  
void (*EwiseMaximum) (const AlignedArray&, const AlignedArray& , AlignedArray*)=elewiseTmp<&max_f>; 
void (*EwiseLog) (const AlignedArray&,  AlignedArray*)=elewiseTmp2<&log_f>;     // 只有入出。
void (*EwiseExp) (const AlignedArray&,  AlignedArray*)=elewiseTmp2<&exp_f>;  
void (*EwiseTanh) (const AlignedArray&, AlignedArray*)=elewiseTmp2<&tanh_f>; 

// 实例化scalar对应的模板
void (*ScalarMul) (const AlignedArray&, scalar_t , AlignedArray*)=scalarTmp<&mul_f>;  
void (*ScalarDiv) (const AlignedArray&, scalar_t , AlignedArray*)=scalarTmp<&div_f>;
void (*ScalarEq) (const AlignedArray&, scalar_t , AlignedArray*)=scalarTmp<&equal_f>;  
void (*ScalarGe) (const AlignedArray&, scalar_t , AlignedArray*)=scalarTmp<&ge_f>;
void (*ScalarMaximum) (const AlignedArray&, scalar_t , AlignedArray*)=scalarTmp<&max_f>;
void (*ScalarPower) (const AlignedArray&, scalar_t , AlignedArray*)=scalarTmp<&power_f>;


/// END YOUR SOLUTION


// 根据索引[i,j],返回矩阵对应位置的元素
scalar_t& matrix_ele(const AlignedArray& a,uint32_t row_id, uint32_t col_id,uint32_t row_size ){
    return *(a.ptr+ row_id * row_size + col_id);
}

scalar_t& matrix_ele(AlignedArray* out,uint32_t row_id, uint32_t col_id,uint32_t row_size ){
    return *(out->ptr+ row_id * row_size + col_id);
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  // (m,n)  * (n,p)
  // 最原始的版本： a的i行，b的j列，依次遍历。得到cij
  // for (uint32_t i=0;i<m;i++){
  //   for (uint32_t j=0;j<p;j++){
  //     scalar_t cij=0;
  //     for(uint32_t k=0;k<n;k++){
  //       //cij+= a[i,k] * b[k,j];    // b先变k,跨行访问
  //       cij+= matrix_ele(a,i,k,n) * matrix_ele(b,k,j,p);    // b先变k,跨行访问
  //     }
  //     //c[i,j]=cij;
  //     matrix_ele(out,i,j,p)=cij;
  //   }
  // }

   // a依然沿着行访问。对每个拿进来的a[i,k], 重复利用,把对应的b[k]行消费完
  for (uint32_t i=0;i<m;i++){
    for(uint32_t k=0;k<n;k++){  // 消耗b的第k行
        //scalar_t aik=a[i,k];    // aik重复使用。放在寄存器中
        scalar_t aik=matrix_ele(a,i,k,n);
        for (uint32_t j=0;j<p;j++){
            //c[i,j]+= aik * b[k,j];    // 消耗b的第k行,b沿着行访问
            if (k==0){
                matrix_ele(out,i,j,p)=0; // 对cij. i不变，a[i,k]消耗b的第k行。 a[i,0]消耗b的第0行， a[i,k]消耗b的第k行。因此k=0时先初始化cij
            };
            matrix_ele(out,i,j,p)+= aik * matrix_ele(b,k,j,p);
        } 
    }
  }
  /// END YOUR SOLUTION
}

// __restrict__关键字：表示内存只能由这一个指针改变。告诉编译器，对象已经被指针所引用，不能由该指针外其他方式修改该对象的内容
//                    从而可以让编译器知道，只有a在操作这片内存。可以放心的进行优化。比如把几个修改指令合并成一个
//                    这样即使我们写的代码是非vector化的，编译器也会把他们优化成一个指令？一起操作。相当于向量化了
//                    否则编译器会认为a和b有可能指向相同内存。只能一句句执行，修改a;修改b;修改a(结果是3); 结果不同于修改a;修改a;修改b（结果是b）; 
//                    所以编译器只能严格按顺序执行，不能合并修改a的指令. 相当于无法并行化
// __builtin_assume_aligned: 告诉编译器,指针指向的内存,已经至少与align字节对齐了。编译器可以根据这个提示做一些优化，
//                           比如可以一次加载对齐的n个元素，而非一个个加载。
// 小矩阵乘法：内联
inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  
   * Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  
   * Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:                                                                         Vk
   *   a: compact 2D array of size TILE x TILE     a对应的小矩阵（V1,V2）        ....___....   
   *                                                                       V1 ....|__|...
   
   *   b: compact 2D array of size TILE x TILE     b对应的小矩阵 (V2,V3)            v2 
                                                                                . .. . 
                                                                                .    .
                                                                                . __ .  
                                                                            vk  .|  |.
                                                                                .|__|.
                                                                                .    .
                                                                                .    .
      每次把VV个元素拿进来。a中每个元素可以重用V2次（和b每列乘）     
                         b中每个元素可以重用V1次（和a每行乘）

      用完a行第k块，b列第k块后，这2个块完成使命；计算了C[V1,V2]的一部分。
      对于C[V1,V2]最左上角的元素 本来是a的首行乘b的首列。 但是a,b都被切成了k个大小为V的块。假如切成了3个块，该元素相当于用3次才算完。

      遍历完a中第i大行，（同时是b中第j大列）。 c{i,j}才被完整算出来了

      前提是寄存器能放下VV个元素（8*8*4=256个字节）。一个64位的寄存器，可以放8字节。一共需要32个寄存器                                                                        

   *   out: compact 2D array of size TILE x TILE to write to  计算小矩阵C(V1,V3)
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);  // （V1,V2）.默认已经按每V个元素对齐 （V=8）（8*4字节=256）。地址都是256的整数倍，可以一次取256的倍数到缓存
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN YOUR SOLUTION
  // 内存连续。直接乘：
  // a是V1,Vk个。 b是Vk,V2个
  for (uint32_t i=0;i<TILE;i++){
    for(uint32_t k=0;k<TILE;k++){  // 消耗b的第k行
        //scalar_t aik=a[i,k];    // aik重复使用。放在寄存器中
        //scalar_t aik=matrix_ele(a,i,k,n);
        for (uint32_t j=0;j<TILE;j++){
            //c[i,j]+= aik * b[k,j];    // 消耗b的第k行,b沿着行访问
            // matrix_ele(out,i,j,p)+= aik * matrix_ele(b,k,j,p);
            *(out+i*TILE+j) += *(a+i*TILE+k) * *(b+k*TILE+j);     // 最后2维 vivj,内存连续
        } 
    }
  }
  
  /// END YOUR SOLUTION
}


void zero(float* a,int size){
    for(int i=0;i<size;i++){
      *(a+i)=0;
    }
}

// a: 已经按在逻辑上被当做是 m/v,n/v个 v,v的小矩阵了
// b: 是逻辑上n/v,p/v个 v,v的小矩阵
// 但底层内存都是连续的
// 输出的c,也是按每个a[i],b[j]块，得到小矩阵c[v,v].  共m/v,p/v个c[v,v]
void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder. （假设v可以同时被m,n,p整除。都可以切成v,v的块）
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *   * Args:      
   *                                                                               vk
   *   a: compact 2D array of size TILE x TILE     a对应的小矩阵（V1,V2）        ....___....   
   *                                                                       V1 ....|__|...
   
   *   b: compact 2D array of size TILE x TILE     b对应的小矩阵 (V2,V3)            v2 
                                                                                . .. . 
                                                                                .    .
                                                                                . __ .  
                                                                            vk  .|  |.
                                                                                .|__|.
                                                                                .    .
                                                                                .    .
      每次把VV个元素拿进来。a中每个元素可以重用V2次（和b每列乘）     
                         b中每个元素可以重用V1次（和a每行乘）

      用完a行第k块，b列第k块后，这2个块完成使命；计算了C[V1,V2]的一部分。
      对于C[V1,V2]最左上角的元素 本来是a的首行乘b的首列。 但是a,b都被切成了k个大小为V的块。假如切成了3个块，该元素相当于用3次才算完。

      遍历完a中第i大行，（同时是b中第j大列）。 c{i,j}才被完整算出来了

      前提是寄存器能放下VV个元素（8*8*4=256个字节）。一个64位的寄存器，可以放8字节。一共需要32个寄存器。
      即使寄存器不够，也会拿到缓存来，下次访问较快。每次是256字节的连续内存（V,V）
   */
  /// BEGIN YOUR SOLUTION
  //  把大矩阵切成每个小矩阵。
  //  a的第i大行，长度n,切成了k=n/vk个块。 B的第j大列，长度n,也切成n/vk个块
  //     第i大行的第k块(v1,vk)，消耗B的第j大列的第k个块(vk,v2)。 计算C[v1,v2]的一部分。 每个块中元素，拿进来重复参与对b中对应块的计算，可以重用v2次
  //                                                                            b块中元素同，重复拿来与a的对应块每行相乘。重用v1次。都不需要再从内存加载

   uint32_t a_row=m/TILE;   // a被横向切成的数目
   uint32_t K=n/TILE;       // a中每个大行的块数。同b中每个大列的块数
   uint32_t b_col=p/TILE;   // b被纵向切成的大块数目。

   //Fill(out,0);

   //float small_a[TILE][TILE];    // (v1,vk),拿进来的都放这里。 如果直接用一个指针的话，内部计算矩阵乘时，还需要从内存中一个一个拿
   //float small_b[TILE][TILE];
   float small_c[TILE*TILE];     // 一段连续内存，缓存c[i][j]

   uint32_t offset_a;
   uint32_t offset_b;
   uint32_t offset_c;
   for (uint32_t i=0; i<a_row;i++){     // 遍历a的第i个大行
      for (uint32_t j=0; j<b_col;j++){  // 与b的第j个大列   ： 计算c[i][j]
         offset_c=(i*(p/TILE)+j) * TILE*TILE;    // c[v1,v2]，位于第i大行，第j大列. 每个块v1v2个连续元素 (也可以这里初始化成0)
         zero(small_c,TILE*TILE);                      // 这里初始化成0
         for (uint32_t k=0; k<K;k++){   // 取出其中的小矩阵 [v1,vk],[vk,v2]. 做小矩阵乘法
           // 从python端传入时，这些小矩阵已经是连续的了，按最后2维是VV,compact的.每个块内部连续，块间行主序排布。乘法时可直接用
           // 第ij个块，经过了i*行，每行n/V个块。又经历了j列。共经历了 i*(n/Vk)+j个块。每个块VV个元素
           offset_a=(i*(n/TILE)+k) * TILE*TILE;   // 第i大行，经过i*(n/v1)个快。 第k块，经历了k个块。每个块v1vk个连续元素     [v1,vk]起始处offset
           offset_b=(k*(p/TILE)+j) * TILE*TILE;   // 第j大列。第k行含k*(p/v2)个块。 第j大列，过j个块 每个块vkv2个连续元素   [vk,v2]起始处offset
           AlignedDot(a.ptr+offset_a,b.ptr+offset_b,small_c);  // 结果写入临时缓存
      }
      // 计算完了C[i][j]块，写入C：
      for(int c=0;c<TILE*TILE;c++){
        *(out->ptr+offset_c+c)=*(small_c+c);
      }
    }
   }
  /// END YOUR SOLUTION
}


 //  调用设备端reduce函数，把A compact,reduce后拷给out
 //  A: 已经reshaBpe到最后一维（且compact好的）。只需要reduce最后这一维（已经compact）
 //  out: 已经分配好连续内存，放reduce后的矩阵。最后一维大小是1了
void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over    。已经把要reduce的维度转到最后一维了：(2,4,5,3)
   *   out: compact array to write into.. 原来(2,3,4,5),现在(2,1,4,5)  如果reduce维度1
   *   reduce_size: size of the dimension to reduce over.   最后一维的原始大小3。要reduce的维度大小
   */

  /// BEGIN YOUR SOLUTION
  // 最后一维：  a[1,2,3,4 | 5 6 7 8]  每reduce_size个a[i]-a[i*n]，放到out[i]
  scalar_t max=*(a.ptr);
  scalar_t value;
  for (int g=0;g<a.size/reduce_size;g++)  {    // 第g组
      for (int i=0;i<reduce_size;i++){
        value=*(a.ptr + g*reduce_size +i);    //  中第i个
          if (value>max){
             max=value;
          }
      }
      *(out->ptr+g)=max;
      max=*(a.ptr + (g+1)*reduce_size);      // 重置,到下一组首个
  }
  int remain=a.size%reduce_size;
  if (remain!=0){
    int g=a.size/reduce_size;
    for (int i=0;i<remain;i++){
        value=*(a.ptr + g*reduce_size +i);    //  中第i个
          if (value>max){
             max=value;
          }
    } 
    *(out->ptr+g)=max;
  }
  /// END YOUR SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
    // 最后一维：  a[1,2,3,4 | 5 6 7 8]  每reduce_size个a[i]-a[i*n]，放到out[i].同上
  scalar_t sum=0;
  scalar_t value;
  for (int g=0;g<a.size/reduce_size;g++)  {    // 第g组
      for (int i=0;i<reduce_size;i++){
        value=*(a.ptr + g*reduce_size +i);    //  中第i个/ sum
        if (i==0){
          *(out->ptr+g)=value;      
        }else{
          *(out->ptr+g)+=value;
        }
      }
      sum=0;                                 // 重置
  }
  int remain=a.size%reduce_size;
  if (remain!=0){
    int g=a.size/reduce_size;
    for (int i=0;i<remain;i++){
        value=*(a.ptr + g*reduce_size +i);    //  中第i个
        if (i==0){
          *(out->ptr+g)=value;      
        }else{
          *(out->ptr+g)+=value;
        }
    } 
  }
  /// END YOUR SOLUTION
}


}  // namespace cpu
}  // namespace needle

// 测试：
using namespace needle::cpu;
int main(){
  //void(*p)(const AlignedArray&, const AlignedArray& , AlignedArray*);   // 模板函数本身的指针.可以像函数名一样使用。  (声明)
  //p= elewiseTmp<&ewise_mat_f>;       
  
  //fptr1 = square;   // 这里可以赋值。函数内

  AlignedArray a(24);
  // AlignedArray b(6);
  // AlignedArray out(4);
  // AlignedArray *out_p=&out;
  uint32_t m=2;
  uint32_t n=3;
  uint32_t p=4;
  // a
  // 0 1 2
  // 3 4 5  (2,3)

  // b: [3,2]
  // 1 1
  // 1 1
  // 1 1

  // out:
  // 3 3
  // 12 12
  
  for (int32_t i=0;i<m*n*p;i++){
    *(a.ptr+i)=i;
  }
  // for (uint32_t i=0;i<n*p;i++){
  //   *(b.ptr+i)=1;
  // }
  // for (uint32_t i=0;i<m*p;i++){
  //   *(out.ptr+i)=0;
  // }

  // Matmul(a,b,out_p,m,n,p);
  std::vector<int32_t> shape;
  shape.push_back(m);
  shape.push_back(n);
  shape.push_back(p);


  // size_t gid=11;
  // convert_gid_2_moffset(gid,24,shape,shape,0);

  return 0;
}


PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;     // 把__tile_size__,作为module的一个属性传回去。在python端可以调用： m.__tile_size__/ m._device_name__

  //py::class_：绑定c++中的类，绑定到类名Array上
  py::class_<AlignedArray>(m, "Array")
       // py::init:  包这个类的一个构造方法的参数。绑定c++中这个类对应的构造方法。
       // 在python端直接按modles.Array(size)调用,分配连续内存。返回指针
       // TODO：用take_ownership，Python 获得其所有权.管理其声生命周期。 python端删除该对象时，也会让c++调用析构方法 （指针指向静态对象，或者c++也删除时，会有问题）
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)           // 这个类上的ptr_as_int方法，被绑定到python端的ptr方法上。可以在python端直接调用该对象的ptr方法。
      .def_readonly("size", &AlignedArray::size);      // 对应python端，该对象的size方法

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  // 把传入的连续数组a，copy给nparray. 新建了一个py::array_t数组
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;  // numpy对应的strides,是带字节的
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });       // 转成numpy类似的strides（带字节数目）
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset); // 把底层数据，按照shape.stides解释成一个逻辑上的nparray返回.
  });

  // convert from numpy (with copying)
  // 把原始np矩阵a，对应的底层内存， 复制到该设备array对应的地址中，作为底层数组
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);             // 包成compact函数，被python调用。传入原来矩阵的内存（可能不连续，含strides,offset等）和已经在底层分配好的连续内存
                                         //                                                                             (通过make中调用的Array函数)
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);


  // 给python端，输入自己，other,结果 对应的内存地址。每个是个独立的函数
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
