"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

# 导入array_api包后，以及其中的NDArray类
# NDarray重载了各种操作，底层都是调用.so文件，操作c++分配的内存空间
from .backend_selection import array_api, NDArray


# 该Tensor的op,把输入的多个Tensor的底层数据，拼成一个tuple。作为该Tensor的cached_data。
# 生成一个TensorTuple
class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        # 把n个input Tensor的cacahe_data,合成一个tuple,作为该TensorTuple的cached_data
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        # 输出梯度是一个TensorTuple对象（同cache_date）
        # 对应的输入梯度，是n个tuple各自的梯度。分别拆成n个
        return tuple([out_grad[i] for i in range(len(out_grad))])  # 这里用到了TensorTuple的getitem.实现靠的是TupleGetItem Op

# 把n个Tensor组合成一个Tensor（TupleTensor）
def make_tuple(*args):
    # 输入是n个Tensor
    # 生成一个TensorTuple类的对象。input是这n个Tensor组成的list,op是该MakeTensorTuple
    # 对应的数据是compute得到的.调用里边op的compute，计算cache_data. input一般是n个Tensor
    # 得到的cached_data，是n个input Tensor的cacahe_data，tuple化
    return MakeTensorTuple()(*args)
                                     # 通过inputs,该op,调用compute,生成cache_data
                                     # 其中make_from_op是一个类方法，通过cls.__new__(cls)。new一个TensorTuple对象。使得输出是一个TensorTuple对象

# 针对Tensor的op
class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple): # 传入的a是TensorTuple，且对应op是多个输入拼的
            return a.inputs[self.index]                      #      取a的第i个输入Tensor
        return Tensor.make_from_op(self, [a])                # 否则生成一个新Tensor. inputs是[a]

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        # 梯度是先传到第i个Tensor(out_grad)。再回传该TensorTuple
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)      # 所有子Tensor grad，组成大TupleTensor的grad

#  TensorTuple中，第i个Tensor
def tuple_get_item(value, index):
    return TupleGetItem(index)(value)    # 调用__call__,传入的value是TupleTensor。默认取value的第i个输入Tensor

# 针对TensorTuple的op，
class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)  # 调用TensorTupleOp里的__call__。
                                       # new一个TensorTuple对象，输出调用compute计算x+c0,x+c1.


class EWiseAdd(TensorOp):
    # 前向作用于原始对象NDArray（底层可以是自己的各种实现）.因此不一定是一个tensorOP.直接是原生输入的前向计算
    # 传入的是Tensor缓存的cached_data(各种类型),而不是直接是Tensor。
    # 2Tensor做加法，底层是值cached_data做加法，结果存入新Tensor的cached_data
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    # 指定该op反向计算的逻辑。定义如何根据来的梯度，计算输出的梯度。
    # 输入是该节点输出值本身的梯度vi。输出该op对每个原始输入贡献的梯度vki。node是该op节点的原始输出值，forward图中有。作为反向图中该op的另一个输入
    # 构建反向计算图时，输入vi（下一节点j在反向图的梯度，经过sumop等得到的）,node（原前向图中的该值）,经过该op，输出vki。
    # 计算对每个输入该op的节点贡献的梯度vki。作为该反向op的输出值。（这里k=2，对应原始的2个输入)   之后会据此计算vk（经过sum op)
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

# 封装op,依然返回一个Tensor. 原始a,b（不论底层是什么）经过前向计算后的结果
# 这里a，b也已经是原始输入经过封装后的Tensor了。才能传给op
def add(a, b):
    # 初始化一个TensorOp（EWiseAdd()），并调用__call__ 通过Add()(a,b)
    # 底层调用了make_from_op(op,input),返回new的Tensor （通过_init(op, inputs)）
    # 只是指定了该tensor对应的op，input,不一定实际去算了
    # 如果是eager mode,该tensor还调用了realize_cached_data()
    # 即op.compute，根据该op的输入，得到输出Tensor的值，存在该tensor的cached_data中。
    # 输出的该tensor，实际上对应的是一个op操作。存了op和input,以及必要的cache_data用作前向计算
    return EWiseAdd()(a, b)



class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar  # 原生cached_data的计算。在Tensor计算时，实际传入的是底层的cached_data
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # y=x^n  ->  dy/dx= n*(x^n-1)
        x_tensor=node.inputs[0]         # （m,n） 输入的Tensor x
        dy_dx= self.scalar * x_tensor**(self.scalar-1)   # n*(x^n-1)  x^n-1，Tensor计算.重载了__pow__。返回一个新Tensor
        return (out_grad*dy_dx,)       # 对输入a贡献的梯度vki。维度同a  (Tensor乘法,底层按位乘，重载了__mul__)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b                     # 按位置除
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node:Tensor):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs          # 输入Tensor. 输出Tensor是node
        dy_da= Tensor(init.ones(*b.shape,device=node.device,dtype=node.dtype))/b            # 1/b  node/a           y对a的梯度：Tensor   按位除.  但是直接是1/b . a有可能是0  y=a/b  by=a
        dy_db=-1 * a / (b**2)    # a* (-1) * b(-2)    y对b的梯度：Tensor
        return out_grad*dy_da,out_grad*dy_db
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        dy_dx=1/self.scalar
        return (out_grad*dy_dx,)        # Tensor. 总是输出tuple
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        N = len(shape)
        perm = list(range(N))
        if N > 1:
            if self.axes!=None:
                perm[self.axes[0]] = self.axes[1]
                perm[self.axes[1]] = self.axes[0]
            else:
                perm[-1] = N - 2  # 原来（0，1，2，3）  现在（0，1，3，2）
                perm[-2] = N - 1
        #return array_api.transpose(a,perm)
        return a.permute(perm)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)  # Tensor的转置. 是自定义的Tensor计算,转特定2维
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        orig_shape=node.inputs[0].shape
        return out_grad.reshape(orig_shape)   # 自定义Tensor计算
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)

import numpy as np

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad:Tensor, node):
        ### BEGIN YOUR SOLUTION
        orig_shape=node.inputs[0].shape  # (2,1)
        new_shape=self.shape             # 广播相当于把原来的矩阵，乘了n倍(内存不变). 反向操作： 对原始x的贡献，是每个翻倍的贡献 (6,3,2,1)

        # 对原始xi来说，每个翻倍都有一个梯度。总的梯度是所有xi所有翻倍输出的梯度的累加

        # 把那些多的维度找出来，沿着这些维度聚合。 多的维度只能出现在前边/ 或者是同维度但是原始是1的位置

        #让新增的维度放最前边。  old(2,3) ->  new(4，5，2，3) 变成->  (20,2,3)  聚合-> old(2，3)
        # 如果是同维度。但值变了：
        # old (10,1) ->  new(10,9)  -> 沿着非1的维度聚合  -> old(10,1)
        # old (1,4) ->   new(5,4)  ->  沿着非1的维度聚合  -> old(1,4)

        # 同时变了：
        # old (2,1,3) -> new (4,2,5,3)  -> 多出来的维度的位置（0）+ 非1维度（原来是1）的位置(2). 聚合： （2，3）
        # old (2,3,1) -> new (4,2,3,5)  -> 多出来的维度的位置（0）+ 非1维度（原来是1）的位置(3). 聚合： （2，3）

        # 先聚合多出来的维度。统一放前边  : (2,5,3)/(2,3,5)

        # TODO：
        # (15,5) -> (15,1)   (4,1) -> (1,)
        # if len(orig_shape)==1 and orig_shape[0]==1 and len(new_shape)!=len(orig_shape):
        #     grad=out_grad.sum()


        if len(new_shape)!=len(orig_shape):
            dim1=np.prod(out_grad.shape[0:len(orig_shape)])               # 放在第一维。先聚合多出来的维度
            out_grad_reshaped = out_grad.reshape( [dim1]+list(orig_shape))  # 放在第一维。先聚合多出来的维度
            out_grad=out_grad_reshaped.sum(axes=(0,))  # 利用Tensor的沿某个维度sum的函数
        # 同维度的2向量，聚合原来是1的维度: (2,5,3) ->(2,1,3)   (2,3,5)->(2,3,1)
        grad=out_grad  # outgrad，变成和输入维度数目相同了。
        for i,size in enumerate(orig_shape):   # broadcast后新增的每个维度，聚合
            shape=list(grad.shape)
            if size==1:                        # 原始是1的维度。把新的聚合到这一维。 如果新的这一维度也是1，不变。（1，1）-> (1,)/(3,1)->(1,1)
                if shape[i]!=1:
                    shape[i]=1
                    grad=grad.sum(axes=(i,)).reshape(shape)
        return grad                       # 利用Tensor的沿某个维度sum的函数

        # input_shape = node.inputs[0].shape          # 输入维度
        # input_shape_len = len(input_shape) - 1      # 输入维度数目-1。 输入的最后一个维度（当前）
        # self.reduce_dim = []
        # broadcast后的shape从右往左遍历
        # (1)  -> (2,3,2)     记录idx(0,1,2),都不相同  sum
        # （4） ->  （3，4）    记录idx(0)  sum
        #  (1,4) -> (3,4)     记录idx(0)  sum成（4），再reshape成（1.4）
        # （2，1，3） ->  (4,2,5,3)    (倒数第2的1拓展了。第一维拓展了。这2维需要被reduce_sum成1，再被reshape成原始形状) 记录idx(0,2)
        # （2，3） ->  (2,5,3)
        # 所以需要找到所有被拓展了的维度。从最右边开始。
        # 这些维度的值被聚合。sum。 如果input维度小。左边多出来的维度也一样被记录下来，一起被sum_reduce.
        # reduce后数目相等了，再reshape会orig原来的shape(比如某些维度值是1)
        # for idx in range(len(out_grad.shape) - 1, -1, -1):  # 从最后一维开始，到0.
        #     if input_shape_len < 0:
        #         self.reduce_dim.append(idx)                   # 如果input的shape小。剩下多拓展出来的维度也记录下来。最后都被reduce。（通过sum）
        #         continue
        #     # 否则取broadcast后的dim，和input的dim
        #     broadcast_dim_size = self.shape[idx]               # 新shape的最后某维
        #     input_dim_size = input_shape[input_shape_len]      # 和输入的最后
        #
        #     # 比较是否相等，如果不等，说明发生了broadcast，需要将当前dim添加到reduce_dim
        #     if broadcast_dim_size != input_dim_size:
        #         self.reduce_dim.append(idx)                   # 新维度的维度idx是broadcast以后才有的，需要先reduce_sum成1
        #     input_shape_len -= 1
        #
        # # 最后对这些维度reduce_sum. 且reshape保证与原始输入shape一致
        # return reshape(summation(out_grad, tuple(self.reduce_dim)), input_shape)

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

def myreshape(out_grad,axes,wanted_shape):
    # wanted_shape     原始维度 （2，3，4，5）
    # reduced_shape   中间某些维缺失后的维度   （2，5） / 或者（2，）  或者scalar   或者（3，5）.
    #                 需要把这个，拓展到整个X维度
    reduced_shape=list(out_grad.shape)
    if len(reduced_shape) == 0:
        return out_grad.broadcast_to(wanted_shape)  # 全部sum成一个数。out_grad底层是一个scalar Tensor. 直接拓展
    else:  # 现在至少是一维。（1，）（2,）  等。原来是很多维。可能在前，可能在后 比如（1，2，3）/ (2,3,4)/ (3,4,2)  -> (2,)
        # out_grad的维度， 但需要把这个，首先拓展到整个X维度
        # 如果axes是-1，也一样。插到最后一维
        if len(axes) == 1 and axes[0] == -1:  # 原始维度 （2，3，4，5）  现在（2，3，4）变成(2,3,4,1)
            reduced_shape.append(1)
            return out_grad.reshape(reduced_shape).broadcast_to(wanted_shape)
        else:
            for i in axes:
                reduced_shape.insert(i, 1)
        return out_grad.reshape(reduced_shape).broadcast_to(wanted_shape)  # 本质上只需要把(2,1,1,5) -> 拓展成(2,3,4,5)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(axes,int):
            self.axes=tuple([axes,])

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return  a.sum(self.axes) #array_api.sum(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 对求了sum的维度，梯度是1. 但不论沿着什么维度。其实梯度都是1。   dy/dx=1。但dy怎么传呢  dx（m,n） -> dy(m,)  dy拓展到dx大小
        orig_shape = node.inputs[0].shape  # 原始维度 （2，3，4，5）
        sumed_shape = list(out_grad.shape)  # out_grad的维度 是中间某些维缺失后的维度   （2，5） / 或者（2，）  或者scalar   或者（3，5）.
        # 但需要把这个，首先拓展到整个X维度
        # print(sumed_shape)
        # print(orig_shape)    # 可以打印前后对比
        # print(self.axes)

        if len(sumed_shape) == 0 or self.axes == None:
            return out_grad.broadcast_to(orig_shape)  # 全部sum成一个数。out_grad底层是一个scalar Tensor. 直接拓展
        else:  # 现在至少是一维。（1，）（2,）  等。原来是很多维。可能在前，可能在后 比如（1，2，3）/ (2,3,4)/ (3,4,2)/(3,2)  -> (2,)
            # out_grad的维度， 但需要把这个，首先拓展到整个X维度
            # 如果axes是-1，也一样。插到最后一维
            if len(self.axes) == 1 and self.axes[0] == -1:  # 原始维度 （2，3，4，5）  现在（2，3，4）变成(2,3,4,1)
                sumed_shape.append(1)
                # return out_grad.reshape(sumed_shape).broadcast_to(orig_shape)
            else:
                for i in self.axes:
                    sumed_shape.insert(i, 1)
        # return out_grad.reshape(sumed_shape)* array_api.ones(orig_shape, dtype=array_api.float32)  # 借助np的广播机制。把sum_shapee广播到orig_shape上
        return out_grad.reshape(sumed_shape).broadcast_to(
            orig_shape)  # 本质上只需要把(2,1,1,5) -> 拓展成(2,3,4,5) / # 原来（2，3） 现在变成（3），axis=(0,).

        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        #return array_api.matmul(a, b)
        return a.__matmul__(b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A,B=node.inputs
        A_shape=A.shape  # 6,5,2
        B_shape=B.shape  # 2,4     多维。不一定是2维.但需要A的最后一维，和B的倒数第二维相同
        # 当时每个y是一个A乘一个B。（如果AB维度不等，比如B少。但B被拿来多乘了6次。）
        # dy:(6,5,4)   b.T (4.2)   每个A：（6,5,2）
        dy_dA= out_grad.matmul(B.transpose())  # (...,m,n)   均是自定义的Tensor前向乘法。输出Tensor
        # A.t:(6,2,5)     dy:(6,5，4)   -> B(6,2,4) 应该再sum一下。沿着B最终的梯度方向  (否则B原始维度就是这样的话，就不用变了。当初乘的时候就是每个矩阵分别做的贡献)
        dy_dB= A.transpose().matmul(out_grad)  # (...,n,k)   默认只转置最后2维

        lenA=len(A_shape)
        lenB=len(B_shape)
        if (lenA!=lenB):
            if (lenA>lenB):  # A长，B短。当初乘的时候，多次复用了B的小矩阵，去乘A多的矩阵。因此B的几次使用，梯度要聚合下  （等长的话没有复用）
                dy_dB=dy_dB.sum(axes=tuple(range(lenA-lenB)))
            else:
                dy_dA=dy_dA.sum(axes=tuple(range(lenB - lenA)))  # A:(6,2,5)  B(2,4)   多一个维度。只需要聚合第一维

        return dy_dA,dy_dB
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # y=logx    dy/dx=1/x
        return out_grad/ (node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # y=ex   dy/dx= ex =y
        x=node.inputs[0]
        return out_grad*exp(x)  # Tensor
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        #b=array_api.where(a>0,a,0)
        b=array_api.maximum(a,0)
        return b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node:Tensor):
        ### BEGIN YOUR SOLUTION
        # out_grad: 原来的输出node中，非0的部分梯度回传，梯度值不变，其他部分设置成0，不回传梯度。
        # node,是relu输出的Tensor本身，已经是0和其他值(>0)了
        # I{x>0} * out_grad： 用一个mask,把out_grad中，node大于0的部分设成不变
        # x  ( x> 0)   dy/dx=  1
        # 0                    0
        # node是该Tensor本身.
        mask = Tensor(node.numpy()>0,device=node.device,dtype=node.dtype)    # 对应mask，把node 大于0的部分设成1
        #mask=Tensor(array_api.where(y>0,1,0))  # y大于0的地方都置成1。y其他地方都是0
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # log (sum_i (exp zi))    expZi数值上不稳定，容易溢出。 除以最大的值
        # =      log(    (sum_i exp (zi- maxz))  *  exp maxz  )
        # =  log(sum_i exp (zi- maxz))   +  maxz                 每行
        maxz=Z.max(axis=self.axes,keepdims=True)      # (1,2,3,4) -> axis=(0,2)  -> (1,2,1,4)
        sumexp_minus_maz= array_api.summation(array_api.exp(Z-maxz.broadcast_to(Z.shape)),axis=self.axes)      # (2,4)
        return array_api.log(sumexp_minus_maz)+maxz.reshape(sumexp_minus_maz.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # y=log(sum(expx))
        # dy/dx= dlog/dsum   * dsum/dz (z=expx)  * dz/dx
        #      = 1/ sum(expx) (维度同y) * 1(reshape成orig_shape,同summation的grad)  * z(维度同x)
        #      = z/  ( sum(z) (维度同y) *1(reshape成Z))
        #      = softmax(x)  ,reshape成z
        x=node.inputs[0]
        maxx= Tensor(x.realize_cached_data().max(axis=self.axes,keepdims=True),device=x.device,dtype=x.dtype)
        x=x-  broadcast_to(maxx,x.shape)                                     # 分子分母都减去最大值
        expx=exp(x)                                   # 新的z:   exp(x-max_x),防止数值不稳定
        sum=summation(expx,self.axes)                 # sum(expx)
        sumz_grads=divide(out_grad,sum)                # 1/sum(z) (维度同y),乘上同纬度的y_grad
        reshaped_sum=myreshape(sumz_grads, self.axes, wanted_shape=x.shape)
        softmax=multiply(expx,reshaped_sum)   # z/sum(z)   其中sum(z)拓展到原来大小，通过reshape+broadcast
        return softmax
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # dy/dx=(1-tanh**2)
        return (1-tanh(node.inputs[0])**2)  * out_grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)

# 类似concat,但多出来一维（stack的那维度）。变成一个维度更高的矩阵。原来的每个矩阵维度不变。
class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        不是单纯concat,而是新增了一个维度。比如3个5*5的矩阵，stack成 3*5*5 （沿着首维stack）./  5*3*5 （沿着维度1）
        """
        self.axis = axis

    def compute(self, args:tuple[NDArray]):
        ### BEGIN YOUR SOLUTION
        # 输入是一个TensorTuple
        # args：是该TensorTuple的cache_date. 一个tuple。所有的inputs Tensor（的底层），拼成的一个tuple
        n_item=len(args)
        shape=list(args[0].shape)  # 原始每个底层NDarray的大小  (3,5)
        n_dim=len(shape)
        axis=self.axis

        # 按维度，先分配一个新NDarray。再把原来的每个元素.设置到对应位置
        new_shape=shape
        new_shape.insert(self.axis,n_item)    # 2个(3,5) ->  (3,2,5)
        concat_array=NDArray.make(shape=new_shape,device=args[0].device)   # 新array

        slice_tuple=[slice(0,i,1)  for i in new_shape]   # 每一维的slice.  除了新维，其他维都不变
        for i in range(n_item): # 设置该维的每个元素
            s=slice(i,i+1,1)
            slice_tuple[axis]=s   # 设置该维.
            concat_array[tuple(slice_tuple)]=args[i]  # NDarray.__setitem__. value: 一个NDarray(底层会先compact)
        return concat_array     # 作为TupleTensor生成的一个Tensor,底层的数据
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 梯度是1：
        # out_grad：是n个array拼成的一个Tensor。对每个inputs贡献的梯度是1。
        # input是一个TupleTensor，每部分梯度是out_grad中对应部分的梯度

        # n_item=out_grad.shape[self.axis]  # 输入的Tensor数目
        # input_shape=node.inputs[0].realize_cached_data()[0].shape  # 该tensorTuple中，每个元素的大小
        # input_grads=[]
        # slice_tuple = [slice(0, i, 1) for i in out_grad.shape]
        # for i in range(n_item):
        #     s = slice(i, i + 1, 1)
        #     slice_tuple[self.axis]=s   # 设置该维.
        #     # getitem后，每个reshape成原大小
        #     input_grads.append(Tensor(out_grad.realize_cached_data()[tuple(slice_tuple)].compact().reshape(input_shape)))  # 每一份grad,对应的内存NDarray

        # 输出是一个tuple,含n个tensor
        # return make_tuple(*input_grads) # TensorTuple
        return split(out_grad,self.axis)  # 把该Tensor,拆成n个tensor组成的TensorTuple
        #return input_grads
        ### END YOUR SOLUTION


def stack(args, axis):
    # 输入是n个Tensor,输出单个Tensor
    return Stack(axis)(make_tuple(*args))  # 把所有的输入Tensor,拼成一个TupleTensor。再传给Stack
                                           # 会调用TensorOp的call,把该TupleTensor传进去。输出一个大Tensor

# stack的反向操作。把一个Tensor拆成n个。沿着指定维度。 5，2,5  -》 2个(5,5)
# input是一个大的Tensor。 out是一个TensorTuple，内部数据是一系列小tensor构成的tuple
class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n_item=A.shape[self.axis]  # 要拆的Tensor数目
        out_shape=[A.shape[i] for i in range(len(A.shape)) if i!=self.axis]  # 拆分后每个元素的大小
        outs=[]
        slice_tuple = [slice(0, i, 1) for i in A.shape]
        for i in range(n_item):
            s = slice(i, i + 1, 1)
            slice_tuple[self.axis]=s   # 设置该维.
            # getitem后，每个reshape成原大小
            outs.append(A[tuple(slice_tuple)].compact().reshape(out_shape))  # 每一份,对应的内存NDarray
        return tuple(outs)  # tupleTensor的底层
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # n个grad,到一个grad => stack
        # out_grad是一个tupleTensor。inputs是一个大Tensor. stack起来
        return stack(out_grad,self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    # a是
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # out= a.flip(axes)  1234 -> 4321
        # 4的梯度，對應原來1的梯度。也flip
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


# 空洞化，增加感受野： [[1,2],    ==> [[1,0,2,0],
#                    [3,4]]         [0,.... ]
#                                   [3,0,4,0]
#                                   [0,.... ]
# 当10个(32, 32, 8)的img, 用K=1的filter卷积。会得到(32,32)的img
#       stides=2时，输出(H2=16,W2=16) ：[1 2 3 4 5 6] -> [1 3 5]
#       BP时，要由（16，16） -> (32,32)。只有原数据的135处有梯度，24处没算，对应梯度是0
#                 [d1 d3 d5]  -> 可以用Dilate，转为 [d1 0 d3 0 d5 0]。 y轴同，前向时相同步长，bp时相同dilate
class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes             # 只有某些维度要进行空洞化
        self.dilation = dilation     # 每个元素用几个0空洞化

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # [1,2,3] -> [1 0 0 2 0 0 3 0 0] Dilate=2
        orig_shape=a.shape
        new_shape=list(orig_shape)
        new_slices=[ slice(0,orig_shape[i],1) for i in range(len(orig_shape))]
        for i in self.axes:
            new_shape[i]=orig_shape[i]* (self.dilation+1)  # shape= shape* (dilation+1)
            new_slices[i]=slice(0,new_shape[i]-self.dilation,self.dilation+1)  # step=dilation+1  每隔n步，设上原来的值
        dilate_array=NDArray.make(new_shape,device=a.device)
        dilate_array.fill(0)
        dilate_array[tuple(new_slices)]=a  #
        return dilate_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 输入[1,2,3] -> [1,0,2,0,3,0]
        # out_grad中，空洞位置无效，不往前传
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 可以直接getitem，没有的维度，就和原来保持一致，有过dia的维度，只抽取有效数据
        slices=[ slice(0,a.shape[i],1) for i in range(len(a.shape))]
        for i in self.axes:
            slices[i]=slice(0,a.shape[i]-self.dilation,self.dilation+1)
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 输入 [1,0,2,0,3,0] -> [1,2,3]
        # out_grad中，扩大到未使用的位置
        dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A:NDArray, B:NDArray):
        ### BEGIN YOUR SOLUTION
        # A:NHWC
        # B: filter (k, k, input_channels, output_channels).
        if self.padding!=0:
            A=A.pad(axes=( (0, 0), (self.padding, self.padding), (self.padding, self.padding),(0, 0)))
        N,H1,W1,Cin=A.shape
        Ns,Hs,Ws,Cin_s=A.strides   # 初始的strides
        k1,k2,_,C_out=B.shape

        # im2col: get patches of A.
        #         一共可以得到[H2,W2]个patchs,对应矩阵大小是[H2,W2,C_in,k*k]
        # 直接用A原来的数据,获得该patch矩阵
        H2=  (H1 -k1)//self.stride + 1
        W2 = (W1 -k2)//self.stride +1
        new_shape=[N,H2,W2,k1,k2,Cin]
        new_strides=[Ns,self.stride*Hs,self.stride*Ws,
                    Hs,Ws,Cin_s]       # 每个patch间距离不变，是原来HW图中内存间距。 如patch中相邻元素，间距是原来W维的间距Ws
                                        # 相邻Channel对应的patch,间距同原来相邻channel
                                        # H2W2,不同的大patch间，间隔是strides* Hs,strides* Ws
        im2col=A.as_strided(shape=new_shape,strides=new_strides).compact().reshape((N*H2*W2,Cin*k1*k2))

        # weight，转成大矩阵形式 [cin*k1*k2,cout]
        filter=B.compact().reshape((Cin*k1*k2,C_out))

        # conv=im2col * filter
        self.im2col=im2col
        self.filter=B
        return (im2col@filter).reshape((N,H2,W2,C_out))   # [H2,W2,C_in,k*k] * [cin*k1*k2,cout] = [N,H2W2,cout]
        ### END YOUR SOLUTION

    def gradient(self, out_grad:Tensor, node:Tensor):
        ### BEGIN YOUR SOLUTION
        # input: A,B
        #  out= im2col * filter
        #  dw= im2col.T (cin*f,NH2W2) * out (NH2W2,cout)
        N, H2, W2, Cout = out_grad.shape
        k1, k2, Cin, _ = node.inputs[1].shape
        dw=self.im2col.permute((1,0)) @ (out_grad.realize_cached_data().compact().reshape((N*H2*W2,Cout)))
        dw= Tensor(dw.compact().reshape((k1,k2,Cin,Cout)),device=out_grad.device)

        # 可以把卷积看做X不变，把filter变成适当形式的W（线性变化）后，对应的矩阵乘；
        # Y= WX        （x是原始X）（W是修改过的filter）
        # dX=W.T * dY   (Y是卷积结果， 因此dX也是卷积结果，只不过对应的卷积核是W.T。对应各维度flip后的filter)
        # 因此需要对dy做卷积。所用的卷积核是filter flip后的kernel

        # 输入out:(N,H2,W2,cout)   相当于in_channel是cout
        # filter：permute成（k1,k2,C_out,Cin）
        new_filter=Tensor(self.filter.flip(axes=(0,1)).permute((0,1,3,2)),device=out_grad.device)

        # 如果之前有strides,需要先对y空洞化，某些input元素未提供梯度
        out_grad=dilate(out_grad,axes=(1,2),dilation=self.stride-1)
        # 需要把结果，变成所需的维度H1,W1  （N,H1,W1,cin）,
        # H2=(H1-k)//s +1  需要的输出维度H1=H2+k-1   因此需要输入的output的维度H_in=H1-1+k = H2+2k-2  需要padding 上下各k-1
        padH=k1-1
        padW=k2-1
        out_grad.cached_data=out_grad.cached_data.pad(axes = ( (0, 0), (padH, padH), (padW, padW),(0,0)))


        dx= conv(out_grad, new_filter,stride=1,padding=0)  # out:(N,H2,W2,cout)  卷积 filter(k1,k2,C_out,cin)

        # 如果之前有padding，得到原图后，去掉外圈padding   (N,H1,W1,cin)
        slices=[slice(0,dx.shape[i],1) for i in range(len(dx.shape))]
        slices[1]=slice(self.padding,dx.shape[1]-self.padding,1)  # H1=H1-p
        slices[2] =slice(self.padding,dx.shape[2]-self.padding,1) # W1=W1-p
        dx.cached_data = dx.cached_data[tuple(slices)]

        return dx,dw
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



