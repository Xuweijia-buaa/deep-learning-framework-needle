"""Core data structures."""
import needle
from typing import List, Optional, NamedTuple, Tuple, Union
from collections import namedtuple
import numpy
from needle import init

# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

# array_api：底层包本身。含全部ndarray.py和对应的so(通过NDarray间接访问),作为该包
# array_api.NDArray 是对底层数据结构的一个封装。该类含shape,strides等属性，设备对象（封装对应的.so包）
#           NDArray可以作为Tenor的底层数据，进行数据操作，封装更底层的不同设备上的内存操作。
# array_api.array函数：实际上调用NDArray(a, device=device)，生成一个NDarray。大小同a
from .backend_selection import Device, array_api, NDArray, default_device

# tensorOp类。对象被存在每个输出Tensor里。用来根据输入算输出cached_data
class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    # 根据i 对应的梯度Tensor vi，和原始图中的该节点Tensor， 计算该节点对各个输入节点op k贡献的梯度vki Tensor
    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    # 给定前一个节点的grad,计算该op反向的输出（调用gradient函数，计算该op对输入的其他节点贡献的梯度）（Tensor）
    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """ Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

# 可以用()调用。重载了(),使得op(a,b),变成Tensor.make_from_op(self,a,b)。生成且返回新的Tensor，input是a,b, op是op
class TensorOp(Op):
    """ Op class specialized to output tensors, will be alterate subclasses for other structures """

    def __call__(self, *args):
        # 实例化op对象，并调用op的时候，把args传进去。args也是由原生值，封装的tensor。 值存在Tensor的cached_data里
        # 会根据这个op和输入，计算输出的Tensor，返回  （存对应的inputs和op,有可能算cache_data,是tensor对应op的具体输出值）
        return Tensor.make_from_op(self, args)

# 针对TensorTuple的op。
class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    # 不调用原来op的__call__，而是调用TensorTuple的make_from_op。通过args,op,生成
    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)  # 类方法。可以根据类cls，生成对应对象


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # 如果调用后。没有调该函数(lazy mode)。每个输出Tensor,只包含对应的op和所有输入。没计算输出
        # 但eagermode下,会调用该函数，为输出Tensor计算cached_data。相当于该Tensor本次的输出值

        # 计算该tensor内的输出值 (forward pass,调用op的compute)
        # 如果cached_data是NDarray，传入op.compute中计算，用的就是Ndarray封装的函数。底层可能用的.so中的函数
        if self.cached_data is not None:         # avoid recomputation
            return self.cached_data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]  # inputs可以是原生Tensor,传入对应的缓存数据做计算。
        )
        self.cached_data
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],    # input可以是原生Tensor,也可以是自己的各类NDarray （前向计算时）
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs             # 可以只指定input,op. 此时不实际计算op的输出 (调用op()(a,b),通过op初始化,输出一个Tenso对象r时)
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):# 定值。 直接由data产生。值放在cached_data里。 （同op compute输出的那种tensor，对应的值）
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    # 类方法
    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        # 对于一个Tensor，给定op和input
        # 如果requires_grad==False, 直接生成一个const Tensor,（没有op和input，不和其他节点有连接关系）
        # 如果requires_grad==True,  在计算图里（通过和inputs的联系）。

        # cls在python中表示类本身
        value = cls.__new__(cls)                 # new一个该类对象。如果是Tensor类调用，new一个Tensor类
                                                 #                如果是TensorTuple调用，new一个TensorTuple类
        value._init(op, inputs)                  # 指定对应op和input

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()         # 非lazy模式。计算前向输出（即本op的输出，cache_data） TensorTuple对象同，调用对象compute
        return value

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy() if not isinstance(data, tuple) else [x.numpy() for x in data]


### Tuple类的对象。同样调用里边op的compute，计算cache_data. input一般是n个Tensor
class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()  # 输入是多个Tensor. 输出是这些Tensor对应的tuple
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)  # 第i个Tensor

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    # 加一个TensorTuple(对应位置相加)。仍返回TensorTuple
    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: "Tensor"  # 是该Tensor带的grad,也是一个Tensor。对应反向图中的节点（op）。是由反向图中的输入grad，根据op的gradient函数更新的（输入Tensor op,out_grad和该Tensor）

    def __init__(   # 直接输入一个值array，初始化一个Tensor： 如Tensor([0], dtype="float32")。 可以初始化好值（cached_data）
        self,
        array,
        *,
        device: Optional[Device] = None,  # 可以是ndl.cpu(), ndl.cuda().
                                          # 是一个BackendDevice("cuda", ndarray_backend_cuda) 对象。封装对应.so
                                          # 配合底层数据NDarray使用。NDarray据此，调用对应的Device中的函数
        dtype="float32",
        requires_grad=True,              # 如果requires_grad==False, 直接生成一个const Tensor,（没有op和input，不和其他节点有连接关系）
                                         # 如果requires_grad==True,  在计算图里（通过和inputs的联系）。
        **kwargs
    ):
        # 输入一个op的array，可以是任意类型的值。
        # 可以是numpy,可以是封装底层内容的Array类。也可以是其他op输出的Tensor

        # x = ndl.Tensor([1,2,3], device=ndl.cpu(), dtype="float32")
        # y = ndl.Tensor([2, 3, 5], device=ndl.cuda(), dtype="float32")
        if isinstance(array, Tensor):  # Tensor间互相初始化。不同设备/不同精度
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:  # 设置的精度。设备，就是传入array的原始属性。
                cached_data = array.realize_cached_data()        # 生成的新Tensor,仍用array中的cache_data
            else:
                # 设备，精度不同。用numpy做转换。
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()    # 底层device的封装对象。类实现在backend_ndarray里,封装着.so
            # 返回一个NDArray(a, device=device)对象。
            # 作为底层数据。含shape等属性与_handle. 之后op操作都基于NDarray
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        # 生成新Tensor。
        self._init(
            None,                      # op是None, input是[]. 只有值：是传入的cached_data
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    # 传入的数据，转成NDArray(a, device=device)对象。（在底层分配了内存，根据a的shape,strides(默认设置成行主序连续的)）
    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        # array函数：实际上调用NDArray(a, device=device)，生成一个NDarray。大小同a
        return array_api.array(numpy_array, device=device, dtype=dtype)

    # 输入是TensorOp和input(input已经是原始数据封装成的Tensor了。才能传给op.比如constTensor)
    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        # 对于一个Tensor，给定op和input
        # 如果requires_grad==False, 直接生成一个const Tensor,（没有op和input，不和其他节点有连接关系）
        # 如果requires_grad==True,  在计算图里（通过和inputs的联系）。
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)         # 用op作用于输出，输出该op对应的Tensor（首先存op和对应inputs）
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()   # 一个没有inputs的const Tensor，和其他节点无关。bp涉及不到。

            tensor.realize_cached_data() # eager mode，会计算好每个op的输出值。cached_data,放输出tensor里。 否则用到其值时才触发计算
                                         # 设置该Tensor的底层数据。通过op.compute.
                                         # 如果底层是NDarray，调用的op中的操作符，用的是Ndarray封装的，底层可能用的.so中的
        return tensor

    # 把原生data变成一个Tensor。值放在Tensor的cached_data里，可以从来传给op做计算
    # op只接收Tensor输入
    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(                    # const: 同直接用常量初始化一个Tensor。 只有取值cached_data。没有其他如op等
            None,                        # op是None
            [],                          # inputs是空
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    # 对应.data，相当于一个getter
    @property
    def data(self):
        return self.detach()

    # .data的setter。用于A.data=value的情况下调用，把value的底层Array,设置到cached_data中。
    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    # 一个新Tensor.只是底层Array相同。但不在图里。
    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return default_device()
        return data.device

    # 一般是loss Tensor调用：loss.backward()
    def backward(self, out_grad=None):
        #  loss(Tensor)调用。默认梯度是全1. out_grad==None
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)  # out_grad==None. lossTensor调用。
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    # 重载了Tensor加法
    # Tensor间的操作，都基于ops
    # x = ndl.Tensor([1, 2, 3], device=ndl.cpu(), dtype="float32")  cached_data是NDarray
    # y = ndl.Tensor([2, 3, 5], device=ndl.cpu(), dtype="float32")
    # z = x + y
    # 该op前向计算，用的是op.compute计算x,y的底层cachedata(NDarray)。
    # 返回结果也是NDarray,分配了新的底层内存，并通过Ndarray封装的device对象中的.so做了计算
    # 最终的Tensor的cache_data，返回的是该NDarray。NDarray对应方法中，会先new一个NDarray(且分配内存)。再调用底层.so，填充对应的内存(NDarray中的__handle__)。
    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other) # tensor之间的按位置加。会调用对应的TensorOp，返回一个新Tensor
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        ### BEGIN YOUR SOLUTION
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return needle.ops.PowerScalar(other)(self)  # 本质上调用额度PowerScalar Op
        ### END YOUR SOLUTION

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)
       
    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(needle.ops.Negate()(self), other)
        else:
            return needle.ops.AddScalar(other)(-self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    # Tensor操作，封装底层ops。底层ops用NDarray实现前后向计算
    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def split(self,axis):
        return needle.ops.Split(axis)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    # def __getitem__(self, index: int):
    #     return needle.ops.tuple_get_item(self, index)  # 第i个Tensor

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

from functools import reduce
def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    # 一般是loss tensor调用时触发：loss.backward()
    # output_tensor：loss Tensor.
    # out_grad：loss Tensor对应的输入梯度。默认是1。 loss=z
    # 给定最后一个节点的out_grad, 计算之前的每个节点的grad.存在每个Tensor的.grad字段。
    #                          并将grad计算往前传播，直接遍历完（遍历顺序已经确定，就是拓扑排序的反向顺序。grad的依赖更新顺序是对的）
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}    #  保存其他节点对该tensor贡献的所有梯度。
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]          # （初始是传入的loss,全1）
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.

    # 用我们写的后续遍历dfs节点（给定终点），找到图的前向遍历顺序。再直接reverse,反向
    # 输入的output_tensor是最后一个节点。
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    # 严格按对的访问顺序。访问完2，1，node0需要的缓存就算好了. 放在dict[0]里，是1,2对0贡献的梯度
    for node in reverse_topo_order:
        # 算node的梯度时，之前各依赖节点对node的贡献的梯度已经算好了。
        grads_from_last= node_to_output_grads_list[node]

        # 先计算该node本身的梯度（Tensor),并更新到该Tensor的grad域。之后用来做w=w-lr*g。  w是该Tensor的原始输出值。g是对应的总的梯度

        node.grad= reduce(lambda g1,g2:g1+g2 , grads_from_last)  # 一个Tensor   如果有多个，浮点加法的顺序不同，结果的精度不同
        if len(node.inputs) ==0:   # 没有了。某一分支到最开始的节点了。这个节点自己不用做贡献，处理完了
            continue               # 但可能还有一些并行的其他节点需要访问。这里每个节点都需要遍历，至少处理一次。
                                   # 只是该节点不用做贡献，该点本身处理结束（不是return，是continue）

        # 否则更新该节点对所有输入节点的贡献：
        op=node.op
                                                                  # 通过Tensor对应的op，计算反向的往前传的梯度 （通过op的gradient函数）
        contributed_grads=op.gradient_as_tuple(node.grad,node)    # 用一个封装的函数调用。把返回的一些单个grad返回成tuple,对应input数目
        assert len(contributed_grads)==len(node.inputs)

        for i,input_node in enumerate(node.inputs):
            if input_node in node_to_output_grads_list:
                node_to_output_grads_list[input_node].append(contributed_grads[i]) # 不是直接更新到input_node里。而是先更新到dict[k]里.防止有其他节点对k做贡献
            else:
                node_to_output_grads_list[input_node]=[contributed_grads[i]]
    # loss.backward后，从Loss Tensor自己（loss==output_tensor）开始，到最开始的输入节点。都应该有grad了
    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    visited=set()
    topo_order=[]
    for node_tensor in node_list:  # 所有没有出边的终点。往前回溯。找所有依赖.  如果有多于一个终点。终点1顺序定了。再定终点2的。加到同一个res里
        topo_sort_dfs(node_tensor, visited, topo_order)   # 不考虑环

    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    if node in visited:             # 访问过的,直接跳过。遍历下一个分支
        return;

    # if len(node.inputs)==0:         # 到了叶子节点（最开始的任务，没有依赖，直接加入到res. 之后依赖他的才能后序加入）
    #     topo_order.append(node)
    #     return;

    for child in node.inputs:   # node的所有前向依赖。作为孩子节点
           topo_sort_dfs(child, visited, topo_order)

    visited.add(node)
    topo_order.append(node)     # 有依赖的节点。全部依赖都访问完了，加入了res. 才后序加入
                                # （或者没有依赖）,直接不进dfs循环。直接加入res.
    ### END YOUR SOLUTION


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
