import operator
import math
from functools import reduce
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu

# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod              # 封装底层包。     如果是np,底层包是ndarray_backend_numpy
                                    #                    gpu,底层包是ndarray_backend_cuda
                                    #                    cpu，底层包是自己实现的ndarray_backend_numpy
                                     # 底层数据是里边的Array对象，负责具体的一些涉及内存的操作，比如矩阵乘

    def __eq__(self, other):
        return self.name == other.name

    # 输出某个实例化对象时，其调用的就是该对象的 __repr__() 方法，输出的是该方法的返回值。
    # 如打印cuda()
    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)  # 模块中是否有该属性。 mod.f.  设备中函数，都是通过该方式调用

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        # device的one_hot,底层都是基于np生成。之后转到对应device上。
        # 然后给定i,得到这些i对应的one-hot向量。 i可以是一个slice/list
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():# 底层是自己实现的GPU上的操作
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)  # 有这个.so就可以测试 cuda了
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]

# 尽管是矩阵，但底层是连续内存，通过shape, strides映射成n维矩阵，并进行索引
# 最底层操作基于c++.但转置等，braodcast等0拷贝内存操作，完全可以通过操作shape,stride完成.而不需要改变底层数据
class NDArray:
    """A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """
    # 初始化一个NDarray
    # 输入是NDarray: （指定了不同设备）
    #               把设备上的底层数据_handle, 复制到numpy数据中（调用设备上的to_numpy函数）
    #               再new一个NDarray，把这个np内存连续后拷到新设备上（连续内存）  （NDArray(self.numpy(), device=device)）
    # 输入np(list等转np):
    #  new一个Ndarray，并把该np内存连续化，设置到指定/默认device开辟的连续内存中。shape不变,strides连续,odffset=0
    #        1 make:在新设备上用shape，分配连续内存(array.device.Array(prod(shape)))
    #        2 调用设备上的from_numpy函数，把原始np内存(变成连续内存后）复制到NDarray刚刚开辟的设备内存中。（连续）
    def __init__(self, other, device=None):
        """ Create by copying another NDArray, or from numpy """
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy    把该array移到指定设备，按原始大小，strides
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)             # make:按给定的shape,device，在device上开辟了一份连续内存。（strides连续）
            array.device.from_numpy(np.ascontiguousarray(other), array._handle) # 把原始矩阵内存(变成连续内存后）复制到设备中。对应NDarray刚刚开辟的连续内存
                                                                                # 调用设备函数，复制原始np内存到设备Array  source -> array._handle
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)                     # 用other的设置自己的
            self._init(array)

    def _init(self, other):
        self._shape = other._shape           # 该矩阵shape
        self._strides = other._strides       # 每个维度2元素在内存中的间隔。 2行元素上下之间内存间隔(维度0)，2列元素上下之间内存间隔（维度1）,...
        self._offset = other._offset         # 该矩阵在底层flat array中的起始位置. 作为该矩阵真正位置。从这里算矩阵的第一个元素
        self._device = other._device         # cpu/gpu，BackendDevice对象
        self._handle = other._handle         # 底层数据：underlying memory of the array。是Device里的Array对象
                                             # 在make创建一个NDarray时被设置：in NDArray.make function

    @staticmethod
    def compact_strides(shape):    # 如果矩阵完全C连续，给定shape,输出对应的strides
        """ Utility function to compute compact strides """
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):   # [1,n]. 倒着乘，[-1,-n] 共n个维度
            res.append(stride)
            stride *= shape[-i]              # s[i]= s[i+1] * shape[i+1]  倒着乘，从i[-1]。到i[-n]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        # 新建一个NDarray(new一个)并返回。
        # 如果没有指定底层数据，就会在指定device上新分配一个内存（device上的Array结构，通过pybind11绑定，作为handle），备用。是一维连续内存
        # 如果传入了底层数据（handle Array），不分配新内存。底层数据直接用给定的handle.(只是shape等参数变了)，设定新的shape, strides（未指定默认连续）, offset(默认0)，device
        # 尽量用这个函数。不修改底层内存的情况下，新建共享相同handle的NDarray(逻辑矩阵)
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))  # 本质是device中的Array对象。这时候只有默认初始值
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """ Return number of dimensions. """
        return len(self._shape)

    @property
    def size(self):               # python端才有的，记录ndarray带有strides等信息后的真实size，作为property
        return prod(self._shape)

    def __repr__(self):
        # 输出某个实例化对象时，其调用的就是该对象的__repr__()方法，输出的是该方法的返回值。(jupyter中)
        # 输出的是调用so中to_numpy函数，把底层内存转成新的numpy之后(按属性中的shape,strides，offset),对应numpy数组的string格式。
        # 在print该对象时，（含jupyter）,调用该函数，展示返回值
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value):
        """ Fill (in place) with a constant value. """
        self._device.fill(self._handle, value)

    def to(self, device):
        """ Convert between devices, using to/from numpy calls as the unifying bridge. """
        if device == self.device:
            return self
        else:  # 把设备上的底层数据_handle, 复制到numpy数据中（to_numpy） . 再new一个NDarray
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """ convert to a numpy array """
        # 把设备上的底层数据_handle, 复制到numpy数据(pybind数组)。 设置上shape, strides, offset大小的
        # pybind重新分配内存。分配后可能会改变strides
        return self.device.to_numpy(                   # 先转成numpy（用相同底层数据）
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):    # 验证矩阵本身的strides是否满足连续要求
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)  # 矩阵本身的strides，是否满足C连续矩阵给定shape时对应的strides
            and prod(self.shape) == self._handle.size           # 底层数据的size，等于返回的shape
        )

    def compact(self):      # new一个底层内存连续的新NDarray（在本设备上）
        """ Convert a matrix to be compact """
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)  # new一个NDarray,在底层新分配一个连续内存
            self.device.compact(                                # 调用设备端的compact函数，传入本身内存(逻辑矩阵对应内存)。复制到新NDarray分配好的内存中。直接改变了out本身
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out # 返回这个新的NDarray，已经是新的了

    def as_strided(self, shape, strides):
        """ Restride the matrix without copying memory. """
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        ### BEGIN YOUR SOLUTION
        if self.is_compact()!=True:
            self=self.compact()
        assert self.is_compact()
        #print("self.shape",self.shape)
        #print("new_shape", new_shape)
        assert prod(self.shape) == prod(new_shape) # 元素数目相同

        # 原来的所有元素。按照新的shape. 按顺序重新装到新的shape里。
        # 底层数据不变， offset不变
        # 1，2，3，4，5，6
        # 原来是(2,3). stride是（3，1）
        # 1 2 3
        # 4 5 6  hw

        # 现在是（3，2），或者（6，1），或者（1，6）。 shape是新shape,不管原来每一维如何。重新看待新元素
        # reshape后，原始的维度不管了。只是重新排布元素
        # tensor([[1., 2.],
        #         [3., 4.],
        #         [5., 6.]])

        # 不像permute只是视角变了。permute后是
        # tensor([[1., 4.],
        #         [2., 5.],
        #         [3., 6.]])       wh
        # 每个维度本身dim都不变。只是每维排列顺序换了。常用于（B,T,d）与（B,d,T）等转化。只是看待视角变了

        # 每个维度stride（不考虑step）,重新考虑:
        # 首维元素间的stride是剩余其他所有维度dim的乘积（不考虑原始最后一维有strides）
        # 首维相邻元素1，3之间的内存gap是2.  (3,2)中的2
        #        元素12间gap是1

        # 如果原来元素1，2间就有gap==2.  14间间隔是s0=3*s1=6  12间间隔s1=2
        # 那么现在：元素13间gap是s0=2*s1=4      最后一维元素12间的gap不变
        #         因为最后一维元素间的gap不变，所以前边每维相邻元素间间隔是该维2元素，中间相隔的元素数目（dim乘积）* 每个元素的内存间隔（是最后一维元素的内存间隔）

        # 如果reshape(6,1):  首维元素间间隔是1 * 原始最后一维的gap s
        # tensor([[1.],
        #         [2.],
        #         [3.],
        #         [4.],
        #         [5.],
        #         [6.]])

        # reshape(1,6):  首维元素间间隔是6 * 原始最后一维的gap s

        offset=self._offset  # offset不变
        last_dimension_gap=self.strides[-1]  # 原始最后一维，元素间间隔。reshape后保持不变
        # 每个stride。是剩余维dimension的乘积，乘最后一维的元素间隔。也是下一维s[i+1]*dim[i+1]
        stride=[]
        l=len(new_shape)
        dict={}                             # 存每一维的内存间隔
        dict[l-1] = last_dimension_gap      # 一般是1.最后一维的内存间隔
        for i in range(l-2,-1,-1):          # 从倒数第二维到第0维，利用缓存的剩余维的累积乘积，计算前一维：
                dict[i] =  new_shape[i+1] * dict[i+1]   # 下一维的元素数目。乘上对应的内存间隔.是该维相邻元素间间隔

        for i in range(l):
            stride.append(dict[i])

        new_stride = tuple(stride)
        return self.make(new_shape, strides=new_stride, handle=self._handle, offset=self._offset,
                         device=self.device)  # 新shape的矩阵，底层用相同内存数据
        ### END YOUR SOLUTION

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        ### BEGIN YOUR SOLUTION
        # 各维度都没有变。但是观察的角度变了。不是按顺序重装
        # [c,H,w]   permute（1,2,0）  第0维放原来的第1维，第1维放原来的第2维,最后放原来的c维  ->  [h,w,c]

        # 底层数据原来是： (c=2,h=3,w=5)
        # [ (3,5), (3,5)] 共30个元素
        # [  c1  |  c2 ]
        # 每一维元素strides：[15,5,1]  不考虑step,stride是剩余n维dim的乘积

        # permute后。 首维放原来的H维。H维不同元素间在内存间的gap不变。
        #            最后一维放原来的C维。C维间不同元素，在内存间的gap也不变
        # 因此stride只是在不同调整了的维之间swap

        # 考虑原来stride的最后一维不是1.原来c维放到最后一维。内存也不变。同样是swap

        # 1，2，3，4，5，6
        # 原来是(2,3). stride是（3，1）
        # tensor([[1., 2., 3.],
        #         [4., 5., 6.]])   hw

        # permute(1,0)交换后:
        # tensor([[1., 4.],
        #         [2., 5.],
        #         [3., 6.]])       wh
        # 现在的首维是原来的第一维，首维元素12之间的内存间隔不变，还是1。 列元素14之间内存间隔是3.
        # 原始维度信息不变，不重新整理这些元素。原始每个维度的大小也不变
        # h维长2，w维长3.  permute后，只是维度顺序变了。
        #                w维长度仍为3，只是原来放第二维，每个元素长3.
        #                            现在放首维，各元素竖着放了。但每一维本身不变。相当于转置
        #

        # 每个维度，stride变成原始维度的，shape变成原始维度的
        assert len(new_axes)==len(self.shape)  # 维度数目不变
        new_stride=[]
        new_shape=[]
        for idx  in range(len(new_axes)): # permute(0,2,1)
            orig_idx=new_axes[idx]        # 第idx维,被替换成了原来的第permute[idx]维。 如当前维度1被替换成了现在的维度2（permute[1]）
            new_stride.append(self._strides[orig_idx])
            new_shape.append(self.shape[orig_idx])           # 只是swap各维度的dim, 维度相邻数据间内存间隔stride

        new_stride = tuple(new_stride)
        new_shape = tuple(new_shape)

        return self.make(new_shape, strides=new_stride, handle=self._handle, offset=self._offset,
                         device=self.device)  # 新shape的矩阵，底层用相同内存数据

        ### END YOUR SOLUTION

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """
        ### BEGIN YOUR SOLUTION
        old_shape=self.shape  # (2,3)  / (1,3)
        if len(new_shape)>len(old_shape):
            l=len(new_shape)>len(old_shape)
            old_shape=list(old_shape)
            for i in range(l):
                old_shape.insert(0,1)
            old_shape=tuple(old_shape)
            self=self.reshape(old_shape)
        # 不实际增加内存。不论增加几个维度，只是取数逻辑变了。 m,n维增加，对应strides都是0.只按底层取

        # offset不变
        # shape是new_shape

        # 如果是old_shape（3，）   brodcast到（16,3）

        # 检验维度。新维度new_shape (m,n,2,3) / (m,n,z,3)
        #         只能增加之前的维度。或者拓展已经是1的维度。其他维度都必须相等
        for i in range(len(new_shape) - 1, -1, -1):   # 每i个维度(倒序)
            if  i< len(old_shape) :  #  原始也有的维度. dim应相等
                original_dim=old_shape[i]
                if original_dim!=1:
                    assert new_shape[i]==old_shape[i]

        # 每个新增维度，不增加实际内存。stride是0，按原来维度和对应stride索引
        # 对由1拓展成的维，该维相邻2元素间的内存距离同一维时。不论该轴元素i实际是什么。实际上距离也是0
        #               (本质上，上下2个元素其实是同一个)  （1，6，4） -》 （3，6，4）  这3个元素间内存距离是0
        stride=[]
        for i in range(len(new_shape) - 1, -1, -1):   # 每i个维度(倒序)
            if  i< len(old_shape) :  #  原始也有的维度.strides不变
                original_dim=old_shape[i]
                if original_dim!=1:
                    stride.append(self._strides[i])  # 该维没有拓展。和原来保持相同
                else:
                    stride.append(0)  # 拓展了的维度。本质上该维2相邻元素间内存距离是0（是同一个元素）
            else:
                stride.append(0)

        new_shape = tuple(new_shape)
        new_stride = tuple(stride[::-1])  # 倒序遍历的各维度。正序回来
        return self.make(new_shape, strides=new_stride, handle=self._handle, offset=self._offset,
                         device=self.device)  # 新shape的矩阵，底层用相同内存数据

        ### END YOUR SOLUTION

    ### Get and set elements

    def process_slice(self, sl, dim):
        # 把传入的slice，None转成默认的。
        #              start负数，转成n
        #              stop负数，转成n+stop (位置小于start)
        #              不支持同时是负数
        #              （start正数，stop负数）。转成（start, n-|stop|）   9个数：-1：对应位置8，最后一个数
        #              （start负数，stop正数）。转成（n, stop）
        # 最终是正数：（start,stop,step）且step>0 ,stop>start
        """ Convert a slice to an explicit start/stop/step """
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            #start = self.shape[dim]  # bug?修复一下
            start = self.shape[dim] + start
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

     # 切片Slices
    # input：idxs:是一个tuple.共n维. 每一维对应该维的slice,含.start .stop .step。可能是none或者负数step
    # return: 返回对应的切片矩阵（ 底层内存不变，slice只是用new shape, stride, and offset来定义底层数据中的一部分）
    #         不reduce. 维度数目不变
    #         A[:4,2]：返回4，1大小的矩阵。
    #         A[2,2]   返回1，1大小的矩阵
    #         A[1:5,:-1:2,4,:]
    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)  # s: 第i维的slice.
                                              # 如果s是slice类型的变量， 最终变成正数：（start,stop,step）且step>0 ,stop>start
                                              # 如果s不是slice类型的变量，默认是一个int（也设成start,stop,1）。
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"         # 每一维一个slice

        ### BEGIN YOUR SOLUTION
        # 原始底层数据是一个向量 1，2，3，4，5，6
        # 每个维度初始值都变了。先确定新的初始坐标(i,j)： offset: i*s[i]
        # shape变了： 长为：stop-satrt+1   含len/step个元素
        # strides:   这一维。上下2行元素在内存中距离仍然比较远（没有compact）,不变。但如果step>1, 会变远。成为2
        new_shape = []
        offset = self._offset
        new_stride = []
        for i, s in enumerate(idxs):  # 每个维度i对应的slice
            offset += s.start * self._strides[i]  # 如果是一个int.光这个就够了。就是A[i,j]的位置
            new_shape.append(int(np.ceil((s.stop - s.start) / s.step)))  # (0,1,2)  a[::2] : (0,)
            new_stride.append(self.strides[i] * s.step)  # 步长变长了。每2行元素，在内存间距离变大了

        new_shape = tuple(new_shape)
        new_stride = tuple(new_stride)
        return self.make(new_shape, strides=new_stride, handle=self._handle, offset=offset,
                         device=self.device)  # 新shape的矩阵，底层用相同内存数据

        ### END YOUR SOLUTION

    # 直接改变底层数据. 在obj[key]=val时调用。   传入key和val
    # idxs是传入的slice,相当于key   a[key]=b
    # other是传入的value,相当于b    a[key]=一个slice
    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)    # 首先得到了切片
        # 这个view的handle,本身对应的size,是原始内存的size
        # 该原始内存只是一片连续内存。只有python端有strides,offst等信息
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,     # 先compact了，再设置到该切片上
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,                          # 把一个值，设置到该切片上
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        # 先new一个out. (NDarray) 调用底层.so后计算out对应的__handle__
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        # 如果.so文件实现了matmul_tiled方法， device对象的getsttibute,会根据so文件的attribute(a.f)是否存在，判断实现了
        # 设置的__tile_size__，可以被每个维度整除的情况下。使用 （m,n,p都切成一样大小的整块。vm,vn,vp相同）
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):
            # 根据原始矩阵a,以及切块大小，对同一片内存，设置新的shape和strides
            # shape变成 [ m/V, n/V,个，每个大小是[V,V] ]  。4维  a[V,V]一个小块，每行n/V个小块。共m/V行
            # strides。上下2块之间，隔着V行元素。 V*shape[1]
            #         左右2块之间，隔着V个元素：  V*strides[1]  (如果连续就是V)
            #         小块矩阵[V,V]，上下在内存中实际间隔一行。相邻2元素间隔1
            # 变成了新的逻辑矩阵,共4维。最后2维是小矩阵块前2位类似大网格，放置这些矩阵块。
            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__             # 矩阵被切块的大小 V
            a = tile(self.compact(), t).compact()     # 输入原始矩阵a,和切块大小。被切成V,V的小块矩阵
                                                      # 在用作计算时，已经重新分配了连续内存。使得每个VV内部都是连续的。不同VV内存行主序挨着
            b = tile(other.compact(), t).compact()    # b同。也是4维。

            # 输出也是4维。[m/V,p/V,V,V] ->  每个小矩阵对应C[V,V]
            # 输出也是4维。[m/V,p/V,V,V] ->  第i行的a[V:],和第i列的b[,V], 生成对应位置的小矩阵C[V,V]。 共[m/v,p/v]个
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)    # 调用c++中的矩阵计算，求每个小内存中C[V,V]的实际值
                                                                                    # 底层实际上还是连续的(m,p)个。只是逻辑上组织成了4维

            return (
                 out.permute((0, 2, 1, 3))                 # 交换1，2维，变成[m/V,V,p/V,V]
                .compact()                                # 这些小矩阵。重新分配到新内存上。行的沿着行拼接。列的沿列拼接
                .reshape((self.shape[0], other.shape[1])) # 最终reshape成(m.p)
            )

        else:
            # 普通乘法：直接调用cpu端的matmul函数。 传入2个矩阵a,b(首地址),和分配好的内存。在c++端算好新内存每个位置。(连续片段)
            out = NDArray.make((m, p), device=self.device)  # 分配一个全新的连续内存，放结果。设置好对应shape,strides
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis, keepdims=False):
        """ Return a view to the array set up for reduction functions and output array. """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))  # (1,1,1,...,size)  维度数目不变，大小变了。都到最后一维
            out = NDArray.make((1,) * (self.ndim if keepdims else 1), device=self.device)# (1,1,1,...,1)     最终空白矩阵

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(              # (0,1,2,3) 如果axis=1  (0,2,3,1) 交换要reduce的维度1到最后一维。
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)   #  其他维度不变。最后一维放要reduce的这维。最后一维和原来该维交换
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)]) # 其他维不变，该维reduce到1 原来(2,3,4,5),现在(2,1,4,5)
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),  # 也有可能不keepdims,直接少一维
                device=self.device,
            )
        return view, out  # 把要reduce的维度放最后一维的NDarray; 一个reduce后正确大小的空白矩阵NDaray；

    # A:把要reduce的维度放最后一维的NDarray;    2，3，4   + reduce(0/1)  ->   3,4,2 -> compact -> reduce -> 3,4,1(axis=0)/ 2 4 1(axis=1) 连续内存
    # B：一个reduce后正确大小的空白矩阵NDaray；其他维不变，该维reduce到1。  1,3,4(reduce0) 或者2 1 4(reduce1)  但底层内存连续。
                                                                                                     # 可以直接用reduce好的元素。用shape重新定义大小

    def sum(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])# 调用设备端reduce函数，把A compact,reduce后拷给B
        return out

    # A.max(axis=1)
    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims) # view:把要reduce的维度放最后一维的NDarray; 2，3，4   + reduce(0/1)  ->   3,4,2 连续内存
                                                        # 一个reduce后正确大小的空白矩阵NDaray,设置了正确的shape参数，且内存连续（放了reduce了的结果）

        #  底层内存n个一组弄好后，底层是连续的 2，3，4  + reduce(0/1) -> 3,4,1(axis=0)/ 2 4 1(axis=1)
        #  通过设置shape等方式（device的to_numpy函数），把底层连续内存解释成正确形状的np逻辑矩阵
        #  py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset); (会直接copy一份内存，而非view)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out


    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        对图像等进行翻转。 [1,2,3] -> [3,2,1]
                       [[1,2,3], -> [[3,2,1]
                       [4,5,6]]     [6,5,4]]    np.flip(axes=1)

                       [[1,2,3], -> [[6,5,4]
                       [4,5,6]]      [3,2,1]]    np.flip(全部)
        flip一个矩阵，一般需要修改offset,以及把底层内存设置上负的strides。
                    比如flip所有维，底层内存不变。
                    offset设成6
                    strides是负的，往之前的内存找。 原来stride是[3,1],现在是[-3,-1]   6，3 之间内存间隔是-3, 最后一维相邻元素65，内存间隔是-1.
                    A[1,2] (2)=  A[0,0] + i*strides[i] + j*strides[j]
        offset: flip维度i,j, offset是这些维度的最后一个元素的位置
        strides:flip维度i,j,其他维度对应的不变。只变该元素
        """
        ### BEGIN YOUR SOLUTION
        new_offset=0
        new_strides=list(self.strides)
        for i in axes:
            new_strides[i]=-self.strides[i]  # 要翻轉的維度，內存中距離不變，但是負的。前後反向
            new_offset+=self.strides[i]*(self.shape[i]-1)
        flip_array=self.make(shape=self.shape,strides=new_strides,offset=new_offset,handle=self._handle,device=self.device)  # 內存不變。.shape不變
        return flip_array.compact() # 新分配連續內存
        ### END YOUR SOLUTION


    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.

        类似np.pad, 输入每一维的left,right padding的数目。(不padding的维度，对应维是(0,0))
        为了防止cnn导致原始图像大变小 4*4 -> 2*2   (filter3)，先把图像padding成6*6 -> 4*4. 把边缘fliter进来
         （nlp中cnn类似，一句话长4 -> 2 (filter3)。可以先padding成6， 0 w1,w2,w3,w4,0   6->4
        """
        ### BEGIN YOUR SOLUTION
        # 底层数据不变，只修改shape, strides
        # 新建一个全0的数组
        # 总的维度不变
        # 要padding的维度,对应shape增加
        # strides连续。但是值用self中的handle填充. setitem, 每个维度slice是axes[left_pad:left_pad+shape]
        new_shape=[]
        slices=[]
        for i in range(len(self.shape)):
            new_shape.append(self.shape[i]+sum(axes[i]))
            start = axes[i][0]
            slices.append(slice(start,start+self.shape[i],1))
        pad_array=self.make(new_shape,device=self.device)  # 分配了新的连续内存
        pad_array.fill(0)
        pad_array[tuple(slices)]=self

        return pad_array
        ### END YOUR SOLUTION



def array(a, dtype="float32", device=None):
    """ Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()

def flip(a, axes):
    return a.flip(axes)


def summation(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)
