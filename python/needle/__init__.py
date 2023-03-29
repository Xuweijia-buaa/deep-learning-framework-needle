from . import ops
from .ops import *
from .autograd import Tensor
# 导出cuda(),cpu()等函数。 该函数返回一个BackendDevice("cuda", ndarray_backend_cuda) 对象。封装对应.so
from .backend_selection import *

from .init import ones, zeros, zeros_like, ones_like

from . import init
from . import data
from . import nn
from . import optim
