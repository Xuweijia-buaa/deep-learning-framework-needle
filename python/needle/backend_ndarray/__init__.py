# 作为一个包，其中的各类数据结构，提供给Tensor
# 含NDArray类
# 也导出cuda(),cpu()等函数。 该函数返回一个BackendDevice("cuda", ndarray_backend_cuda) 对象。封装对应.so
# 也导出array函数：实际上调用NDArray(a, device=device)，生成一个NDarray。大小同a
from .ndarray import *
