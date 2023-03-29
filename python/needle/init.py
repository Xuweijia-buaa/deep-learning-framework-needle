import math
import needle as ndl


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = ndl.default_device() if device is None else device
    array = device.rand(*shape, dtype=dtype) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = ndl.default_device() if device is None else device
    array = device.randn(*shape, dtype=dtype) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    device = ndl.default_device() if device is None else device
    array = device.full(shape, c, dtype=dtype)
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

# 包中直接导出该函数。使用：ndl.ones
def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = ndl.default_device() if device is None else device
    array = device.rand(*shape) <= p
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = ndl.default_device() if device is None else device
    return ndl.Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )




def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )

import numpy as np
# U[-a,a]   a=gain×sqrt(6/fan_in+fan_out)
def xavier_uniform(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    abs=gain *  np.sqrt(6.0/(fan_in+fan_out))
    return rand(*shape,low=-abs,high=abs,**kwargs)  # 返回一个Tensor. shape=(fan_in, fan_out),as a tuple
    ### END YOUR SOLUTION

# N(0,std^2)
def xavier_normal(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std= gain * np.sqrt(2.0 / (fan_in + fan_out))
    return randn(*shape,mean=0,std=std,**kwargs)  # 剩下的参数，都通过dict传进去。函数会自己找对应键解析到参数上
    ### END YOUR SOLUTION

from functools import reduce


# 调用时，传来的fan_in带了kernel_size
def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if fan_in==1:      # 考虑一维向量的情况。维度用fan_out计算
        dim=fan_out
    else:
        dim=fan_in
    abs=np.sqrt(2) *  np.sqrt(3.0/dim)
    return rand(*shape,low=-abs,high=abs,**kwargs)  # 加大。 减小dim
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if fan_in==1:      # 考虑一维向量的情况。维度用fan_out计算
        dim=fan_out
    else:
        dim=fan_in
    std= np.sqrt(2.0) * np.sqrt(1.0/dim)
    return randn(*shape,mean=0,std=std,**kwargs)
    ### END YOUR SOLUTION

if __name__ == '__main__':
    device=ndl.cpu()
    T=5
    B=3
    V=10
    x = np.random.randint(0, V, (T, B)).astype(np.float32)  # T,B
    x=  ndl.Tensor(x, device=device)
    one_hot_vecs = one_hot(10, x.reshape((B * T,)), device=device)  # (TB,|V|)

   # .reshape((T * B, self.V))
