"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # 𝑢𝑡+1=𝛽𝑢𝑡+(1−𝛽)∇𝜃𝑓(𝜃𝑡)
        # 𝜃𝑡+1=𝜃𝑡−𝛼𝑢𝑡+1
        for i,p in enumerate(self.params):
            if p.grad is None:
                continue
            # if p not in self.u:
            #     self.u[p] = np.zeros(p.shape)
            if p not in self.u:
                self.u[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype).cached_data # 参数，是Tensor下的NDarray
            grad = p.grad.cached_data + self.weight_decay * p.detach().cached_data # 增加l2的梯度。 考虑了L2 loss   l=1/2w2  dl/dw= w
            self.u[p]=  self.u[p]*self.momentum + grad*(1-self.momentum)         # detached: 生成一个constant Tensor。没有梯度没有op的纯常数Tensor.和计算图没什么联系，但仍是一个Tensor. 但底层数据和图中Tensor用的是同一个
            p.cached_data=  p.detach().cached_data  - self.lr*self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i,p in enumerate(self.params):
            if p.grad is None:
                continue
            if p not in self.m:
                self.m[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype).cached_data # 参数，是Tensor下的NDarray
            if p not in self.v:
                self.v[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype).cached_data
            p_data=p.detach().cached_data
            grad=p.grad.detach().cached_data+ self.weight_decay*p_data   # 考虑l2正则的新梯度  Ndarray
            self.m[p]=  self.m[p]*self.beta1 + grad*(1-self.beta1)         # detached: 生成一个constant Tensor
            self.v[p] = self.v[p] * self.beta2 + grad**2 * (1 - self.beta2)
            mt_hat= self.m[p] / (1-self.beta1**self.t)
            vt_hat= self.v[p] / (1-self.beta2**self.t)
            #p.cached_data =   p_data - self.lr *  (mt_hat/ (np.sqrt(vt_hat)+self.eps))
            p.cached_data = p_data - self.lr * (mt_hat / (vt_hat**0.5+ self.eps))
        ### END YOUR SOLUTION
