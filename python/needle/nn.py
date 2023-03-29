"""The module.
"""
from typing import List

from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import needle


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

# 递归得到模型的parameters,是模型属性中所有Parameter类型的属性
def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []

# 递归得到所有子模型（含自己）
def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True        # eval会修改这个，决定eval时行为

    def parameters(self) -> List[Tensor]:
        """递归，返回该module的所有参数, list"""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        '''返回该类的所有子模型(递归，含自己) 是属性中，所有Parameter类型的属性'''
        return _child_modules(self.__dict__)

    def eval(self):                   # 所有子模型training都是False.类似pytorch
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        '''前向计算。用对模型本身的函数调用如model(x,y)实现，传入输入。 args是不定个参数，kwargs是不定个有名字参数（或者一个dict）'''
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias=bias
        self.device=device

        ### BEGIN YOUR SOLUTION
        # 和torch不同。torch是XW.T  我们是直接乘。所以torch的fan_in与W是反过来的. 但fan_in本身是输入节点数目
        self.weight=Parameter(init.kaiming_uniform(in_features,out_features,shape=[in_features,out_features],device=device,dtype="float32", requires_grad=True)) # 后3个参数按名字给，会被**kargs解析
        if bias==True:
            self.bias=Parameter(init.kaiming_uniform(1,out_features,shape=[1,out_features],device=device,dtype="float32", requires_grad=True))
        self.have_bias=bias
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X: (B,in)
        # Y= XW+b  (B,out),也是一个Tensor.生成对应的ops
        Y=needle.matmul(X,self.weight)
        if self.have_bias==True:
            return Y+needle.broadcast_to(self.bias,(X.shape[0], self.out_features))   # (N,in) (in,out) -> (N,out) + (1,out)
        else:
            return Y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape=list(X.shape)
        multidim_size=1
        for i in shape[1:]:
            multidim_size*=i
        return needle.reshape(X,(X.shape[0],multidim_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)    # 底层用我们自己写的op
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return init.ones_like(x,device=x.device)/(ops.exp(-x)+1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):  # 多个model
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for layer in self.modules:   # 多层网络。可以直接用来被执行，输出结果
            x=layer(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def __init__(self,device=None, dtype="float32"):
        super().__init__()
        self.device=device
        self.dtype =dtype

    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        B=logits.shape[0]
        k=logits.shape[1]
        # 维度-1,改成len(x.shape)-1
        logsumexp=needle.logsumexp(logits,axes=(1,))  # 全部  (B,)
        one_hot_y=needle.init.one_hot(k,y,device=self.device, dtype=self.dtype)          # (B,k)    one-hot后的y. k类先one-hot(通过I),再按y的index选取
        zy=needle.summation(needle.multiply(logits,one_hot_y),axes=(1,))       # (B,k)  -> (B,1)   每个y位置的z
        loss= needle.summation(logsumexp-zy)/B                       #  mean log(sum(zi)) - zy   (B,)
        return loss
        ### END YOUR SOLUTION

# reshape (B,) to (B,N)
def mybraodcast(x,wanted_shape):
    B=wanted_shape[0]
    return needle.broadcast_to(needle.reshape(x,(B,1)),wanted_shape)

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # # 补充各特征均值，方差.每个特征一个。 （N,）
        self.weight=Parameter(needle.init.ones(self.dim,device=device,dtype=dtype,requires_grad=True)) # 补充各特征方差.每个特征一个
        self.bias=Parameter(needle.init.zeros(self.dim,device=device,dtype=dtype,requires_grad=True))   # 补充各特征均值.每个特征一个
        # Batch_nrom需要记录训练时，每个特征（在所有样本上的）的滑动均值/方差.在测试时使用  （N,）
        self.running_mean=Tensor(needle.init.zeros(self.dim,device=device,dtype=dtype,requires_grad=False)) # 训练均值的滑动平均 -used at evaluation time
        self.running_var=Tensor(needle.init.ones(self.dim,device=device,dtype=dtype,requires_grad=False)) # 训练方差的滑动平均 -used at evaluation time
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x:(B,n)  每个特征，对各样本求均值，方差。在样本维度上归一化
        B=x.shape[0]
        N = x.shape[1]
        if self.training==True:
            mean=needle.summation(x,axes=(0,))/B                       # 每个特征列的均值（在所有样本上的）  (N,)  （NHW,C）
            # 用来记录每个特征的滑动平均，在测试时使用 (用data，防止多建op)
            self.running_mean.data= self.running_mean.data*(1-self.momentum) +  Tensor(mean.data,dtype=self.running_mean.data.dtype)*self.momentum
            mean=needle.broadcast_to(needle.reshape(mean,(1,N)),x.shape)  # reshape (N)->  (1,N) -> (B,N)
            sub=x - mean
            var= needle.summation(sub**2,axes=(0,))/B   #  每个特征的方差:mean(xi-mean)^2   (N,)
            self.running_var.data = self.running_var.data * (1 - self.momentum)+ Tensor(var.data,dtype=self.running_var.data.dtype) * self.momentum
            var=needle.broadcast_to(needle.reshape(var,(1,N)),x.shape)       # reshape (1,N) -> (B,N)
        else:
            mean = needle.broadcast_to(needle.reshape(self.running_mean, (1, N)), x.shape) # 测试，用训练学到的各特征均值，方差
            var = needle.broadcast_to(needle.reshape(self.running_var, (1, N)), x.shape)
            sub = x - mean

        # w* (x-mean)/sqrt(var+e) +b   w,b不是标量了，每个特征一个 （而layerNorm里，针对单样本。所以共用一个）
        return needle.divide(sub,(var+self.eps)**0.5) * needle.broadcast_to(self.weight,(B,N))+ needle.broadcast_to(self.bias,(B,N))

        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        # 变成[nhw,c]  多维BN，变成2维。每个图片拉成(hw,c)   按channel做BN
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        # 做完一维BN，reshape回去。对维度C，做归一化
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    '''输入是（B,n）  对每个样本的所有特征归一化。   每个样本的不同特征间的差异保留了'''
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # 是用来适当恢复原来方差，均值的可学习参数
        # self.weight=Parameter([1],device=device,dtype=dtype,requires_grad=True)  # 补充原方差  (指定名字才能被*之后的接收，否则会被*接收)
        #                                                                          # grad算出来有问题。维度不是(1,)了. 变成（32，32）了。 验下weight的backward。
        # self.bias=Parameter([0],device=device,dtype=dtype,requires_grad=True)    # 补充原均值
        # 本来是每个样本各自的，但也变成dim维
        self.weight=Parameter(needle.init.ones(self.dim,device=device,dtype=dtype,requires_grad=True)) # 补充各特征方差.每个特征一个
        self.bias=Parameter(needle.init.zeros(self.dim,device=device,dtype=dtype,requires_grad=True))   # 补充各特征均值.每个特征一个
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: (B,n). 对每个样本所有特征归一化
        B=x.shape[0]
        N = x.shape[1]
        assert N==self.dim
        mean= needle.summation(x,axes=(-1,))/N   # 每个样本独自计算：所有特征的均值 mean  (B,)
        mean=mybraodcast(mean,x.shape)
        sub=x - mean
        var= needle.summation(needle.multiply(sub,sub),axes=(-1,))/N   # mean(xi-mean)^2   (B,)
        var= mybraodcast(var, x.shape)

        return needle.divide(sub,needle.power_scalar(var+self.eps,0.5)) * needle.broadcast_to(self.weight,(B,N))+ needle.broadcast_to(self.bias,(B,N))  # w* (x-mean)/sqrt(var+e) +b   w,b都是标量。但不能当scalar,主动broadcast

        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training==True:
            #if np.random.rand() <  self.p:         # 以概率p被drop成0
            #within_p = needle.init.randb(*x.shape,device=x.device,dtype=x.dtype,requires_grad=False)  # 每个xi是否被dropout 1/0。 True的被drop
            within_keep = needle.init.randb(*x.shape, p=1-self.p,device=x.device, dtype=x.dtype, requires_grad=False)  # 留下的节点概率是1-p. mask是1
            return x*within_keep/(1-self.p)                # 以概率p dropout一部分节点,取值变成0。剩余节点适当增大1/(1-p)，弥补dropout的损失

        else: # eval:不dropout
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same (padding后，经过cnn的img,大小同原图(不考虑strides)。padding_size=k-1.   H2=H1-k+1)
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding=kernel_size//2
        #self.padding = kernel_size-1
        # filter:(k,k,in,out)
        shape=(kernel_size,kernel_size,in_channels,out_channels)
        k2=kernel_size*kernel_size
        self.weight=Parameter(init.kaiming_uniform(k2*in_channels,k2*out_channels,shape=shape,device=device,dtype=dtype, requires_grad=True)) # 后3个参数按名字给，会被**kargs解析
        # bias:每个大filter对应一个bias  （H2W2,c_out）,对应cout个bias.
        if bias:
            abs=1.0/(in_channels * kernel_size ** 2) ** 0.5
            self.bias=Parameter(init.rand(out_channels, low=-abs, high=abs, device=device, dtype=dtype, requires_grad=True))
        self.have_bias=bias
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        'x:NCHW'
        ### BEGIN YOUR SOLUTION
        x=x.transpose(axes=(1,2)).transpose(axes=(2,3))             # 转成NHWC  NCHW -> NHCW -> NHWC
        conv=ops.conv(x,self.weight,self.stride,self.padding)       # (H2,W2,cout)
        N,H2,W2,Cout=conv.shape
        if self.have_bias:
            conv=conv+needle.broadcast_to(self.bias.reshape((1,1,1,Cout)),(N,H2,W2,Cout))
        conv=conv.transpose(axes=(2,3)).transpose(axes=(1,2))      # NHWC -> NCHW
        conv.cached_data=conv.cached_data
        return conv
        ### END YOUR SOLUTION

class ConvBN(Module):
    'conv -> batch -> relu'
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.conv=Conv(in_channels, out_channels, kernel_size, stride, bias, device, dtype)
        #self.batch2d=BatchNorm2d(in_channels,device=device,dtype=dtype)
        self.batch2d = BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu=ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # conv -> BN2d -> relu
        'x:NCHW'
        x=self.conv(x)   # NCHW
        x=self.batch2d(x)# NCHW
        x=self.relu(x)
        return x


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size

        ht= xt(N,d) * Wih(d,h) + ht-1(N,h) * Whh(h,h)  +b(h,)
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size=input_size   # d
        self.hidden_size=hidden_size # h
        self.bias=bias
        self.nonlinearity=nonlinearity
        k=1.0/hidden_size
        # 2个大参数 （d,4h）  (h,h)
        self.W_ih=Parameter(init.rand(input_size,hidden_size,low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype))
        if bias==True:
            self.bias_ih=Parameter(init.rand(hidden_size,low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype))
            self.bias_hh=Parameter(init.rand(hidden_size,low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype))
        self.nonlinear=Tanh()
        if nonlinearity=='relu':
            self.nonlinear = ReLU()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.

        ht= xt(N,d) * Wih(d,h) + ht-1(N,h) * Whh(h,h)  +b(h,)
        """
        ### BEGIN YOUR SOLUTION
        B,d=X.shape
        ht = X @ self.W_ih
        if h==None:
            h=init.zeros(B,self.hidden_size,device=X.device,dtype=X.dtype)
        ht += h @ self.W_hh        # ht= xt(N,d) * Wih(d,h) + ht-1(N,h) * Whh(h,h)  ->  (N,h)
        if self.bias==True:
            broadcast_shape=ht.shape
            ht=ht+self.bias_ih.broadcast_to(broadcast_shape) +self.bias_hh.broadcast_to(broadcast_shape)
        return self.nonlinear(ht)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size=input_size   # d
        self.hidden_size=hidden_size # h
        self.bias=bias
        self.nonlinearity=nonlinearity
        self.num_layers=num_layers
        self.device=device
        self.dtype=dtype
        self.rnn_cells=[]
        for i in range(num_layers):
            if i==0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        T,B=X.shape[0], X.shape[1]
        #H=init.zeros(T,B,self.hidden_size,device=self.device,dtype=self.dtype) # T，B, h
        hs=h0.split(0) if h0!=None else None
        last_h=[]
        for l in range(self.num_layers):  # 按每层算
            h=hs[l] if hs!=None else None # 每层的初始时刻输入状态
            xt_s=X.split(0)               # 该层的输入序列本身，按时间切分
            ht_s=[]
            cell=self.rnn_cells[l]
            for t in range(T):   # 当前层的输出H(T,B,d),按时间依次算ht. 该层用同一个cell
                ht=cell(xt_s[t],h)        # xt,ht-1 ->ht. （B,h）
                h=ht                      # 该ht作为下一时刻的输入状态
                ht_s.append(ht)
            H=ops.stack(ht_s,axis=0)
            X=H                           # 该层输出，作为下一层的输入
            last_h.append(ht)             # 每层最后一个输出 B,h
        last_h=ops.stack(last_h,axis=0)   # (n_layer个)
        return H,last_h
        # 没实现get/setitem op。 只能用Stack/Split op实现index，如X[t]
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size=input_size   # d
        self.hidden_size=hidden_size # h
        self.bias=bias
        k=1.0/hidden_size
        # 2个大参数 （d,4h）  (h,h)
        self.W_ih=Parameter(init.rand(input_size,4*hidden_size,low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype))
        if bias==True:
            self.bias_ih=Parameter(init.rand(hidden_size,low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype))
            self.bias_hh=Parameter(init.rand(hidden_size,low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype))
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # ht= xt(N,d) * Wih(d,4h) + ht-1(N,h) * Whh(h,4h)  ->  (N,4h)
        B,d=X.shape
        ht = X @ self.W_ih
        if h==None:
            h0 = init.zeros(B, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(B, self.hidden_size, device=X.device, dtype=X.dtype)
            h = (h0, c0)
        h0=h[0]
        c0=h[1]
        ht += h0 @ self.W_hh
        if self.bias==True:
            broadcast_shape=ht.shape
            ht=ht+self.bias_ih.broadcast_to(broadcast_shape) +self.bias_hh.broadcast_to(broadcast_shape)
        # 分给4个激活函数  （N,4h） -> 4个(N,h)
        N,_=ht.shape
        matrix_results=ht.reshape((N,4,self.hidden_size)).split(1)  # 4个(N,h)
        i= Sigmoid()(matrix_results[0])
        f= Sigmoid()(matrix_results[1])
        g= Tanh()(matrix_results[2])
        o= Sigmoid()(matrix_results[3])
        ct=f*c0 + i*g if h!=None else i*g
        ht= o*Tanh()(ct)
        return ht,ct
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size=input_size   # d
        self.hidden_size=hidden_size # h
        self.bias=bias
        self.num_layers=num_layers
        self.device=device
        self.dtype=dtype
        self.lstm_cells=[]
        for i in range(num_layers):
            if i==0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        T,B=X.shape[0], X.shape[1]
        hs= h[0].split(0) if h!=None else None
        cs =h[1].split(0) if h!= None else None
        last_hs=[]
        last_cs = []
        for l in range(self.num_layers):  # 按每层算
            hc=(hs[l],cs[l]) if hs!=None else None # 每层的初始时刻输入状态
            xt_s=X.split(0)               # 该层的输入序列本身，按时间切分
            ht_s=[]
            cell=self.lstm_cells[l]
            for t in range(T):   # 当前层的输出H(T,B,d),按时间依次算ht. 该层用同一个cell
                ht,ct=cell(xt_s[t],hc)        # xt,ht-1 ->ht. （B,h）
                hc=(ht,ct)                    # 该ht作为下一时刻的输入状态
                ht_s.append(ht)               # 该层整体输出
            H=ops.stack(ht_s,axis=0)
            X=H                           # 该层输出，作为下一层的输入
            last_hs.append(ht)            # 每层最后一个输出 B,h
            last_cs.append(ct)
        last_h=ops.stack(last_hs, axis=0)   # (n_layer个)  n_layer,B,h
        last_c=ops.stack(last_cs, axis=0)
        return H,(last_h,last_c)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.randn(num_embeddings,embedding_dim,device=device,dtype=dtype))
        self.V=num_embeddings
        self.d=embedding_dim
        self.device=device
        self.dtype=dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)  T，B

        Output:
        output of shape (seq_len, bs, embedding_dim)  T,B,d
        """
        ### BEGIN YOUR SOLUTION
        T,B=x.shape
        one_hot_vecs = init.one_hot(self.V, x.reshape((B*T,)), device=self.device, dtype=self.dtype) # (TB,|V|)
        embed=one_hot_vecs@self.weight   # (TB,|V|) * (|V|,d)  ->  (TB,d)
        return embed.reshape((T,B,self.d))  # (T,B,d)
        ### END YOUR SOLUTION
