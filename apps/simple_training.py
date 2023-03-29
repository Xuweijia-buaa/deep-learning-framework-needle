import sys



sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time
import needle as ndl
import needle.init as init

device = ndl.cpu()
def acc(logits,labels):
    return np.mean(np.argmax(logits,1) == labels)

def err_rate(logits,labels):
    return np.mean(np.argmax(logits,1) != labels)

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt==None:
        model.eval()
    else:
        model.train()   # 给定opt.是training mode
        opt.reset_grad()
    avg_loss=0.0
    avg_err_rate=0.0
    #avg_acc=0.0
    all_len=0.0
    for batch in dataloader:
        logits=model(batch[0])  # (N,logits)
        labels=batch[1]
        loss = loss_fn(logits, labels)
        b=len(labels.numpy())
        avg_loss += loss.numpy()*b                       # 累积loss
        avg_err_rate+=acc(logits.numpy(),labels.numpy()) *b   # 累积acc
        all_len+=b
        if opt!= None:
            opt.reset_grad() # 先清0，再算梯度
            loss.backward()  # 算各种梯度
            opt.step()       # 更新模型各个参数们。 根据Tensor当前值和对应梯度。cache_data
    print(avg_err_rate/all_len,avg_loss/all_len)
    return avg_err_rate/all_len,avg_loss/all_len
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss,device=ndl.cpu(), dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        model.train()
        train_acc, train_loss = epoch_general_cifar10(dataloader, model, loss_fn(device,dtype),opt)  # dataloader：随机初始化2
    return train_acc, train_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.eval()
    test_acc, test_loss= epoch_general_cifar10(dataloader, model, loss_fn,opt=None)
    return test_acc, test_loss
    ### END YOUR SOLUTION

def repackage_hidden(h):
    # 让h脱离原来的计算图。变成新的Tensor
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, ndl.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

### PTB training ###
def epoch_general_ptb(data, model:LanguageModel, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function  (T1,B) B个流
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt==None:
        model.eval()
    else:
        model.train()   # 给定opt.是training mode
        opt.reset_grad()
    avg_loss=0.0
    avg_err_rate=0.0
    all_len=0.0
    T1,B=data.shape         # T1长的流，B个
    if model.seq_model=='rnn':
        h0 = init.zeros(model.num_layers, B, model.hidden_size, device=device,dtype=dtype)
        h=h0
    elif model.seq_model=='lstm':
        h0 = init.zeros(model.num_layers, B, model.hidden_size, device=device,dtype=dtype)
        c0 = init.zeros(model.num_layers, B, model.hidden_size, device=device,dtype=dtype)
        h=(h0,c0)
    for i in range(0,T1-1,seq_len):    # 长为T1的原始序列，被截取成n段。每段是bptt长的序列(bptt,B)。
        # 每次取bptt长。把长为T的流，截成T/bptt段，每段长bptt. 对应数据(bptt,B)
        # 原始一个长为T的序列，切成了最长为bptt的几段。每段最后的输出h,作为下一段的输入h,逻辑上仍是一个长为T的序列的lstm
        # 但每段输入前一段的h时，只用值，不把计算图连在一起。输入的h只是const Tensor。每次计算图只到本次bptt的初始时刻
        X, y = ndl.data.get_batch(data, i, bptt=seq_len, device=device,dtype=dtype)  # X:(T,B)  y:是对应的next序列 （TB,）

        logits,h=model(X,h)     # logits:(TB,V) 相当于 (N,logits)
                                # h: 本次bptt输出的最后一个时刻的h,作为下一段bptt的初始h。逻辑上仍在训练一段长序列

        h=repackage_hidden(h)   # 让前一次的h detach()， 脱离原来的计算图。在本次计算图中，变成 const Tensor
                                # 多次bptt,都是针对同一个T长序列。拆成了多份。
                                # 因此每次输入的初始h，都是上份bptt输出的最后时刻对应的h。多个bptt训练完，逻辑上仍是该T长序列的lstm
                                # 但每次输入的h, 需要被当做是一个const Tensor，从之前的计算图中剥离出来。在本次计算中作为常数
                                # 否则本次计算bp时，bp到初始h后，还会bp到该h的input,把之前的计算图连起来。
                                #因此多次bptt间，每次输入的h需要detach一下，单纯作为一个const Tensor，不影响本次计算图大

        labels=y           # (TB,1) 相当于（N,）V分类的真实值。 预测序列中每个word的下一个word，用TB个word
        loss = loss_fn(logits, labels)
        b=len(labels.numpy())
        avg_loss += loss.numpy()*b                       # 累积loss
        avg_err_rate+=acc(logits.numpy(),labels.numpy()) *b   # 累积acc
        all_len+=b
        if opt!= None:
            opt.reset_grad() # 先清0，再算梯度
            loss.backward()  # 算各种梯度
            opt.step()       # 更新模型各个参数们。 根据Tensor当前值和对应梯度。cache_data
        print("acc:"+str(avg_err_rate/all_len),"loss:"+str(avg_loss/all_len))
    return avg_err_rate/all_len,avg_loss/all_len
    ### END YOUR SOLUTION


def train_ptb(model:LanguageModel, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function  (T1,B)  B个流
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        model.train()
        train_acc, train_loss = epoch_general_ptb(data, model, seq_len, loss_fn(device,dtype),opt,clip, device, dtype)
    return train_acc, train_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.eval()
    test_acc, test_loss= epoch_general_ptb(data, model, seq_len, loss_fn(device,dtype),opt=None,clip=None, device=device, dtype=dtype)
    return test_acc, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes

    # train cnn
    # device = ndl.cuda()
    device = ndl.cpu()
    # dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)  # 自己实现的getitem,__len__
    #
    # # 自己实现的迭代器。实现了__iter__，__next__方法，作用于dataset
    # dataloader = ndl.data.DataLoader( \
    #     dataset=dataset,
    #     batch_size=128,
    #     shuffle=True,
    #     # collate_fn=ndl.data.collate_ndarray,
    #     device=device,
    #     dtype="float32"
    # )
    # model = ResNet9(device=device,
    #                 dtype="float32")  # 基于自己的nn。参数都是Tensor。Tensor间交互，构建在自己的op和底层NDarray上。底层Array实现strides,shape等抽象，但封装了自己的cuda,c++算子
    # train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #               lr=0.001, weight_decay=0.001, device=device, dtype="float32")
    # evaluate_cifar10(model, dataloader,device=device, dtype="float32")


    np.random.seed(0)
    corpus = ndl.data.Corpus("data/ptb", max_lines=20)
    seq_len = 10
    num_examples = 100
    batch_size = 16
    seq_model = 'rnn'
    num_layers = 2
    hidden_size = 10
    n_epochs=2
    train_data = ndl.data.batchify(corpus.train, batch_size=batch_size, device=device, dtype="float32")  #(T1,B)
    model =  LanguageModel(30, len(corpus.dictionary), hidden_size=hidden_size, num_layers=num_layers, seq_model=seq_model, device=device)
    train_acc, train_loss = train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
    test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=seq_len, device=device)
    if str(device) == "cpu()":
        np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)  # 差不多，数据集有点不太一样 差0.03812313
        np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)
    elif str(device) == "cuda()":
        np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)