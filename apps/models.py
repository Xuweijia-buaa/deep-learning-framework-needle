import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

# input X: (2, 3, 32, 32)   NCHW  N=2  HW=32 C=3
#
cnn_params=[
    [3,16,7,4],  # in_channels, out_channels, kernel_size, stride  参数数目：in*out*kernel*2 + out  + (BN2 2out) out=16
    [16,32,3,2], #
    [32,32,3,1],
    [32,32,3,1],
    [32,64,3,2],
    [64,128,3,2],
    [128,128,3,1],
    [128,128,3,1]  # out_channel:128     HW *128
]
class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        # 8个带BN的CNN Block:
        self.cnn_blocks=[]
        for i in range(8):
            self.cnn_blocks.append(nn.ConvBN(*cnn_params[i],bias=True, device=device, dtype=dtype))

        #  其中2个res
        seq1=  nn.Sequential(self.cnn_blocks[2],self.cnn_blocks[3])
        seq2 = nn.Sequential(self.cnn_blocks[6], self.cnn_blocks[7])
        self.res=[nn.Residual(seq1),nn.Residual(seq2)]
        self.cnn_blocks[2]=None
        self.cnn_blocks[3] = None
        self.cnn_blocks[6] = None
        self.cnn_blocks[7] = None

        self.flat=nn.Flatten()
        self.linear1=nn.Linear(128,128,device=device,dtype=dtype)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(128,10,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # x:N CHW
        x=self.cnn_blocks[0](x)
        x=self.cnn_blocks[1](x)
        x=self.res[0](x)
        x=self.cnn_blocks[4](x)
        x=self.cnn_blocks[5](x)
        x = self.res[1](x)        # Cin=64 HW=2
        x=self.flat(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.output_size=output_size
        self.num_layers=num_layers
        self.embed=nn.Embedding(output_size,embedding_size,device,dtype)
        self.seq_model=seq_model
        if seq_model=='rnn':
            self.rnn=nn.RNN(embedding_size,hidden_size,num_layers,True,'tanh',device,dtype)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, True, device, dtype)
        self.linear=nn.Linear(hidden_size,output_size,True,device,dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)     T,B
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        T,B=x.shape
        x=self.embed(x)   # (T,B,d)
        x,h=self.rnn(x,h)   #  (T,B,d) ->  (T,B,h)   h:()
        x=self.linear(x.reshape((T*B,self.hidden_size)))  #  (T,B,h)  ->  (T,B,V) ->(TB,V)
        return x,h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    device = ndl.cpu()

    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")  # (T1,B)
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)