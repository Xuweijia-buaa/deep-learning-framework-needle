import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd
import needle as ndl

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:     # 水平翻转,每个通道都水平翻转。
            return np.flip(img,1)   # 只翻转水平方向
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        H,W,C=img.shape
        new_img=np.zeros(img.shape)
        # 原图：
        # 向上dx。下边的padding成0。原图的: [dx:H]    (x>0)  ,  移到新图[0:H-dx]
        # 向下dx。上边的padding成0：原图的： [0:H+x]    (x<0)  ，放在新图的[-x:H]
        # 左右类似
        orig_crop=img[max(shift_x,0):min(H+shift_x,H),  max(shift_y,0):min(W+shift_y,W),:]
        if orig_crop.size!=0:   # 有内容才能取到。否则是空白
            new_img[max(-shift_x,0):min(H-shift_x,H), max(-shift_y,0):min(W-shift_y,W),:]=\
                img[max(shift_x,0):min(H+shift_x,H),  max(shift_y,0):min(W+shift_y,W),:]   # 但是要被放在padding后的位置

        return new_img
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device: ndl.Device=ndl.cpu(),
        dtype : str= "float32"
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device=device
        self.dtype=dtype
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

        self.i = 0

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # 调用iter(obj)时，obj必须实现了__iter__方法。调用时本质上会调用__iter__方法
        # 一个迭代器必须实现__iter__方法，来返回迭代器对象自身。 因此一个迭代器对象obj满足：iter(obj)==obj
        if self.shuffle:# 乱序。在迭代器调用iter方法时，首先确定本次迭代的顺序
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        return self        # 返回该迭代器，也就是对象本身。调用next时，输出该ordering指定的每个batch
        ### END YOUR SOLUTION

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        # 迭代器对象被调用next(obj)时，会调用的方法。返回迭代器的下一项内容
        if self.i<len(self.ordering):
            indexs = self.ordering[self.i]  # 把该batch对应的index取出来:array([ 8, 23, 27,  5, 24, 15,  6, 12, 28, 33])
            batches= [Tensor(x,device=self.device,dytpe=self.dtype) for x in self.dataset[indexs]]  # 每个元素返回对应的B个。改成Tensor
            self.i+=1
            return batches    # 返回tuple
        else:
            raise StopIteration
        ### END YOUR SOLUTION

import gzip
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(label_filename, 'rb') as path:
            y = np.frombuffer(path.read(), np.uint8, offset=8)  # 8开始有整齐的数据.以字节为单位. 每个字节一个数据，代表一个uint  （m,）

        with gzip.open(image_filename, 'rb') as path:
            buffer = path.read()
            x = np.frombuffer(buffer, np.uint8, offset=16).reshape(len(y), 28 * 28)  # 16开始有整齐的数据.以字节为单位. 同样是一个字节一个像素
            x = x / x.max()
            x = x.astype(np.float32)     #(m,h,w)
        self.x=x
        self.y=y
        self.transforms=transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # 如果index是个array: dataset[indexs]. 也是可以的。是n张图片
        if isinstance(index,int):
            x=self.apply_transforms(self.x[index].reshape(28,28,1)).reshape(self.x[index].shape)  # .reshape(28,28,1)
        else:
            if isinstance(index,slice): # slice或者array   a[1:3]
                B=index.stop-index.start
            else:                       # 一样接收array     a[[3,1,2]]
                B=len(index)
            x=np.array([self.apply_transforms(x) for x in self.x[index]]).reshape(B,-1) # 单张图分别transform

        # array([ 8, 23, 27,  5, 24, 15,  6, 12, 28, 33])
        return tuple([x,self.y[index]])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION

# dataset 包含10个类的图片
# 被划分为five training batches 一个test batch, 每个batch 10000 images。共60000张彩色图片
# 每个图片32*32大小，3通道彩色
import pickle
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms=transforms
        self.p=p
        base_folder='/media/xuweijia/DATA/Courses/deep-Learning-System/hw4/data/cifar-10-batches-py'
        # dict[b'data']中，-- a 10000x3072 numpy array of uint8，对应10000个照片
        # 每行3072，对应每张图片32*32*3,是3个channel,每个channel32*32的像素。
        # 各channel顺序排列,The first 1024 entries contain the red channel
        # dict[b'labels']  # 10000个labels
        raw_data=[]
        raw_labels=[]
        if train:
            for i in range(5): # 5个训练file
                file = os.path.join(base_folder, 'data_batch_{}'.format(i+1))
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    raw_data.append(dict[b'data'])
                    raw_labels.append(dict[b'labels'])
        else:
            file=os.path.join(base_folder,'test_batch')
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                raw_data.append(dict[b'data'])
                raw_labels.append(dict[b'labels'])
        self.X=np.concatenate(raw_data,axis=0)/255
        self.y=np.concatenate(raw_labels)

        self.X=self.X.reshape(-1,3,32,32)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index,int):
            x=self.apply_transforms(self.X[index])
            y=self.y[index]
        else:# slice或者array   a[1:3]
            if isinstance(index,slice):
                B=index.stop-index.start
            else: # array     a[[3,1,2]]
                B=len(index)
            x=np.array([self.apply_transforms(x) for x in self.X[index]]) # 单张图分别transform
            y=self.y[index]
        return x,y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])

class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.word2idx)
        ### END YOUR SOLUTION

class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        base_dir='/media/xuweijia/DATA/Courses/deep-Learning-System/hw4/data/ptb'
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)
        #self.valid = self.tokenize(os.path.join(base_dir, 'valid.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids  看成一个大seq
        """
        ### BEGIN YOUR SOLUTION
        tokens=[]
        #self.dictionary.add_word('<eos>')
        with open(path,'r',encoding="utf8") as f:
            for i,line in enumerate(f):
                line=line.split() + ['<eos>']
                for w in line:
                    self.dictionary.add_word(w)
                tokens.extend([ self.dictionary.word2idx[w] for w in line])
                if max_lines!=None and i==max_lines-1:
                   break
        return tokens  # 看做一个大seq
        ### END YOUR SOLUTION



def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.  (T,B)   原来的l个句子，每个句子拆成
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.  abcd...f 和ghi...kl 当做2个独立的句子
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).

    data:大seq.按时间排列，长为l
    output: 拆成B列，对应B个独立的时间流。互相无影响。 如果B=4,每列包含T1个元素。 T1=l//4 剩下的元素抛弃
            之后每次取一个batch，是从这些时间流中，取出长为bptt的序列 （bptt,B）  对应B个独立的小seq,用来训练
            1      7       i=0  (取序列 (12,78)),作为一个batch，用来训练 （B=2,T=3）
            2      8       i=1  (取序列 (34,910)),作为下一个batch训练。初始h是 上个batch最后输出的h
            3      9
            ..     ..
    """
    ### BEGIN YOUR SOLUTION
    # 把整个数据拆成B个流。 (B,T1)  转置成（T1,B）且内存连续
    l=len(data)
    T=l//batch_size                            # 每个batch对应的流，包含的元素数目。之后按index,取每个流中对应的部分序列，作为B个序列样本
    batched_series=np.zeros((T,batch_size))
    for i in range(batch_size):                # 第b列：长为T的序列
        batched_series[:,i]=data[i*T:(i+1)*T]
    return batched_series                      # (大T,B)   总共有B个大T长的序列。B个序列独立。每个batch内，从中选取长为bptt的一部分
    ### END YOUR SOLUTION

# 得到第i个batch.
def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐  ┌ b h n t ┐
    └ b h n t ┘X └ c i o u ┘Y  (bptt,B)
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    (大T,B)   总共有B个大T长的序列。每个batch,从中选取长为bptt的一部分
    i - index
    bptt - Sequence length

    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray       从大T中选取i-i+bptt的序列。B个序列组成一个batch
    target - Tensor of shape (bptt*bs,) with cached data as NDArray     对应的next_word,即i+1对应的B个序列
    """
    ### BEGIN YOUR SOLUTION
    # 把长为T的流，截成T/bptt段.  每次取bptt长。对应数据(bptt,B)
    # 原始一个长为T的序列，切成了最长为bptt的几段。每段最后的输出h,作为下一段的输入h,逻辑上仍是一个长为T的序列的lstm
    # 但每段输入前一段的h时，只用值，不把计算图连在一起。输入的h只是const Tensor。每次计算图只到本次bptt的初始时刻。
    T,B=batches.shape

    seq_len=min(bptt,T - i -1)   #  到 i=T-1-bptt后，最后一个取不到bptt长了。 最后一个i是位于n*bptt
    X=batches[i:i+seq_len,:]      # 从流中，选取长为bptt的序列。一共B个序列。作为（bptt，B） 训练数据。
    Y=batches[i+1:i+1+seq_len,:]  # 和对应的label.这里是next-word,即流中下一个序列

    X=Tensor(ndl.NDArray(X,device=device),device=device,dtype=dtype)
    Y=Tensor(ndl.NDArray(Y,device=device).reshape((seq_len*B,)),device=device,dtype=dtype)
    return X,Y
    ### END YOUR SOLUTION