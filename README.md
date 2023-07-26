# needle：a torch-like deep-learning-framework
Public repository and stub/testing code for cmu 10-714.

* 类似pytorch,可自动求导。实现了step(),backward()等逻辑。bp时，按拓扑排序计算每个Tensor的grad.
* 前端以Tensor为核心，底层的数据结构如图:\
  其中涉及到内存操作的算子，最终dispatch到3类不同的implement上(numpy/cuda/cpp)
  ![image](https://github.com/Xuweijia-buaa/-Xuweijia-buaa-deep-learning-framework-needle/blob/main/python/needle/needle.svg)

