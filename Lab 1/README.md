# MSRA

project_handin_Lab1

## 实验报告（Lab 1 - 框架及工具入门示例）

### 实验目的

1. 了解深度学习框架及工作流程（Deep Learning Workload）
2. 了解在不同硬件和批大小（batch_size）条件下，张量运算产生的开销

### 实验原理

通过在深度学习框架上调试和运行样例程序，观察不同配置下的运行结果，了解深度学习系统的工作流程。

### 文件功能介绍

1.	MNIST样例程序：

    代码名称：`mnist_basic.py`

2.	可视化模型结构、正确率、损失值

    代码名称：`mnist_tensorboard.py`

3.	网络性能分析

    代码名称：`mnist_profiler.py`

4.	网络性能分析结果

    文件夹名称：`profiler`
    
    功能说明：该文件夹包含`CUDA_False`和`CUDA_True`两个文件夹，前者是在`profile`函数中`use_cuda=False`，在CPU和GPU上训练的性能结果仅包含CPU的性能分析，后者是在`profile`函数中`use_cuda=True`，在GPU上训练的性能结果包含CPU和GPU的性能分析。

### 实验环境

||||
|--------|--------------|--------------------------|
|硬件环境|CPU（vCPU数目）|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 16 &nbsp; &nbsp; |
||GPU(型号，数目)|&nbsp; &nbsp; NVIDIA Corporation Device 2520 (rev a1) &nbsp; &nbsp; |
|||&nbsp; &nbsp; Intel Corporation Device 9a60 (rev 01)&nbsp; &nbsp; |
|||&nbsp; &nbsp; GeForce RTX 3060显卡（laptop）&nbsp; &nbsp; |
|软件环境|OS版本|&nbsp; &nbsp; Linux操作系统ubuntu20.04版本|
||深度学习框架<br>python包名称及版本|&nbsp; &nbsp; pytorch:1.7.1+cu110|
||CUDA版本|&nbsp; &nbsp; CUDA Version 11.0.228|
||||

### 实验结果

1. 模型可视化结果截图
   
|||
|---------------|---------------------------|
|<br/>&nbsp;<br/>神经网络数据流图<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/143921374-01ba8929-80c8-4b2b-9447-3037b44bdb3d.png)|
|<br/>&nbsp;<br/>损失和正确率趋势图<br/>&nbsp;<br/>&nbsp;|batch_size=64:<br/>![图片](https://user-images.githubusercontent.com/90028974/143924011-f638429a-afa5-4831-824e-f7f6e4ee6dc6.png)<br/><br/>batch_size=16:<br/>![图片](https://user-images.githubusercontent.com/90028974/143922140-fea353e6-7d3b-4202-8523-8a401eacc07a.png)<br/><br/>batch_size=1:<br/>![图片](https://user-images.githubusercontent.com/90028974/143921144-c7bd2f18-e51d-43a0-8133-4daf146b1f01.png)|
|<br/>&nbsp;<br/>网络分析，使用率前十名的操作<br/>&nbsp;<br/>&nbsp;|GPU:<br/>![图片](https://user-images.githubusercontent.com/90028974/143981019-1aff070a-24bd-4d74-9fbb-eee762c50715.png)<br/><br/>CPU:<br/>![图片](https://user-images.githubusercontent.com/90028974/144005267-31fffeef-eb4e-4b8e-b743-d92c6f621b02.png)|
||||


2. 网络分析，不同批大小结果比较
    1.	GPU版：

|||
|------|--------------|
|批大小 &nbsp;| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 结果比较 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|<br/>&nbsp;<br/>1<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/144044033-7d3f9b63-7c8e-4c24-9773-0d95d0843191.png)|
|<br/>&nbsp;<br/>16<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/144044090-d0e0287a-982c-4180-be82-d1b125dd80fa.png)|
|<br/>&nbsp;<br/>64<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/144044143-1e763060-37df-49d1-b64b-eeab25300fb3.png)|
|||

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ii.CPU版：

|||
|------|--------------|
|批大小 &nbsp;| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 结果比较 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|<br/>&nbsp;<br/>1<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/144044197-3a918570-465c-4165-86d7-e4b3b79dbbed.png)|
|<br/>&nbsp;<br/>16<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/144044256-74af8fd2-9709-4ce8-b656-c775ab78cfac.png)|
|<br/>&nbsp;<br/>64<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/144044305-336245c0-b049-4072-9f9b-088743b9b549.png)|
|||

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; iii.GPU_CUDA_True版（仅统计CUDA使用率高于1%的操作）：  

||
|--------------|
![图片](https://user-images.githubusercontent.com/90028974/144047745-bef55403-b620-4a71-ab4b-96dcd79d7890.png)
||

### 实验结果分析

1.	由于MNIST的三个代码是基于同一套参数和网络来进行不同的性能分析方式，因此，三个代码得到的最终平均测试准确率为9914/10000 (99%)，平均测试损失率为0.0275；

2.	对于在GPU上的运算，由上面三个表格可以得出，在神经网络的计算过程中，卷积层的操作占用CPU（GPU）比例最大、占用时间最长（从profiler的所有性能指标从高到底排序看，均是卷积操作占据了网络的大部分使用率，dropout、池化和激活函数的计算占用CPU（GPU）很少（因为dropout就是按照一定概率让某个神经元的激活函数置0，max_pooling就是以一定步长按照给定的max_pooling窗口大小选出窗口内的最大值）；

3.	对于在CPU上的运算，与GPU相同的是卷积层操作的时间开销和CPU占用率都是最大的，且对CPU的总体占有率比在GPU上训练要更加平均一些，另外，在CPU上训练对于addmm运算和卷积层运算的时间开销和CPU占用率的差别比在GPU上训练小了近一半（即当用GPU训练时，卷积操作占有率大概是addmm的7～8倍；当用CPU训练时，卷积操作占有率仅是addmm的约3～4倍）；

4.	在GPU上训练测试的时间远小于在CPU上的训练和测试时间（约15～20倍），且在二者上的整体运算时间均随batch_size的增加而增大，但是在每一个batch_size中的计算时间随batch_size增加而减小。

### 实验总结与思考

1.	本次实验的性能评估主要用到了`profile`函数和`tensorboard`工具，可以较为清晰地看到训练准确率和损失率随着训练次数的增加而收敛，以及各种操作在CPU和GPU上的计算时间和占有率；

2.	对于参考代码画出的`tensorboard`图像，上面直接把每次训练的结果都画到了一张图上，导致所有训练曲线的颜色都相同，同时还存在上一次训练的收敛点与下一次训练的起始点的直线连接，即出现了准确率图像上的横跨右上和左下角的直线（或损失率图像上横跨左上和右下角的直线），图像比较不美观；我认为有一个简单的改进方法：把每一次的训练结果保存在一个文件中，那么`tensorboard`在画图时会调用不同的文件，就会把不同的训练曲线赋上不同的颜色，之前出现的两次训练曲线的连线也可以由此来消去（因为本次实验主要是理解深度学习框架，所以美观部分就留着以后写论文时用）；

3.	`profiler`参考代码在这次实验中仅各种操作在CPU上的性能，但是如果将代码中的`profile`函数中的`use_cuda=False`改成`use_cuda=True`，即可输出包含CPU和GPU的性能指标，具体结果见上表和`profile/CUDA_True`中的文件。

4.	addmm函数是在调用torch.nn.functional库进行相关运算（如log_softmax、ReLU）时会调用的一个linear函数（该函数位于functional.py这一脚本文件中），具体代码如下：<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![图片](https://user-images.githubusercontent.com/90028974/144052867-051a3135-cd59-4afb-8243-36e074bf1668.png)<br/>
    这个函数实现的功能官方文档给出的解释是一种矩阵相乘和相加运算：<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![图片](https://user-images.githubusercontent.com/90028974/144054150-7bfaeec3-e4c2-430e-805f-a6e09ea3be4c.png)<br/>
    
5.	torch.nn.Conv2d函数功能的官方解释为：<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![图片](https://user-images.githubusercontent.com/90028974/144056446-5742bbbd-d89e-428f-bc7a-a4c5a5f87e73.png)<br/>

6.	根据上述4和5的解释，对比Conv2d和addmm函数的实现，二者均为对矩阵的乘加运算，出现上面性能差异的原因（即总体来看Conv2d的用时和对CPU、GPU的占有率均高于addmm操作，但是当用GPU训练时，前者占有率大概是后者的7～8倍，而当用CPU训练时，前者占有率仅是后者的约3～4倍），我猜测是因为GPU的加入让Conv2d操作中矩阵运算的并行效率更高，更加充分地利用GPU和CPU资源，而addmm函数可能是本身无法利用GPU进行更高程度的并行计算，导致在GPU加入后该操作对GPU和CPU的占有率没有很大的变化。（仅是个人理解，若有错误，请老师指正）
    

### 具体步骤

1.	安装依赖包。PyTorch==1.5, TensorFlow>=1.15.0

2.	下载并运行PyTorch仓库中提供的MNIST样例程序。

3.	修改样例代码，保存网络信息，并使用TensorBoard画出神经网络数据流图。

4.	继续修改样例代码，记录并保存训练时正确率和损失值，使用TensorBoard画出损失和正确率趋势图。

5.	添加神经网络分析功能（profiler），并截取使用率前十名的操作。

6.	更改批次大小为1，16，64，再执行分析程序，并比较结果。

7.	【可选实验】改变硬件配置（e.g.: 使用/ 不使用GPU），重新执行分析程序，并比较结果。


### 参考资料

* 样例代码：[PyTorch-MNIST Code](https://github.com/pytorch/examples/blob/master/mnist/main.py)
* 模型可视化：
  * [PyTorch Tensorboard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) 
  * [PyTorch TensorBoard Doc](https://pytorch.org/docs/stable/tensorboard.html)
  * [pytorch-tensorboard-tutorial-for-a-beginner](https://medium.com/@rktkek456/pytorch-tensorboard-tutorial-for-a-beginner-b037ee66574a)
* Profiler：[how-to-profiling-layer-by-layer-in-pytroch](https://stackoverflow.com/questions/53736966/how-to-profiling-layer-by-layer-in-pytroch)


