# Lab 1 - 框架及工具入门示例

## 实验目的

1. 了解深度学习框架及工作流程（Deep Learning Workload）
2. 了解在不同硬件和批大小（batch_size）条件下，张量运算产生的开销


## 实验环境

* PyTorch==1.5.0

* TensorFlow>=1.15.0

* 【可选环境】 单机Nvidia GPU with CUDA 10.0


## 实验原理

通过在深度学习框架上调试和运行样例程序，观察不同配置下的运行结果，了解深度学习系统的工作流程。

## 实验内容

### 实验流程图

![](/imgs/Lab1-flow.png "Lab1 flow chat")

### 具体步骤

1.	安装依赖包。PyTorch==1.5, TensorFlow>=1.15.0

2.	下载并运行PyTorch仓库中提供的MNIST样例程序。

3.	修改样例代码，保存网络信息，并使用TensorBoard画出神经网络数据流图。

4.	继续修改样例代码，记录并保存训练时正确率和损失值，使用TensorBoard画出损失和正确率趋势图。

5.	添加神经网络分析功能（profiler），并截取使用率前十名的操作。

6.	更改批次大小为1，16，64，再执行分析程序，并比较结果。

7.	【可选实验】改变硬件配置（e.g.: 使用/ 不使用GPU），重新执行分析程序，并比较结果。


## 实验报告

### 实验环境

||||
|--------|--------------|--------------------------|
|硬件环境|CPU（vCPU数目）|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 16 &nbsp; &nbsp; |
||GPU(型号，数目)|&nbsp; &nbsp; NVIDIA Corporation Device 2520 (rev a1) &nbsp; &nbsp; |
|||&nbsp; &nbsp; Intel Corporation Device 9a60 (rev 01)&nbsp; &nbsp; |
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
|<br/>&nbsp;<br/>网络分析，使用率前十名的操作<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/143981019-1aff070a-24bd-4d74-9fbb-eee762c50715.png)|
||||


2. 网络分析，不同批大小结果比较

|||
|------|--------------|
|批大小 &nbsp;| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 结果比较 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|<br/>&nbsp;<br/>1<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/143981125-aa4d3fb1-29c1-4146-8696-2c6ce9daf5ae.png)|
|<br/>&nbsp;<br/>16<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/143981186-56b858c4-b4dc-4773-ab51-0ddb835a0f40.png)|
|<br/>&nbsp;<br/>64<br/>&nbsp;<br/>&nbsp;|![图片](https://user-images.githubusercontent.com/90028974/143981253-effb4157-7a02-4ac1-b9d0-8e9c02ae1d74.png)|
|||

## 参考代码

1.	MNIST样例程序：

    代码位置：Lab1/mnist_basic.py

    运行命令：`python mnist_basic.py`

2.	可视化模型结构、正确率、损失值

    代码位置：Lab1/mnist_tensorboard.py

    运行命令：`python mnist_tensorboard.py`

3.	网络性能分析

    代码位置：Lab1/mnist_profiler.py

## 参考资料

* 样例代码：[PyTorch-MNIST Code](https://github.com/pytorch/examples/blob/master/mnist/main.py)
* 模型可视化：
  * [PyTorch Tensorboard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) 
  * [PyTorch TensorBoard Doc](https://pytorch.org/docs/stable/tensorboard.html)
  * [pytorch-tensorboard-tutorial-for-a-beginner](https://medium.com/@rktkek456/pytorch-tensorboard-tutorial-for-a-beginner-b037ee66574a)
* Profiler：[how-to-profiling-layer-by-layer-in-pytroch](https://stackoverflow.com/questions/53736966/how-to-profiling-layer-by-layer-in-pytroch)


