# MSRA

project_handin_Lab2

## 实验报告（Lab2 - 定制一个新的张量运算）

### 实验目的

1.	理解DNN框架中的张量算子的原理
2.	基于不同方法实现新的张量运算，并比较性能差异

### 实验原理

1. 深度神经网络中的张量运算原理
2. PyTorch中基于Function和Module构造张量的方法
3. 通过C++扩展编写Python函数模块
4. 实现卷积层（Convolutional）的自定义张量运算

### 文件功能介绍

1.	PyTorch原有张量运算Linear

    代码名称：`mnist_profiler.py`

2.	基于Python API实现定制化张量运算Linear

    代码名称：`linear_handin_py.py`

3.	基于C++ API实现定制化张量运算Linear

    代码名称：`linear_handin_cpp.py`

4.	基于Python API实现定制化张量运算Conv2d

    代码名称：`multiconv2d_handin.py`

5.	基于Python API实现定制化张量运算Linear+Conv2d

    代码名称：`linear_multiconv2d_handin.py`

6.	基于C++ API实现定制化张量运算Linear+Conv2d

    代码名称：`linear_multiconv2d_handin_cpp.py`

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



#### 注：以下时间花销的测量均在电脑未插电情况下完成，若电脑插电，则训练平均时长会减小约2.7s，测试平均时长减小约0.2s

### 实验结果1（Linear层为例）

|||||||
|---------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
|||| &nbsp; &nbsp; &nbsp; 性能评测 |
| 实现方式（Linear层为例）|  epoch | train_time/epoch(s) | test_time/epoch(s) | test_loss/epoch | test_acc/epoch |
|<br/> <br/>PyTorch原有张量运算<br/> <br/>&nbsp;|&nbsp;&nbsp;&nbsp; 14 &nbsp;| 7.634793383734567 | 0.9857945612498692 |&nbsp; &nbsp; &nbsp; 0.0277 &nbsp;|&nbsp; 9917/10000<br>&nbsp; &nbsp; &nbsp; (99%) |
|<br/> <br/>基于Python API的定制化张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; 14 &nbsp;| 8.388102684702192 | 1.1764095340456282 |&nbsp; &nbsp; &nbsp; 0.0295 &nbsp;|&nbsp; 9895/10000<br>&nbsp; &nbsp; &nbsp; (99%) |
|<br/> <br/>基于C++的定制化张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; 14 &nbsp;| 7.718833548682077 | 1.0303474494389124 |&nbsp; &nbsp; &nbsp; 0.0272 &nbsp;|&nbsp; 9909/10000<br>&nbsp; &nbsp; &nbsp; (99%) |
||||||||

### 实验结果2（Conv2d层为例）

|||||||
|---------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
|||| &nbsp; &nbsp; &nbsp; 性能评测 |
| 实现方式（Conv2d层为例）| epoch | train_time/epoch(s) | test_time/epoch(s) | test_loss/epoch | test_acc/epoch |
|<br/> <br/>PyTorch原有张量运算<br/> <br/>&nbsp;|&nbsp;&nbsp; 14 &nbsp;&nbsp;| 7.634793383734567 | 0.9857945612498692 |&nbsp; &nbsp; &nbsp; 0.0277 &nbsp;|&nbsp; 9917/10000<br>&nbsp; &nbsp; &nbsp; (99%) |
|<br/> <br/>基于Python API的定制化张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; 14 &nbsp;| 7.865629860333034 | 1.0374169519969396 |&nbsp; &nbsp; &nbsp; 0.1167 &nbsp;|&nbsp; 9641/10000<br>&nbsp; &nbsp; &nbsp; (96%) |
||||||||

### 实验结果3（Linear+Conv2d层为例）

|||||||
|---------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
|||| &nbsp; &nbsp; &nbsp; 性能评测 |
| &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp;  实现方式<br/> &nbsp;（Linear+Conv2d层为例）| &nbsp; epoch | train_time/epoch(s) | test_time/epoch(s) | test_loss/epoch | test_acc/epoch |
|<br/> <br/>PyTorch原有张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; 14 &nbsp; &nbsp; &nbsp; &nbsp;| 7.634793383734567 | 0.9857945612498692 |&nbsp; &nbsp; &nbsp; 0.0277 &nbsp;|&nbsp; 9917/10000<br>&nbsp; &nbsp; &nbsp; (99%) |
|<br/> <br/>基于Python API的定制化张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; 14 &nbsp;| 7.964399627276829 | 1.0546047346932548 |&nbsp; &nbsp; &nbsp; 0.1101 &nbsp;|&nbsp; 9656/10000<br>&nbsp; &nbsp; &nbsp; (97%) |
|<br/> <br/>基于C++的定制化张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; 14 &nbsp;| 7.806821141924177 | 1.004484840801784 |&nbsp; &nbsp; &nbsp; 0.1099 &nbsp;|&nbsp; 9655/10000<br>&nbsp; &nbsp; &nbsp; (97%) |
||||||||

### 实验结果分析

1.	基于Python和C++ API对Linear层实现定制化张量运算（见实验结果1）的结果显示，在训练次数均为14个epoch时：
    1.	对于训练平均时间和测试平均时间，PyTorch原有张量运算的时间开销最小，定制化张量运算比原有张量运算时间开销大，其中基于C++ API比基于Python API时间开销小；
    2.	对于测试数据上的平均损失率和准确率，PyTorch原有张量运算和基于两种API的定制化张量运算的测试损失率和准确率相差不大，即均有很高的准确率。

2.	基于Python和C++ API对Conv2d层实现定制化张量运算（见实验结果2）的结果显示，在训练次数均为14个epoch时：
    1.	对于训练平均时间和测试平均时间，PyTorch原有张量运算的时间开销最小，定制化张量运算时间开销稍大；
    2.	对于测试数据上的平均损失率和准确率，PyTorch原有张量运算和定制化张量运算的测试损失率和准确率相差较大，自定义二维卷积层的测试性能较差。

3.	基于Python和C++ API对Linear和Conv2d层实现定制化张量运算（见实验结果3）的结果显示，在训练次数均为14个epoch时：
    1.	对于训练平均时间和测试平均时间，PyTorch原有张量运算的时间开销最小，定制化张量运算比原有张量运算时间开销大，其中基于C++ API比基于Python API时间开销小；
    2.	对于测试数据上的平均损失率和准确率，PyTorch原有张量运算的准确率高达99%，但是基于两种API的定制化张量运算的测试损失率和准确率却不如前者，准确率仅有97%（我之前在PyTorch原有张量运算的基础上用全连接FNN网络，其中只有一层隐藏层（大小为256），准确率也有97%），可见当把PyTorch的二维卷积运算替换成自定义卷积核的卷积操作时，该网络的性能几乎失去了卷积层的优势，损失率和准确率与单隐藏层的全连接前馈网络基本无异，可以说自定义二维卷积层的性能在整个网络的性能评估上起主要作用（即整个网络的性能由最差的一个环节决定），原因分析见`实验总结与思考`模块。

### 实验总结与思考

1.	在本次实验中，我依次实现了基于Python和C++ API的自定义线性张量层、二维卷积层以及两者的混合，性能评估如上，具体性能指标见`profile`文件夹；

2.	对于自定义卷积层，我的实现方式是改变torch.nn.Conv2d函数中的卷积核，将自己定义的卷积核操作替换原有核，经过查询大量资料，我发现torch.nn库和torch.nn.functional库均可以实现二维卷积，但是前者的卷积函数没有给替换核的接口，较难更改卷积核，但是后者的卷积函数的第二个参数weight是一个切入点，因此可以把自定义的卷积核（类似二维高斯卷积核）作为weight传入torch.nn.functional.conv2d函数中，然后调用F.conv2d函数实现自定义的卷积层。另外，由于手写数字识别的数据集输入图片大小为28*28*1的灰度图，实现该卷积操作只需要把上面定义好的函数用到网络中，但若扩展想一下，若输入为RGB的色彩图，则需要在网络中实现三通道的二维卷积操作，简单实现即在现有基础上定义三通道的关系即可；

3.	在实现线性全连接层的参数初始化过程中，我没有用[-0.1,+0.1]上的均匀分布来初始化权重，而是用`bound = 1 / math.sqrt(self.weight.size(0))`作为均匀分布的上下界，由试验结果来看，前者的参数会使得训练核测试准确率更快达到较高值，这属于超参数的调整，由于神经网络的不可解释性，我目前也没有发现这两者出现性能区别的原因；

4.	对于这次实验以及神经网络的特性，我一直有一个疑惑，在我修改了本次代码中的一些参数因子后，训练和测试效果最大会提升1到2个点，但是这些参数的调整是不可解释的（至少目前我的认知是这样的），因此这种学习方法会被很多人诟病说大多数搞神经网络的人都沦为了调参侠，只要有资源，有显卡，有算力，就可以把问题丢给机器去学习，甚至目前有些问题就是用全连接网络实现了最好的性能效果。但是我认为深度学习应该是基于机器学习的，神经网络固然好，但是许多突破性的进展都是将NN与传统的机器学习方法结合实现的，比如随机森林、Adaboost等与神经网络的结合，性能提升可能达到5到10个点。因此，总结起来，问题归结为：
    
    1.	神经网络目前是否有比较有说服力的参数调整的理论解释？
    2.	目前深度学习的突破点有哪些？以及这些突破点的灵感是否有很多都继承于传统机器学习的方法？

### 具体步骤

1.	在MNIST的模型样例中，选择线性层（Linear）张量运算进行定制化实现

2.	理解PyTorch构造张量运算的基本单位：Function和Module

3.	基于Function和Module的Python API重新实现Linear张量运算

    1. 修改MNIST样例代码
    2. 基于PyTorch  Module编写自定义的Linear 类模块
    3. 基于PyTorch Function实现前向计算和反向传播函数
    4. 使用自定义Linear替换网络中nn.Linear() 类
    5. 运行程序，验证网络正确性
   
4.	理解PyTorch张量运算在后端执行原理

5.	实现C++版本的定制化张量运算

    1. 基于C++，实现自定义Linear层前向计算和反向传播函数，并绑定为Python模型
    2. 将代码生成python的C++扩展
    3. 使用基于C++的函数扩展，实现自定义Linear类模块的前向计算和反向传播函数
    4. 运行程序，验证网络正确性
   
6.	使用profiler比较网络性能：比较原有张量运算和两种自定义张量运算的性能

7.	【可选实验，加分】实现卷积层（Convolutional）的自定义张量运算

### 参考资料

* EXTENDING PYTORCH: https://pytorch.org/docs/master/notes/extending.html
