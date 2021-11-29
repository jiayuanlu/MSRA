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

4.	基于Python API实现定制化张量运算Linear+Conv2d

    代码名称：`linear_multiconv2d_handin.py`

5.	基于C++ API实现定制化张量运算Linear+Conv2d

    代码名称：`linear_multiconv2d_handin_cpp.py`

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

### 实验结果1（Linear层为例）

|||||||
|---------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
|||| &nbsp; &nbsp; &nbsp; 性能评测 |
| 实现方式（Linear层为例）| &nbsp; epoch | train_time/epoch(s) | test_time/epoch(s) | test_loss/epoch | test_acc/epoch |
|<br/> <br/>PyTorch原有张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; &nbsp; 14 &nbsp; &nbsp; &nbsp; &nbsp;| 7.634793383734567 | 0.9857945612498692 |&nbsp; &nbsp; &nbsp; 0.0277 &nbsp;|&nbsp; 9917/10000<br>&nbsp; &nbsp; &nbsp; (99%) |
|<br/> <br/>基于Python API的定制化张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; &nbsp; 14 &nbsp;| 8.388102684702192 | 1.1764095340456282 |&nbsp; &nbsp; &nbsp; 0.0295 &nbsp;|&nbsp; 9895/10000<br>&nbsp; &nbsp; &nbsp; (99%) |
|<br/> <br/>基于C++的定制化张量运算<br/> <br/>&nbsp;|&nbsp; &nbsp; &nbsp; 14 &nbsp;| 7.718833548682077 | 1.0303474494389124 |&nbsp; &nbsp; &nbsp; 0.0272 &nbsp;|&nbsp; 9909/10000<br>&nbsp; &nbsp; &nbsp; (99%) |
||||||||

### 实验结果2（Linear+Conv2d层为例）

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
2.	基于Python和C++ API对Linear和Conv2d层实现定制化张量运算（见实验结果2）的结果显示，在训练次数均为14个epoch时：
    1.	对于训练平均时间和测试平均时间，PyTorch原有张量运算的时间开销最小，定制化张量运算比原有张量运算时间开销大，其中基于C++ API比基于Python API时间开销小；
    2.	对于测试数据上的平均损失率和准确率，PyTorch原有张量运算的准确率高达99%，但是基于两种API的定制化张量运算的测试损失率和准确率却不如前者，准确率仅有97%（我之前在PyTorch原有张量运算的基础上用全连接FNN网络，其中只有一层隐藏层（大小为256），准确率也有97%），可见当把PyTorch的二维卷积运算替换成自定义卷积核的卷积操作时，该网络的性能几乎失去了卷积层的优势，损失率和准确率与单隐藏层的全连接前馈网络基本无异，原因分析见`实验总结与思考`模块。

### 实验总结与思考

1.	在

## 具体步骤

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

## 参考资料

* EXTENDING PYTORCH: https://pytorch.org/docs/master/notes/extending.html
