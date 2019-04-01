# 卷积网络架构

卷积神经网络一般由三种层构成：**卷积层，　池化层，　全连接层**，同时将用于使每个元素成非线性排列的RELU激活函数也称为一个层，下面是这些层一起构建的神经网络架构：



## １．Layer patterns 层模式

```
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
```

> 此模型中，*代表重复次数，？代表可有可没有．
>
> 一般而言，`N<=3` `M>=0` `0<= K <=3` 
>
> 堆叠一些CONV-RELU用于提取数据特征，然后可以选择进行池化操作，重复该操作Ｍ次，直到图片的空间维度到达了一个较小的量级
>
> 接下来的全连接层进行特征的汇总，对于学习到的特征进行不同权重的映射
>
> 最后一层ＦＣ进行输出

**一些具体实例：**

+ `INPUT -> FC`:  用于进行线性分类

- `INPUT -> CONV -> RELU -> FC`

- `INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC`. Here we see that there is a single CONV layer between every POOL layer.

- `INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC` Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation.

  ​

## :bulb:  卷积核的使用：以小博大

在选用卷积核时优先选用多级较小的卷积核而不是选用一个较大的卷积核．在一个较大的感受野上，可以采用多个小卷积核层叠的方式（自然每个卷积核之间是非线性的关系），达到更优的效果

采用小卷积核的优势在于：

+ 首先，当使用小卷积核时，多个卷积层叠加，之间采用了RELU等非线性激活函数，相较于单层的大卷积核，使得数据特征更容易表达
+ 使用小卷积核可以减少参数数量

```
假设将３个3x3的卷积核堆积在一起进行一个较大感知野的卷积操作：
第一层卷积核可以获取的input volume 的　ｖｉｅｗ　大小为 3x3
以第一层卷积结果，使用第二个卷积核进行操作，可以得到的 input volume view 为　5x5
以第二层卷积结果，使用第三个卷积核进行操作，可以得到的　input volume view 为 7x7
```

**可由以下逻辑证明：**

![stack](http://media.innohub.top/180504-stack.jpg)

**参数数量的减少*：**

假设所有的卷的channels = *C* 即所有卷的深度都是 *C* 

对于 一个7x7的卷积核而言，使用参数共享，每个fliter 需要的参数数量为：　7x7x*C* 

根据V2有*C*层，所需要Ｃ个fliter　参数数量：　49C**2

如果使用小卷积核：

每个filter: 3x3xC

每个卷积核：　ＣX(3x3xC)

一共三个卷积核：　27C**2

**不足之处在于，在进行反向传播时需要更大的空间去存储卷积层的中间结果**　

## 2. Layer Sizing Patterns 图层大小调整模式

再此架构中，每一部分的输入值有以下要求：

+ **Input Layer:** 输入图片的数据，size最好可以被２多次整除　例如：32 64 96 224 384 and 512
+ **conv layer:** 最好使用小的fliters (eg: 3x3 or at most 5x5) 步长为１，0界宽度可以使得卷积层输出的size与输入层一致（eg: F = 3, P=1  INPUT-3+2+1 输出不变）满足公式：　Ｐ＝（Ｆ－１）／２
+ **pool layer:** 最常用2x2的感知野（receptive fields） 步长为２，得到的卷可以减少75%的数据

这种方式减少了关于尺寸大小的问题，因为所有的卷积层都保留了输入层的空间维度，而POOL层负责进行空间下采样，即负责降维

或者不在卷积层中进行０填充，此时我们必须十分仔细的跟踪整个ＣＮＮ体系中的输入卷，确保fliter可以正常工作

## 3. residual network

一般而言，深度网络应该比浅层网络表现出更优性能，当没有过拟合的危险时．那么当我们建造了一个有Ｎ层的神经网络，并且达到了一定的精度，至少，一个具有相同结构的，只是最后一层进行了一次映射发的n+1层的网络应该具有相同的精确度，由此推之，n+2 n+3..层网络都应如此



但是在实际应用中，深度网络的表现都会有所下降

ResNet的作者考虑这些问题时，提出了一个假设：直接计算对应关系很难去学习，并且提出了一个假设，不再学习由x到Ｈ（X）的底层映射，而是学习两者之间的残差(residual) ,然后再去计算函数映射．

`F(X) = H(X) - X`

