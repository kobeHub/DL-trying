# CNN简介

## 1. 特征

**Background:**

深度学习的基本原理是基于人工神经网络，信号从一个神经元进入，经过**非线性**的*activation function*,

传到下一层神经元；再经过该层神经元的激活，继续向下传递，如此循环往复到输出层，正是由于这些非线性函数的反复叠加，才使得神经网络有了足够的capacity来抓取复杂的pattern

**所以使用activation function是必须的，如果不使用激活函数，每层的输出都是上一层的线性组合，与没有隐藏层效果相当，这种情形就是最原始的感知机(Preceptron)**



**Regular Neural Networks(RNN):**

一个神经网络的最简单的结构包括输入层，隐含层，输出层，每一层上有多个神经元，上一层的神经元通过激活函数映射到下一层的神经元，每个神经元之间有相应的权值，普通的神经网络架构都是全连接的，即每一层的每个神经元的输入包含上一层的全部输出



**Convolutional Neural Networks(CNNs/ConvNets):**

卷积神经网络与普通神经网络相似，由神经元构成，具有可学习的weights and biases，每个神经元接收一个输入进行点积运算后，并且可以选择非线性跟随．整个网络仍然表现出单一的可微分分数函数：

**从一端的原始图像像素(raw pxiel)到另一端的分数函数**

并且在最后一层（全连接层）仍然具有损失函数(SVM/Softmax) 	ConvNet体系结构明确地假设输入是图像，这允许我们将某些属性编码到体系结构中。这些使得forward 函数　更有效地实施并极大地减少了网络中的参数数量

![tri-nural](http://media.innohub.top/full.png)

## 2.与全连接神经网络的比较

对于一个32\*32\*3的图像而言，如果采用RNN(regular neural nets) 将其转化为一维矩阵，如果采用全连接架构，对于每一个像素都需要一个神经元　所以普通神经网络的一层将需要　32\*32\*3= 3027 个权重

不适合进行图像的处理，而且过多的参数极易导致过拟合

**使用神经网络:**

+ 特征提取的高效性
+ 数据格式简易性
+ 参数数目少量性

![compare](http://cs231n.github.io/assets/cnn/cnn.jpeg)

卷积神经网络每一层使得其神经元分布在三个维度上，每一层把一个3-D的volume输入转化为另一个3-D volume．图中的width, height代表了图像的维度大小，depth为3代表色素通道(R G B 3 channels)

> A ConvNet is made up of Layers. Every Layer has a simple API: It transforms an input 3D volume to an output 3D volume with some differentiable function that may or may not have parameters.



# 卷积层（Convolutional Layer）

## 1.Local Connectivity 局部连接

当处理高维的输入时，例如图片．使用神经元间的全连接显然可行性不大．可以将每一个神经元仅连接到一个局部区域的输出卷(volume) .这种连接的空间扩展是一个超参数，称为一个神经元的感知域(receptive field 等价于filter的大小)　**这种联系在空间上是局部的，在depth上是和输入卷相同的**

> 例如：对于32\*32*3的图像数据，感知域filter shape=[5,5] ，默认步数为１，　卷积层的每个神经元都有在一个［５，　５，３］的区域的输入卷，每个神经元都有  5\*5\*3=75  个权值weights(还有一个bias误差值)与之对应
>
> 用6个过滤器filter后得到的激活图形状是　28\*28\*6

> *Example 2*. Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of 3*3*20 = 180 connections to the input volume. Notice that, again, the connectivity is local in space (e.g. 3x3), but full along the input depth (20).

​	    ![l1](http://media.innohub.top/180502-l1.png)

## 2.local  region * input volume   求点积

对于每个过滤器其depth一定与 input volume　depth相同，才可进行点乘运算，下面以深度为３　的filter为例：(参数共享式)

+ 首先局部感受域的每一层，分别与filter 每一层的weights做点乘，得到３个暂时结果
+ 将三个暂时结果相加同时加上误差值 bias 得到卷积层的对应位置的值
+ 按一定的步长平移filter　重复以上步骤，每多一个filter卷积层的depth就＋１

![mul1](http://media.innohub.top/180502-mul1.png)

卷积层０的第一个元素计算：

s1 = 1\*1 + 1\*1+ 1\*1

s2 = 2\*1 + 1\*1 

s3 = 2\*1 + 1\*1 

output[0, 0, 0] = s1 + s2 +s3 +bias = 10

其他计算类似最终得到　［3, 3, 2］的卷积层输出

![mul2](http://media.innohub.top/180502-mul2.png)

![计算单层卷积](http://media.innohub.top/180502-l3.png)

## 3.空间排列 Spatial arrangement

有三个超参数控制着，卷积层的神经元数目，排列方式．

+ **depth:深度**　　由所使用的过滤器的数目确定　　
+ **stride:步长**　　过滤器的移动的步长，即每次移动的像素数，stride=1时每次移动一个像素位置　实际应用中很少用大于２的步长，步长越大得到的空间输出量越小     **S**
+ **zero-padding: 0 界**　有时将输入的卷边界用０填充会得到跟好的效果，０界的大小是一个超参数，其优点在于可以控制输出的空间的大小（可以使用这种策略使得输入和输出的卷具有相同的空间维度）    **P**

在假定输入卷的宽度为W, 卷积层神经元感知域的宽度为F, 步长为S，０界的宽度为P

则宽度上的神经元数目计算公式为：

`(W - F + 2P)/S +1`

高度上的神经元数目与之相同

深度即为filer的数量

**步长的限制：**

**在选用过滤器移动步长时应该慎重考虑，否则会造成输入卷无法被filer均分,即在公式`(W-F+2P)/S +1` 计算得到的不是整数，此时会报出错误或者添加一个0界使其满足要求**

![l2](http://media.innohub.top/180502-l2.png)



## 4 .参数共享　Parameter  Sharing

**1. Two Small insight:**

**(1) 功能是分层的**

使用低复杂度的功能可以组合形成高复杂度的功能，　这样比直接学习高复杂度的功能更为有效，例如检测圆圈的检测器可以用来检测；篮球或者脸部

**(2) 特征是平移不变的**

如果一个特征在（x, y）处计算有效，那么在(x', y') 处计算该特征也是有效的

**2. Real-world Example: **

[Krizhevsky et al](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) 的架构赢得 `ImageNet challenge` 使用的图片大小是[227, 227, 3 ] 在第一个卷积层所使用的神经元的参数为：receptive filed:  **F=11**		stride:**S=4**	no zero-padding:**P=0**

神经元数目为：（227-11）/4+1=  55,过滤器一共有96个所以得到的第一层卷积输出为　［55, 55 ,96］

在shape=[55,55,96]的卷中，每个神经元对应的输入的卷的感知域大小为［11, 11, 3］

```
神经元数目：
	55*55*96 = 290,400
对于每个神经元都需要 11*11*3 = 363个weights以及一个bias
故总参数数目为：
	364*290400 = 105,705,600 
```

根据以上分析可知，参数的数目很大．如果１中的两个假设成立，将会极大地减少参数数量．根据特征的平移不变性，对于单个二维深度切片（大小为［５５，５５，９６］的卷具有９６个深度切片每个大小为［５５，５５］）．限制每个深度切片中的神经元使用相同的权重和偏差

```
采用这种参数共享方案：
每个深度切片只需要原本一个神经元的参数：
参数数量为：
	96*(11*11*3 + 1) = 34,944
```

在反向传播实践中，volume中的每个神经元都会计算其权重的梯度，但是这些梯度会叠加在每个深度切片上，并且每个深度切片仅更新一组权重

**请注意，如果单个深度切片中的所有神经元都使用相同的权重向量，则可以在每个深度切片中将CONV层的正向传递计算为神经元权重与输入音量的卷积（因此名称：卷积层）。这就是为什么通常将权重集合称为过滤器（或内核），与输入进行卷积的原因。**

**3. 特例**

注意，在某些情形下参数共享假设可能失效，当ConvNet的输入图像具有特定的中心结构时，可能失效．例如：我们应该期望在图像的一侧学习完全不同的特征．一个实际的例子是当输入居中的脸的图像，希望在不同的空间位置都可以学习不同的眼镜以及头发的特征，在这种情形下，通常会放宽参数共享方案，将该层称为本地连接层**(Locally-Connected Layer)**

## 5.tensorflow 中建立卷积核的函数

```python
tf.layers.conv2d(
    inputs, 
    filters, 
    kernel_size, 
    strides=(1, 1), 
    padding='valid', 
    data_format='channels_last', 
    dilation_rate=(1, 1), 
    activation=None, 
    use_bias=True, 
    kernel_initializer=None, 
    bias_initializer=<tensorflow.python.ops.init_ops.Zeros object at 0x7ff604fc7128>, 
    kernel_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None, 
    kernel_constraint=None, 
    bias_constraint=None, 
    trainable=True, 
    name=None, 
    reuse=None)
```

This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

本层创建了一个卷积层，用于和输入层的卷作点积并且输出一个张量（实际上是交叉对应），如果`use_bias`参数为真则会产生一个误差值的可初始化对象，一个误差向量被创建并且加入到输出集中，如果激活函数参数非空，将或被使用在输出集中

+ `input`:输入的张量对象作为卷积核的数据

+ `filters`: 整数，指定卷积核的数量，对应输出的卷积层depth

+ `kernel_size`:包含两个整数的元组或者列表，确定二维卷积核的width 以及　height

+ `stride`:包含两个整数的元组或者列表，确定卷积核在width  height方向上每次移动的步长，可以是单个整数确定两个方向上的移动

+ `padding`: 可以取值 `'valid'`  or  `'same'` 

  使用`valid`选项会舍弃input volume中的多余的元素列或者行使之满足步长要求

  `same` 选项会填充两侧的0界使得满足步长要求，如果需要加入的0界的数量是奇数，则右侧默认多一行


## 6　扩张卷积

最新的发展是引入了一个新的超参数到卷积层的计算中　Dilated convolutions ([paper by Fisher Yu and Vladlen Koltn](https://arxiv.org/abs/1511.07122) ) 即在卷积核做平移求卷积时　可以通过给input volume 间添加特定的间隔，获得不同的卷积结果．

```
eg:
一个一维的卷积核size为３，当dilation_rate = 1时：
w[0]*x[0] + w[1]*x[1] + w[2]*x[2].
dilation_rate = 2时：
 w[0]*x[0] + w[1]*x[2] + w[2]*x[4]
```

