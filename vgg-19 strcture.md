# imagenet-vgg-verydeep-19 模型解析

[TOC]



## 简述

vgg-19模型是牛津大学工程科学系，视觉几何组(Visual Geometry Group)所提出的用于大规模图像识别的卷积神经网络，发布于２０１５年.该模型在小感知域，以及小步长的基础上强调了卷积模型的深度．通过增加更多的卷积层来增加网络的深度．

## 卷积网络架构

首先根据其命名，一共有19层网络含有可训练的参数，其中卷积层有１６层，还有三层全连接层(Fully Connected)，其中每个池化层以及卷积层都使用了ReLU激活函数．

从左到右神经网络的深度不断增加，每个网络相对于上一个增加的层用**加粗表示**,对于卷积层的表示用（感知域的大小 - 通道数）进行标志．其基本模型组成如下图Ｅ所示：

![ConNet Configuration](http://media.innohub.top/180609-con.png)

对于该网络，使用了小卷积核堆积，从而在获取相同感知域的前提下，减少参数，利用较小的`strides=1` 捕捉更多的局部特征．利用2个３＊３的卷积核可以得到5*5的感知域,三个这样的小卷积核堆积即可以得到一个7\*7的感知域

在这种情形下

+ 我们合并了单个非线性整流层，而不是简单的使用一个，这使得决策功能更具有区别性．

+ 减少了参数数量：

  假设每一个3x3的卷积filter都有C个channels,(卷积层的输出也为C)如果使用单一的7x7的卷积核，需要的参数数量为：

  7x7xCxC = 49C*\*2 如果使用３个3x3的卷积核，参数数量为：3(3x3xC)xC = 27C**2

**模型Ｃ中的感知域为1x1的卷积核**

在模型Ｃ中纳入1x1的卷积转化，是为了在不影响感知域的前提下，增加决策函数的非线性影响．其本质上是对其空间的线性投影，输入与输出的shape是相同的．

## 预训练模型数据详解

对于训练集`imagenet-vgg=-verydeep-19.mat` 是类似于单行表的的matlab数据，可以使用scipy.io.loadmat　方法进行加载，得到的是一个字典  字典的键值包括：

`dict_keys(['__header__', '__version__', '__globals__', 'layers', 'meta'])` 

`layers`对应的是一个二维array实际上是一个包含43个元素的ndarray, 分别对应vgg-19中４３个各层操作的结果

```python
>>> vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
>>> type(vgg)
<class 'dict'>

>>> layers = vgg['layers']
>>> layers.shape
(1, 43)
>>> type(layers)
<class 'numpy.ndarray'>
```

现在可以提取出第一层的ndarray进行分析：

```python
# 虽然是量维，但是第一维是虚的，所以可以直接忽略
layer = layers[0]
# 选取第一个卷积层
l0 = layer[0]		# 现在l0的shape为（１，１）所以还有两个虚维度，可以直接index进去
 dtype=[('name', 'O'), ('type', 'O'), ('weights', 'O'), ('size', 'O'), ('pad', 'O'), ('stride', 'O'), ('precious', 'O'), ('dilate', 'O'), ('opts', 'O')])
# 可以看到第一层的数据类型，一共有９个元素，分别对应者９个键值
l0 = l0[0][0]		# 现在的数据就是裸数据

l0[2]  #  该层训练所得到的参数　到达该层需要的index->vgg['layers'][lay_ids][0][0][0][2]
#  参数包含weights以及bias再看它的shape
l00 = l0[2]
l00.shape   # ---->(1,2)  再去掉一个虚维度,就只剩下两个元素即权值和误差
lo = l00[0]
# 再看lo[0].shape----->(3, 3, 3, 64)
#!!!!!!!!!!!!!!!!!!!!第一个卷积核的shape!!!!!!!!!!!!!!!!!!!!!!!分解完成
weights_of_layid = vgg['layers'][0][lay_ids][0][0][2][0][0]
bias_of_layid = vgg['layers'][0][lay_id][0][0][2][0][1]

name = vgg['layers'][0][lay_id][0][0][0][0]

name = vgg['layers'(字典key)][0(去掉"虚"的维,相当于np.squeeze)][layer(层索引)][0][0(连续两次去掉虚的维度)][0(name,type,weights..9类信息的索引,从0开始)][0(去掉"虚"的维,相当于np.squeeze)]

weight, bias = vgg['layers'(字典key)][0(去掉"虚"的维,相当于np.squeeze)][layer(层索引)][0][0(连续两次去掉虚的维度)][2(name,type,weights..9类信息的索引,从0开始)][0(去掉"虚"的维,相当于np.squeeze)][(0/1)]
```

## 每一层的具体对应

| 编号 | 卷积层                           |
| ---- | -------------------------------- |
| 0    | conv1_1    (3, 3, 64, 64)        |
| 1    | relu                             |
| 2    | conv1_2    (3,3,3,64)            |
| 3    | relu                             |
| 4    | maxpool                          |
| 5    | conv2_1     (3, 3, 64, 128)      |
| 6    | relu                             |
| 7    | conv2_2     (3, 3, 128, 128)     |
| 8    | relu                             |
| 9    | maxpool                          |
| 10   | conv3_1    (3, 3, 128, 256)      |
| 11   | relu                             |
| 12   | conv3_2    (3, 3, 256, 256)      |
| 13   | relu                             |
| 14   | conv3_3    (3, 3, 256, 256)      |
| 15   | relu                             |
| 16   | conv3_4    (3, 3, 256, 256)      |
| 17   | relu                             |
| 18   | maxpool                          |
| 19   | conv4_1    (3, 3, 356, 512)      |
| 20   | relu                             |
| 21   | conv4_2    (3, 3, 512, 512)      |
| 22   | relu                             |
| 23   | conv4_3    (3, 3, 512, 512)      |
| 24   | relu                             |
| 25   | conv4_4    (3, 3, 512, 512)      |
| 26   | relu                             |
| 27   | maxpool                          |
| 28   | conv5_1    (3, 3, 512, 512)      |
| 29   | relu                             |
| 30   | con5_2     (3, 3, 512, 512)      |
| 31   | relu                             |
| 32   | con5_3     (3, 3, 512, 512)      |
| 33   | relu                             |
| 34   | con5_4     (3, 3, 512, 512)      |
| 35   | relu                             |
| 36   | maxpool                          |
| 37   | FC           (7, 7, 512, 4096)   |
| 38   | relu                             |
| 39   | FC           (7, 7, 4096, 4096)  |
| 40   | relu                             |
| 41   | FC            (1, 1, 4096, 1000) |
| 42   | softmax                          |



