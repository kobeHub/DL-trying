# 图片的滤镜转换

将一个图片的stye应用到另外一张图片中去，需要同时考虑到所参照的style以及要改变的图片的损失函数 ．卷积神经网络具有很多层，每一层就像一个函数可以提取特定的特征，而较低层次更倾向于提取内容的特征，而较深的卷积层倾向于提取模式的特征

+ **Content loss:** 

  衡量原始图片与模式转换后的图片的内容上的差异损失

+ **Style loss:**

  衡量所参照的模式的图片与转换后图片的style的损失

![style transfer](/home/kobe/Python/learn/DL/TFinfo/Pic/style.png)

那么损失函数就可以重新审视：

+ **Content loss:**

  度量所产生的图片的内容层所映射的特征与原始图片的损失

+ **Style loss:**

  度量所产生的图片的模式层的映射特征与模式提供者的损失

## 使用预训练数据

```
VGG: Oxford (Visual Geometry Group)
AlexNet: Toronto
GoogleNet: Google
```

**Loss functions:**

![loss function](http://media.innohub.top/180530-loss.png)

