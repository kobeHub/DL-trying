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

## tf中的赋值操作

### 1. tf.assign 更新值

```python
assign(ref, value, validate_shape=None, use_locking=None, name=None)
    Update 'ref' by assigning 'value' to it.
```

该操作在一个张量被赋值后，对该张亮的值进行更新，可以使得需要重新赋值的链式操作更为简单

+ ref: 需要是一个来自Variable的可变张量，可以没有被初始化
+ value: 新值
+ validate_shape: 若为真，会对value的shape进行预测，看其是否匹配原来的张量，如果为假，则直接进行赋值
+ use_locking: 是否用锁

```python
>>> a = tf.get_variable(name='weight', initializer = tf.constant(0.0))
>>> with tf.Session() as sess:
...     sess.run(a.initializer)
...     print(a.eval())
...     sess.run(a.assign(10))
...     print(a.eval())
... 
```



### 2. setattr()  &   getattr()

```python
setattr(obj, name, value, /)
	对于给定的对象设置相应的属性下的值
　　注意，属性name一定要是str,相当于　setattr(x, 'y', 0)   -->    x.y = 0
getattr(object, name[, default])  ->   value
```



