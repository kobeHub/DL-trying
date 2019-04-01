#1.tf.layers.max_pooling2d  二维最大池化层

```python
max_pooling2d(inputs, 
              pool_size, 
              strides, 
              padding='valid', 
              data_format='channels_last', 
              name=None)
    Max pooling layer for 2D inputs (e.g. images).

```

> inputs: 一定是秩为４的张量，
>
> strides：可以是一个整数或者二元序列
>
> pool_size: 可以是一个整数或者一个二元序列，分别指示[pool_height, pool_width]
>
> data_format: 一个字符串对象，用于表示输入维度的排序　　可以是 'channels_last'   (对应着NHWC)  				'channels_first':对应着NCHW 

# 2 . tf.layers.dense 建造全连接网络

```python
dense(inputs, 
      units, 
      activation=None, 
      use_bias=True, 
      kernel_initializer=None, 
      bias_initializer=<tensorflow.python.ops.init_ops.Zeros object at 0x7f0d6f682b38>, 
      kernel_regularizer=None, 
      bias_regularizer=None, 
      activity_regularizer=None, 
      kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=None)
    
    Functional interface for the densely-connected layer.
    
```

> 用于建造全连接网络层：
>
> **input:** 全连接蹭的输入
>
> **units:** 输出空间的维度
>
> **activation:** 调用的激活函数，默认为空时是线性函数
>
> **use_bias:** 为真时使用误差值
>
> **kernel_initializer:** weights矩阵的初始化函数，默认为get_variables()
>
> 完成的操作是`outputs = activation(input*kernal+bias)`
>
> 用于建造密集连接层
>
> ****

# 3.tf.layers.dropout  防止或减少过拟合

对于**全连接层**进行训练时，由于参数数量较多，通常输出时会按照一定的比例输出，防止过拟合．

Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。示意图如下：

![example](https://img-blog.csdn.net/20170623161515351?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHVhaHVhemh1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



在训练的每次更新过程中，将输入数据的保留下的数据按照　1/(1-rate)　的比率进行缩放，以便于在训练时间和推理时间内他们的和保持不变

```python
dropout(inputs, 
        rate=0.5, 
        noise_shape=None, 
        seed=None, 
        training=False, 
        name=None)
    Applies Dropout to the input.
    
# rate: 舍弃率
# noise_shape:  一个一维的张量，用于表示对于特定维度进行　随机的保留／舍弃的标志
#	比如：shape(inputs)=[k, l, m, n]
#   noise_shape = [k, 1, 1, n]
#	那么batch_size, channels维度的数据不会被进行舍弃，而 width, height维度的数据进行随机舍弃
#	training: 用于标志是否以训练模式（随机舍弃）返回
```

# 3.0 tf.nn.dropout

```python
dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
    Computes dropout.
    
# keep_prob: 每个元素被保留的概率，与上面函数的区别　　keep_prob  = 1- rate
```



# 4. 张量的基本数学运算

+ `tf.equal(x, y, name=None)`

​        返回一个（x == y）element-wise

+ `tf.argmax(input, axis=None, name=None, dimension=None, output_type = tf.int64)`

  在张量的某个维度返回最大值所在的索引

+ `tf.cast(x, dtype, name=None)`

  改变一个张量的数据类型

  ```python
  x = tf.constant([1, 5, 3], dtype = tf.int32)
  tf.cast(x, tf.float32)
  ```

  ​