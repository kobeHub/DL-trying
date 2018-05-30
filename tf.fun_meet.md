# tensorflow 所用的函数总结

## １.`tf.nn.embedding_lookup():`用法

`embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)`

> 用于在params的列表中进行并行查找，是tf.gather的泛化，根据划分规则partition_strategy进行划分
>
> params可以被解释为大嵌入张量的分割，如果张量列表的长度大于一，则根据划分策略将id的每个标识符进行分割．　如果id空间不能均分分区数量，则每个第一个（max_id + 1）％len（params）分区将被分配一个id。
>
> ```
> 如果partition_strategy是“mod”，我们将每个id分配给分区p = id％len（params）。例如，13个ID分成5个分区：[[0,5,10]，[1,6,11]，[2,7,12]，[3,8]，[4,9]]
>
> 如果partition_strategy是“div”，我们会以连续的方式将id分配给分区。在这种情况下，13个ids被分成5个分区：[[0,1,2]，[3,4,5]，[6,7,8]，[9,10]，[11,12]]
>
> 查找结果连接成一个稠密张量。返回的张量具有形状（ids）+形状（params）[1：]。
> ```

```python
>>> a = tf.constant([[1,2,6,6], [6,69,996,65], [56,3,65,5], [56,95,6,6]])
>>> d = tf.nn.embedding_lookup(a, [1, 3])
>>> with tf.Session() as sess:
...     sess.run(a)
...     sess.run(d)
... 
array([[  1,   2,   6,   6],
       [  6,  69, 996,  65],
       [ 56,   3,  65,   5],
       [ 56,  95,   6,   6]], dtype=int32)
array([[  6,  69, 996,  65],
       [ 56,  95,   6,   6]], dtype=int32)##
```

## 2.随机数函数

+ `tf.set_random_seed(seed)`

  为了使得同一个图中的不同会话间的随机变量可以保持相同值，可以在变量声明前添加此条件，此时不同会话的随机变量值相同，但是在同一个session中调用两次同一个随机变量，值不一定相同，因为执行了两次

  ```python
  >>> tf.set_random_seed(1234)
  >>> a = tf.random_uniform([1])
  >>> b = tf.random_normal([1])
  >>> with tf.Session() as sess:
  ...     sess.run(a)
  ...     sess.run(a)
  ...     sess.run(b)
  ...     sess.run(b)
  ... 
  >>> with tf.Session() as sess1:
  ...     sess1.run(a)
  ...     sess1.run(a)
  ...     sess1.run(b)
  ...     sess1.run(b)

  #########################################################################
  array([0.06164038], dtype=float32)
  array([0.66036296], dtype=float32)
  array([1.2197604], dtype=float32)
  array([0.34607798], dtype=float32)

  array([0.06164038], dtype=float32)
  array([0.66036296], dtype=float32)
  array([1.2197604], dtype=float32)
  array([0.34607798], dtype=float32)
  ```

  也可以仅使得一个随机变量保持一致性，只需要在定义时添加特定的seed参数


+ `tf.random_normal()`:

  > `random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)`
  >
  > 从一个正态分布中获取随机数参数如下：
  >
  > shape: 输出的tensor的shape可以是数字张量或者是python一位数组
  >
  > mean: 正态分布的平均值
  >
  > stddev: 正态分布的标准差

+ `tf.random_uniform():`

  > ```
  > random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
  > ```
  >
  > 从一个均匀分布中返回随机数：
  >
  > shape: according upper
  >
  > minval: 均匀分布的下界默认为０
  >
  > maval:均匀分布的上界
  >
  > seed: 用于为分布创建一个随机的种子
  >
  > 对于浮点数而言，默认不指定范围时的均匀分布是[0 , 1)，但是对于整数而言必须指定范围

+ `tf.truncated_normal()`:

  > ```python
  > truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
  > ```
  >
  > 从一个截断的正态分布获取随机数
  >
  > shape: upper
  >
  > mean:平均值
  >
  > stdddev: 标准差
  >
  > 将数值大于平均值２个标准差的数值删去并且获取随机数

+ `tf.random_shuffle()`:

  > **`random_shuffle(value, seed=None, name=None)`**
  >
  > 在第一维度随机打乱张量的顺序，可用在随机梯度下降中
  >
  > **value:** 待打乱的tensor 
  > **seed:** A Python integer. 随机数种子 
  > **name:** 名称，可选

## 3. `tf.reduce_*()`

+ **`tf.reduce_mean()`:**

  > ```python
  > reduce_mean(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)
  > ```
  >
  > 计算一个张量在多个维度上的均值沿`axis`中给出的尺寸减少`input_tensor`。除非`keepdims`成立，否则`axis`中每个条目的张量等级减1。如果`keepdims`为true，则缩小的维度将保留长度为1。
  >
  > **注意：一些参数已经被弃用，在之后版本中会被替换　keep_dims ------>   keepdims**
  >
  > reduction_indices: The old (deprecated) name for axis.
  >       keep_dims: Deprecated alias for `keepdims`.
  >
  > axis: 给定所要减少的维度，即在此基础上进行求均值，若原来的shape=[3, 2]  
  >
  > ​	使用reduce(c, 0)  则减少了第一维，shape=[2,]  即对列求均值

  ```python
  >>> x = tf.constant([[1, 2], [3., 5.], [5, 3.]])
  >>> x.shape.as_list()
  [3, 2]
  >>> a = tf.reduce_mean(x, 0)
  >>> a.shape
  TensorShape([Dimension(2)])
  >>> x.shape.as_list()[0]
  3
  >>> a.shape.as_list()
  [2]
  >>> a
  <tf.Tensor 'Mean_1:0' shape=(2,) dtype=float32>
  >>> with tf.Session() as s:
  ...     s.run(a)
  ... 
  ###########array([3.       , 3.3333333], dtype=float32)

  >>> b=  tf.reduce_mean(x, 1)
  >>> b.shape
  TensorShape([Dimension(3)])
  >>> b.shape.as_list()
  [3]
  >>> with tf.Session() as s:
  ...     s.run(b)
  ... 
  ###############array([1.5, 4. , 4. ], dtype=float32)
  ###############
  #若对第一维求值则删去的是x.shape[0]得到的tensor的形状是[2, ]**
  ```

  **注意：在求均值是注意dtype默认为float64但是会根据输入判断**

  ```python
  x = tf.constant([1, 0, 1, 0])
  tf.reduce_mean(x)  # 0
  y = tf.constant([1., 0., 1., 0.])
  tf.reduce_mean(y)  # 0.5
  ```

+ **`tf.reduce_max()`:**

  ```python
  reduce_max(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)
      Computes the maximum of elements across dimensions of a tensor. (deprecated arguments)

  ```

## 4. tf.nn.nce_loss()

```python
nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, 
sampled_values=None, remove_accidental_hits=False, partition_strategy='mod', nam
e='nce_loss')
    Computes and returns the noise-contrastive estimation training loss.

```

  > 用于计算噪音对比估计的训练损失
  >
  >   **通常用于训练并且计算完整的sigmoid loss 并用于评估和计算，在这种情况下必须设置partition_atrategy= 'div'  以便于两个损失值一致**

  ```python
  if mode == "train":
    loss = tf.nn.nce_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        ...,
        partition_strategy="div")
  elif mode == "eval":
    logits = tf.matmul(inputs, tf.transpose(weights))
    logits = tf.nn.bias_add(logits, biases)
    labels_one_hot = tf.one_hot(labels, n_classes)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits)
    loss = tf.reduce_sum(loss, axis=1)

  ```
> Args:
>
> **weights:** 一个shape为[num_class, dim]的张量或者是一个张量列表shape[class_num ,dim]沿着０		　　　　　维进行连接，可能分区的嵌入
>
> **bias:** shape[num_claswqses]
>
> **labels:**目标类shape=[batch_size, num_true]  dtype=int64  
>
> **input:** 输入的张量　　shape=[batch_size, dim]
>
> **num_sampled:** 对于每一批数据进行随机取样的classes的数目
>
> **num_classes:** 所有可能的类的数目
>
> **num_true:** 每一个训练例子对应的目标的数目

## 5. tf.nn.conv2d 求单层２维卷积

```python
tf.nn.conv2d(input, 
             filter, 
             strides, 
             padding, 
             use_cudnn_on_gpu=True, 
             data_format='NHWC', 
             dilations=[1, 1, 1, 1], 
             name=None)
根据给定的4-D输入以及filter　计算二维卷积结果
给定的输入张量具有shape = [batch, in_height, in_width, in_channels]
卷积核具有shape = [filter_height, filter_width, in_channels, out_channels]
然后进行以下操作：
1. 将卷积核转化为一个二维矩阵 [filter_height*filter_width*in_channels, out_channels]
2. 从输入的卷中提取图像信息，组成一个虚拟张量具有
shape=[batch, out_height, out_width, filter_height*filter_width*in_channels]
3. 对于每个虚拟张量，右乘filter矩阵
４．ｐａｄｄｉｎｇ可取值'VALID' 'SAME'
	'VALID＇ 在步长不能够满足匹配时，会舍弃数据后部分的多余
    'SAME'	通过添加padding 在经过卷积运算后，输出的卷积shape与输入一致
```

+ input: A 4-D tensor ,shape需要遵循'data_format'

+ filter: 四维卷积核，具有和输入卷相同的shape

+ strides: 长度为４的张量，确定了在每个维度移动窗口的步长，维度的格式遵循　`data_format`

  注意需要满足 strides[0] = strides[3] = 1　通常在高和宽维度上使用相同的步长

+ padding: 可选　'SAME'(补全０界满足步长)  OR   'VALID（删去右侧数据以满足步长）'  指定了需要padding时的选择方式

+ data_format: 指定数据shape  N: batches,   H:height    W:width       C:  channels


## 6. tf.squeeze 去除shape中的为１的维度

```python
squeeze(input, axis=None, name=None, squeeze_dims=None)
    Removes dimensions of size 1 from the shape of a tensor.
```

对于给定的张量，该操作会删去所有为１的维度，并且返回一个相同类型的张量，如果不想去除所有维度１，可以根据给定的参数`axis=[1,3]` 删除指定维度

```python
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    tf.shape(tf.squeeze(t))  # [2, 3]
 # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
```

## 7.tf.reshape 改变张量形状

```python
reshape(tensor, shape, name=None)
    Reshapes a tensor.
    
    Given `tensor`, this operation returns a tensor that has the same values
    as `tensor` with shape `shape`.
    
    If one component of `shape` is the special value -1, the size of that dimension
    is computed so that the total size remains constant.  In particular, a `shape`
    of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
    
    If `shape` is 1-D or higher, then the operation returns a tensor with shape
    `shape` filled with the values of `tensor`. In this case, the number of elements
    implied by `shape` must be the same as the number of elements in `tensor`.
    
    For example:
    
```
    # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
    reshape(tensor, shape, name=None)
    
        Reshapes a tensor.
    
    Given `tensor`, this operation returns a tensor that has the same values
    as `tensor` with shape `shape`.
    
    If one component of `shape` is the special value -1, the size of that dimension
    is computed so that the total size remains constant.  In particular, a `shape`
    of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
    
    If `shape` is 1-D or higher, then the operation returns a tensor with shape
    `shape` filled with the values of `tensor`. In this case, the number of elements
    implied by `shape` must be the same as the number of elements in `tensor`.
    
    For example:
    
    ```
    # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
    reshape(tensor, shape, name=None)
    
        Reshapes a tensor.


​       

    Given `tensor`, this operation returns a tensor that has the same values
    as `tensor` with shape `shape`.
    
    If one component of `shape` is the special value -1, the size of that dimension
    is computed so that the total size remains constant.  In particular, a `shape`
    of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
    
    If `shape` is 1-D or higher, then the operation returns a tensor with shape
    `shape` filled with the values of `tensor`. In this case, the number of elements
    implied by `shape` must be the same as the number of elements in `tensor`.
    
    For example:
    
    ```
    # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # tensor 't' has shape [9]
    reshape(t, [3, 3]) ==> [[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]
    
    # tensor 't' is [[[1, 1], [2, 2]],
    #                [[3, 3], [4, 4]]]
    # tensor 't' has shape [2, 2, 2]
    reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                            [3, 3, 4, 4]]
    # 注意如果使用参数－１则会根据ｔｅｎｓｏｒ　shape自动对应相应的维度并且满足元素数量和ｓｈａｐｅ的关系
