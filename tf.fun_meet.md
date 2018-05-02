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
> **bias:** shape[num_classes]
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



