# what is tensorflow?

## Tensor

tensor 即张量，可以进行简单的数据映射：

+ ０－d tensor ---------　标量
+ 1-d tensor    --------- 向量
+ ２－d tensor   --------　矩阵

tensorflow 中每个操作符作为一个节点，节点之间由边连接，边中流动的数据以张量形式表示，有一类边中没有数据流动，称为是依赖控制，作用是起始节点执行完后再执行目标节点，进行灵活的条件控制．

> Nodes: 操作符，变量，常量
>
> Edges: 张量



## Session

tensorflow 的每一个操作都需要事先定义好，通过一个会话（session）来执行该操作例如：

```python
import tensorflow as tf
a = tf.add(3, 5)
with tf.Session() as sess:		# 用以替代　se = tf.Session()  se.close()
	print(sess.run(a))

```

session会为当前的值和变量分配空间，并且进行运算

`tf.Session.run(fetchs, feed_dict=None, options=None, run_metadata=None)`

> 该方法运行tensorflow计算的一个步骤，通过运行必要的图片段以执行每个动作
>
> 替换　feed_dict 的值为对应的输入值

> fetchs 是运算所需的值的list





## Distributed Computation

可以把一个图划分为几个块在多GPU下并行运算，进行分布式计算，需要指定某个GPU运行某个会话计算某个子图

```python
# Creates a graph.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
  c = tf.multiply(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(c))

```

## 多图

:bug:**BUG Alert**

+ 多图运算需要多个会话，而每个会话默认都会最大限度的利用现有计算资源

+ 如果不通过pynum/python，无法在图之间传递数据，意味着不可以分布式计算

+ 最好是通过多个不连接的子图来实现需求

  ​

**坚持使用？？？**

```python
# 处理默认图
g = tf.get_default_graph()

# 用户创建的图，设置为默认图
g = tf.Graph()
with g.as_default():
	x = tf.add(3, 5)
sess = tf.Session(graph=g)
with tf.Session() as sess:
	sess.run(x)

   
#    不要把默认图与用户图混合使用

g = tf.Graph()
# add ops to the default graph
a = tf.constant(3)
# add ops to the user created graph
with g.as_default():
	b = tf.constant(5)

g1 = tf.get_default_graph()
g2 = tf.Graph()
# add ops to the default graph
with g1.as_default():
	a = tf.Constant(3)
# add ops to the user created graph
with g2.as_default():
	b = tf.Constant(5)

```





---------



# TensorBoard

tensoeflow是一个系统的软件，不仅提供深度学习算法架构，而且具有很多有用的工具，tensorboard是一个可视化工具，可以在执行一个确定的操作后，生成一个事件日志，然后进行可视化输出：

```python

import tensorflow as tf
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
#	在指定文件夹创建输出对象
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
	print(sess.run(x))
writer.close()
```

**然后执行操作并且激活可视板：**

```shell
$ python3 [my_program.py] 
$ tensorboard --logdir="./graphs" --port 6006
```

![tensor](http://media.innohub.top/180410-tboard.png)



----

# 内建类型

## 1.constant op

`tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)`

用于建立一个常量tensor

+ `dtype` 用于确定`value` 的类型，可以不定，可以由value的类型来确定

+ `ｓｈａｐｅ`　可选参数， 如果给定，就确定了产生的tensor的维度，比如［３，２］就是３x2的矩阵，［３，５，２］是３个５x2的矩阵的列表

+ `value` 可以是常量或者一个列表，长度必须小于或等于`shape`所定义的长度，如果小于时，最后一个元素会填充剩余位置

+  verify_shape:   Boolean that enables verification of a shape of values.

  ​

```python
>>> a = tf.constant([2,2], name='vector')
>>> b = tf.constant([56,656,3,6,3], shape=[3,5,2])
>>> with tf.Session() as sess:
...     sess.run(a)
...     sess.run(b)
... 

######################################################################
array([2, 2], dtype=int32)
array([[[ 56, 656],
        [  3,   6],
        [  3,   3],
        [  3,   3],
        [  3,   3]],

       [[  3,   3],
        [  3,   3],
        [  3,   3],
        [  3,   3],
        [  3,   3]],

       [[  3,   3],
        [  3,   3],
        [  3,   3],
        [  3,   3],
        [  3,   3]]], dtype=int32)
```

### 特定值常量

可以创建一个以特定值填充的，自定义维度的常量tensor

**以０填充：**

`tf.zeros()`:**初始化以０填充的tensor**

`tf.zeros_like`:**把传入的tensor变为相应维度的０tensor**

```python
tf.zeros(shape, dtype=tf.float32, name=None)
# create a tensor of shape and all elements are zeros
tf.zeros([2, 3], tf.int32) ==> [[0, 0, 0], [0, 0, 0]]


tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
# create a tensor of shape and type (unless type is specified) as the input_tensor but all elements are zeros.
# input_tensor [[0, 1], [2, 3], [4, 5]]
tf.zeros_like(input_tensor) ==> [[0, 0], [0, 0], [0, 0]]

```

**以１填充：**

```python
tf.ones(shape, dtype=tf.float32, name=None)
# create a tensor of shape and all elements are ones
tf.ones([2, 3], tf.int32) ==> [[1, 1, 1], [1, 1, 1]]



tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
# create a tensor of shape and type (unless type is specified) as the input_tensor but all elements are ones.
# input_tensor is [[0, 1], [2, 3], [4, 5]]
tf.ones_like(input_tensor) ==> [[1, 1], [1, 1], [1, 1]]
```

**一般用法：**

```python
tf.fill(dims, value, name=None) 
# create a tensor filled with a scalar value.
tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
```

**序列划分：**

+ `tf.lin_space(start, stop, num, name=None)`

  将区间［start, stop］(可以是减区间)，　平均划分为ｎｕｍ－１个区间，即得到一个num个元素的列表，是递增或递减序列

  **注意:num 必须为正整数,start, stop 必须为单精度浮点数，即小数位至少一位**

  ```python
  >>> b = tf.lin_space(13.0, 10.0, 4)
  >>> sess.run(b)
  array([13., 12., 11., 10.], dtype=float32)

  ```

+ `tf.range([start], limit=None, delta=1, dtype=None, name='range')`

  ```python
  # create a sequence of numbers that begins at start and extends by increments of delta up to but not including limit
  # slight different from range in Python

  # 'start' is 3, 'limit' is 18, 'delta' is 3
  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
  # 'start' is 3, 'limit' is 1,  'delta' is -0.5
  tf.range(start, limit, delta) ==> [3, 2.5, 2, 1.5]
  # 'limit' is 5
  tf.range(limit) ==> [0, 1, 2, 3, 4]
  ```

  **不像numpy，tensorflow 的序列是不可遍历的！！！！！！！！**

  ```python
  for _ in np.linspace(0, 10, 4): # OKfor _ in tf.linspace(0.0, 10.0, 4): # TypeError: 'Tensor' object is not iterable.for _ in range(4): # OKfor _ in tf.range(4): # TypeError: 'Tensor' object is not iterable.
  ```

  ****

## 2.算术运算

![math](http://media.innohub.top/180410-math.png)

**除法的神奇：**

```python
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
with tf.Session() as sess:
	print(sess.run(tf.div(b, a)))             ⇒ [[0 0] [1 1]]
	print(sess.run(tf.divide(b, a)))          ⇒ [[0. 0.5] [1. 1.5]]
	print(sess.run(tf.truediv(b, a)))         ⇒ [[0. 0.5] [1. 1.5]]
	print(sess.run(tf.floordiv(b, a)))        ⇒ [[0 0] [1 1]]
	print(sess.run(tf.realdiv(b, a)))         ⇒ # Error: only works for real values
	print(sess.run(tf.truncatediv(b, a)))     ⇒ [[0 0] [1 1]]
	print(sess.run(tf.floor_div(b, a)))       ⇒ [[0 0] [1 1]]
```

+ div:   符合python2.7语义，即x,y为整数时，结果为整数，xy为浮点数时，结果为浮点数,但是被除数不是浮点数时，除数不可以是浮点数，即y的类型需要和x保持一致

```python
>>> sess.run(tf.div(5.0, 2.0))
2.5
>>> sess.run(tf.div(5.0, 2))
2.5
>>> sess.run(tf.div(5, 2))
2
>>> sess.run(tf.div(5, 2.0))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Expected int32, got 2.0 of type 'float' instead.
    
```

+ divide  python3语义

+ truediv: 符合python3语义，所有的整数被先转换为浮点数，输出结果为浮点数

+ floordiv:  对整数而言和div一样，类似于python3　的 `x//y`  可以使用`tf.floor(tf.div(x, y))`  保证输出一定是一个整数，以防某些整数以浮点数表示

+ realdiv: 返回实际类型的x/y元素，如果xy是实数，则返回浮点数

  ```python
  >>> sess.run(tf.realdiv(56, 8))
  0
  >>> sess.run(tf.realdiv(56.0, 8))
  7.0
  >>> sess.run(tf.realdiv(56.1, 8))
  7.0125
  >>> sess.run(tf.realdiv(56.1, 8.0))
  7.0125
  ```

+ truncatediv:截断指定负数将分数量舍入到零。即-7 / 5 = -1。这匹配C语义，但它不同于Python语义