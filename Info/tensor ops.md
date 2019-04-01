# tensorflow 中常用的operations

## 特性分析

+ tensoflow使用图来建立计算任务，
+ 在会话(session)的上下文(context)执行图中的操作，包括变量的实例化，ops的依次执行
+ 使用tensor表示数据
+ 使用variable来维护状态
+ 使用feed和fetch可以为任意的操作(arbitrary operation)赋值或者从中获取数据

## 综述

Tensorflow 是一个编程系统，使用图来表示计算任务，图中的节点被表示为ops,一个op获得０到多个tensor执行计算，输出一定数量的张量．一个tersorflow 描述了图的构建过程但是在执行时必须借助会话执行(或者是借助eager execution)　

由于其延迟加载的特性，需要进行张量的判断时就需要特定的操作来满足

## tf.cond 用于张量的条件判断

![ if...else..error](http://media.innohub.top/180523-error.png)

如图所示,无法把张量作为bool运算的条件，此时可以通过tf.cond来构建子图

```python
cond(pred, true_fn=None, false_fn=None, strict=False, name=None, fn1=None, fn2=None)
    Return `true_fn()` if the predicate `pred` is true else `false_fn()`. (deprecated arguments)
# 根据判断条件来执行相应的函数，为真时执行true_fn 否则执行false_fn
# 对于两个待执行的函数，返回值的类型，shape, batch相同
#　但是单值列表或者元组除外，他们被隐式解压为一个值，默认为
#	args:
#	strict: 为真是禁止隐式解压，默认为假

```

```python

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: x + y, lambda: x - y)
print(sess.run(out))

####################################################################
tf.greater(x, y, name=None)
	对张量x, y逐元素比较，返回（x > y）
```

## tf.case　用于switch程序控制

```python
case(pred_fn_pairs, default=None, exclusive=False, strict=False, name='case')
    Create a case operation.
    # nested 嵌套
#	对于多项选择的张量操作，可以选用tf.case
#	tf.case支持在`tensorflow.python.util.nest`中实现嵌套结构，所有可调用的对象返回列表
#	元组（可能嵌套）的结构相同
#	除了单值列表，会在strict参数为假时被隐式解压为单个值，但是参数为真时则不解压
Args:
#pred_fn_pairs:	张量运算得到的布尔条件以及其对应的将要执行的函数的二元组的字典
#default: 可选参数，相当于switch的default，当其他bool均为false时
#excultive: 当为真是，所有的张量比较条件都会进行运算，如果多于一个为真则会报错．如果为假则比较到第一个条件为真时即停止比较，并且执行相应的函数并返回，如果全部选项皆为假，则执行default

exclusive: 独家的，独有的 为真即表示仅识别一个
```

```python
f1 = lambda: tf.constant(17)
f2 = lambda: tf.constant(23)
r = case([(tf.less(x, y), f1)], default=f2)

def f1(): return tf.constant(17)
def f2(): return tf.constant(23)
def f3(): return tf.constant(-1)
r = case({tf.less(x, y): f1, tf.greater(x, z): f2},
         default=f3, exclusive=True)

```

## tf.where tf.gather  按条件查询

```python
where(condition, x=None, y=None, name=None)
    Return the elements, either from `x` or `y`, depending on the `condition`.
    
#    返回x或者y中的符合condition的元素
#	　当ｘ和ｙ均为none时，那么函数返回condition条件为真的元素的坐标，所有元素按照行主排列
#	　坐标以一个二维张量的形式返回，其中第一位代表的是条件为真的元素的数量，第二维代表的是元素的索引(indices)

#如果非None，`x`和`y`必须具有相同的形状。
#   如果`x`和`y`是标量，`condition`张量必须是标量。
#    如果`x`和`y`是更高级别的向量，则`condition`必须是大小与`x`的第一维相匹配的向量，或者必须具有与`x`相同的形状。`condition` tensor作为掩码，根据每个元素的值选择输出中相应的元素/行应该从“x”（如果为true）或“y”（如果为false）中取出。如果“condition”是矢量，`x`和`y`是更高级别的矩阵，则它选择从'x`和`y`复制哪一行（外部维度）。如果`condition`具有与`x`和`y`相同的形状，那么它会选择从`x`和`y`复制哪个元素。

```

**tf.gather根据索引获取元素：**

`gather(params, indices, validate_indices=None, name=None, axis=0)`
` Gather slices from params axis axis according to indices.`

参数indices必须是整数类型的张量可以是任意维度，通常为０维或者１维

```python
x = tf.get_variable('x1', initializer=tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
  			30.97266006,  26.67541885,  38.08450317,  20.74983215,
  			34.94445419,  34.45999146,  29.06485367,  36.01657104,
  			27.88236427,  20.56035233,  30.20379066,  29.51215172,
  			33.71149445,  28.59134293,  36.05556488,  28.66994858]))
	sess.run(x.initializer)
	indices = tf.where(x > 30)
	gather = tf.gather(x, indices)
	print(sess.run(indices), '\n',gather.eval())
##############################################################################
[[ 2]
 [ 4]
 [ 6]
 [ 8]
 [ 9]
 [11]
 [14]
 [16]
 [18]] 
 [[31.190735]
 [30.97266 ]
 [38.084503]
 [34.944454]
 [34.45999 ]
 [36.01657 ]
 [30.20379 ]
 [33.711494]
 [36.055565]]
```

## tf.diag 产生对角矩阵

```python
diag(diagonal, name=None)
	Return a diagonal tensor with a given diagonal values
```

```python

# 'diagonal' is [1, 2, 3, 4]
tf.diag(diagonal) ==> [[1, 0, 0, 0]
                       [0, 2, 0, 0]
                       [0, 0, 3, 0]
                       [0, 0, 0, 4]]
```
## tf.matrix_determinant 求矩阵行列式

```python
matrix_determinant(input, name=None)
    Computes the determinant of one or more square matrices.
    
    The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
    form square matrices. The output is a tensor containing the determinants
    for all input submatrices `[..., :, :]`.
```

