# 数据类型

## 1. 基本数据类型：

 * `tf.float16`: 16-bit half-precision floating-point.

 * `tf.float32`: 32-bit single-precision floating-point.

 * `tf.float64`: 64-bit double-precision floating-point.

* `tf.bfloat16`: 16-bit truncated floating-point.
* `tf.complex64`: 64-bit single-precision complex.
* `tf.complex128`: 128-bit double-precision complex.
* `tf.int8`: 8-bit signed integer.
* `tf.uint8`: 8-bit unsigned integer.
* `tf.uint16`: 16-bit unsigned integer.
* `tf.uint32`: 32-bit unsigned integer.
* `tf.uint64`: 64-bit unsigned integer.
* `tf.int16`: 16-bit signed integer.
* `tf.int32`: 32-bit signed integer.
* `tf.int64`: 64-bit signed integer.
* `tf.bool`: Boolean.
* `tf.string`: String.
* `tf.qint8`: Quantized 8-bit signed integer.
* `tf.quint8`: Quantized 8-bit unsigned integer.
* `tf.qint16`: Quantized 16-bit signed integer.
* `tf.quint16`: Quantized 16-bit unsigned integer.
* `tf.qint32`: Quantized 32-bit signed integer.
* `tf.resource`: Handle to a mutable resource.
* `tf.variant`: Values of arbitrary types.

> tensorflow的DType类定义了其基本的数据类型，并且提供了类型比较的方法
>
> 如果other的ＤＴＹＰＥ可以转化为当前的类型则返回True

```python
 is_compatible_with(self, other)
      Returns True if the `other` DType will be converted to this DType.
      
      The conversion rules are as follows:
      
	   DType(T)       .is_compatible_with(DType(T))        == True
       DType(T)       .is_compatible_with(DType(T).as_ref) == True
       DType(T).as_ref.is_compatible_with(DType(T))        == False
       DType(T).as_ref.is_compatible_with(DType(T).as_ref) == True
 |      
 |      Args:
 |        other: A `DType` (or object that may be converted to a `DType`).
 |      
 |      Returns:
 |        True if a Tensor of the `other` `DType` will be implicitly converted to
 |        this `DType`.
 |  

```

## constant　的缺陷

constant存在创建的图里，并且在图形加载的任何地方复制，这就造成了在训练过程中如果需要大量的数据，就需要更多的时间，空间去加载，造成了运行速度下降



```python
my_const = tf.constant([1.0, 2.0], name="my_const")
with tf.Session() as sess:
	print(sess.graph.as_graph_def())
```

**图结构：**

![str](http://media.innohub.top/180410-node.png)

**This makes loading graphs expensive when constants are big**

## Variables

**1.A constant is, well, constant. Often, you’d want your weights and biases to be updated during training.**

**2.A constant's value is stored in the graph and replicated wherever the graph is loaded. A   variable is stored separately, and may live on a parameter server.**

### 创建变量

声明一个变量，可以通过创建一个`tf.Variables`类的实例，`tf.constant`是小写的因为constant是一个操作，而Variables是一个类包含多个操作

**Old ways:**

```python

s = tf.Variable(2, name="scalar") 
m = tf.Variable([[0, 1], [2, 3]], name="matrix") 
W = tf.Variable(tf.zeros([784,10]))

###################################################################
老方法不被提倡，不建议直接调用其构造函数，建议使用封装对象
tf.get_variable()
更便于实现变量的共享，利用该函数可以提供给变量内部名称，shape....

```

**New ways:**

```python

tf.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None
)


#	以tf.get_variable 创建变量
s = tf.get_variable("scalar", initializer=tf.constant(2)) 
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())
```

**变量的基本操作：**

```python
x = tf.Variable(...) 

x.initializer # init op
x.value() # read op
x.assign(...) # write op
x.assign_add(...) # and more
```

Separate definition of ops from computing/running ops 
Use Python property to ensure function is also loaded once the first time it is called*注意：**

变量声明后不可以直接用run调用，需要session 先初始化变量，之后才可以进行操作

```python
#	初始化所有变量
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
    
#	初始化变量组
with tf.Session() as sess :
    sess.run(tf.Variables_initializer([a, b]))
    
#	初始化单个变量
with tf.Session() as sess:
    sess.run(w.initializer())
    
#	输出函数可以用eval() 代替
print(w.eval())
```

### Assign 　操作

```python
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
	sess.run(W.initializer)
	print(W.eval()) 				# >> 10

--------

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
sess.run(W.initializer)
sess.run(assign_op)
print(W.eval()) 				# >> 100
```

> 在tensorflow 中每个会话维护自己的一份变量副本，一个会话的操作不影响变量的值，或者说变量的值只有在会话中才有意义，在会话外只有声明

> 注意在会话中执行操作时必须注意其依赖关系，初始化必须放在首位





## Placeholder 占位符

一个典型的ＴＦ程序具有以下两步：

* 创建一个图
* 使用会话执行操作

需要在不知道需要计算的值的情况下，建立图，就需要占位符的参与

`tf.placeholder(dtype, shape=None, name=None)`

```python
#	创建一个3元素向量占位符
a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b  # short for tf.add(a, b)

with tf.Session() as sess:
	print(sess.run(c，feed_dict={a:[1,2,3]}))	#与占位符运算时需要通过字典传入占位符的值 

    
# >> [6, 7, 8]

```

**定义占位符时，最好不要将shape=None，这样会使得很多操作无法进行，因为需要一个明确的范围，**



****

**可以将任何feedable的tensor，作为feed_dict的对象，通过以下方法检测：**

`tf.Graph/is_feedable(tensor)`

```python
# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
	# compute the value of b given a is 15
	sess.run(b, feed_dict={a: 15}) 				# >> 45
```



## 延迟加载　lazy-loading

如果在图完全声明所有操作前即使用会话运行操作则有很大的可能出现延迟加载，为保证正常加载

```
1.Separate definition of ops from computing/running ops 
2.Use Python property to ensure function is also loaded once the first time it is  called*
```

