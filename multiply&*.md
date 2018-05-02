## tensorflow　以及　numpy的乘法运算

## 1.	About Shape

在tf中定义的变量或者常量，一般需要给定shape参数或者根据参入的数据判断其形状,如果仅传入一个数据列表，未指明具体形状，则默认的shape为列表对象的长度

也可以在定义的时候指明形状，**通常情况下，定义一个constant常量，传入的参数列表中的value的值可以是一个常量或者列表，当列表长度小于所定义的shape元素数量时，最后一个元素自动填充剩余位置，但是当开启tf.contrib.eager的调试功能后，value长度小于shape时，会报错TypeError**

![error](http://media.innohub.top/180421-erroe.png)

```python
>>> a = tf.get_variable('v1', initializer=tf.constant([1,2,3,3,5]))
>>> a
<tf.Variable 'v1:0' shape=(5,) dtype=int32, numpy=array([1, 2, 3, 3, 5], dtype=int32)>
>>> b = tf.get_variable('v1', initializer=tf.constant([[1,2,3,3,5]]))
>>> b
<tf.Variable 'v1:0' shape=(1, 5) dtype=int32, numpy=array([[1, 2, 3, 3, 5]], dtype=int32)>
>>> b = tf.get_variable('v1', initializer=tf.constant([1,2,3,3,5], shape=[5,1])) 
>>> b
<tf.Variable 'v1:0' shape=(5, 1) dtype=int32, numpy=
array([[1],
       [2],
       [3],
       [3],
       [5]], dtype=int32)>
>>> d = tf.get_variable('v1', initializer=tf.constant([1,2,3,3,5], shape=[1, 5]))
>>> d
<tf.Variable 'v1:0' shape=(1, 5) dtype=int32, numpy=array([[1, 2, 3, 3, 5]], dtype=int32)>
```

**矩阵转置：**

`tf.transpose(a, perm=None, name='transpose', conjugate=False)`

> 用于进行矩阵的转置，perm参数用于指定需要转置的秩
>
> 对于二维矩阵可以进行默认的标准转置，对于多维矩阵就需要使用到perm指定待转置的秩，若不指定，则默认将所有秩倒序
>
> 比如shape=[2,3,4]的矩阵进行转置时，如果perm参数为空得到的矩阵的shape=[4,3,2]  perm=[0,2,1]时得到的shape=[2,4,3].可以将最内层的二维矩阵进行标准转置
>
> 设定conjugate 为True时，进行矩阵转置时如果是复数，则转化为其共轭复数，即实部相等，虚部互为相反数

```python
>>> a
<tf.Tensor: id=0, shape=(2, 2, 3), dtype=int32, numpy=
array([[[ 1,  2,  3],
        [ 4,  5,  6]],

       [[ 7,  8,  9],
        [10, 11, 12]]], dtype=int32)>
#	转置后的秩与之前相同
>>> b = tf.transpose(a, perm=[0,1,2])
>>> b
<tf.Tensor: id=3, shape=(2, 2, 3), dtype=int32, numpy=
array([[[ 1,  2,  3],
        [ 4,  5,  6]],

       [[ 7,  8,  9],
        [10, 11, 12]]], dtype=int32)>
#	进行标准内部转置
>>> bin = tf.transpose(a, perm=[0,2,1])
>>> b
<tf.Tensor: id=6, shape=(2, 3, 2), dtype=int32, numpy=
array([[[ 1,  4],
        [ 2,  5],
        [ 3,  6]],

       [[ 7, 10],
        [ 8, 11],
        [ 9, 12]]], dtype=int32)>

#	指定秩数的翻转
>>> b = tf.transpose(a, perm=[2, 0 ,1])
>>> b
<tf.Tensor: id=9, shape=(3, 2, 2), dtype=int32, numpy=
array([[[ 1,  4],
        [ 7, 10]],

       [[ 2,  5],
        [ 8, 11]],

       [[ 3,  6],
        [ 9, 12]]], dtype=int32)>
>>> b = tf.transpose(a, perm=[1,0,2])
>>> b
<tf.Tensor: id=12, shape=(2, 2, 3), dtype=int32, numpy=
array([[[ 1,  2,  3],
        [ 7,  8,  9]],

       [[ 4,  5,  6],
        [10, 11, 12]]], dtype=int32)>
#	不指定秩数，默认转置所有
>>> b = tf.transpose(a)
>>> b
<tf.Tensor: id=20, shape=(3, 2, 2), dtype=int32, numpy=
array([[[ 1,  7],
        [ 4, 10]],

       [[ 2,  8],
        [ 5, 11]],

       [[ 3,  9],
        [ 6, 12]]], dtype=int32)>

```



## ２．multiply 运算

###**在tf 或者np中使用＊运算时**

* ＊　means element-wise multiplication in Numpy，即逐元素相乘
* 进行乘法的对象必须在列数上相同，即shape的最后一个数字必须相同

```python
import numpy as np

A = np.ones([2, 3, 4])

B = np.ones([2, 4])

C = A*B[:, None, :]

## 可以将Ｂ转化为shape=[2, 4, 1]的数据，即可逐元素相乘
```

![mul](http://media.innohub.top/180421-mul.png)



### 使用tf.matmul

+ 进行矩阵乘法时，必须保证维数可乘　　h*k  必须与  k\*s　的进行运算

+ 对于高维矩阵相当于只有最后两维起作用，前面的维度可以作为batch，只有batch相同才可以运算

  例如：d

  ​	shape=[2, 3, 4]    shape=[2, 4, 3]  结果的shape=[2, 3, 3]

![mul](http://media.innohub.top/180421-matmul.png)



## 3. np array的截取操作

```python
# 
>>> a = np.array([[2,3,4,5,5], [1,2,3,4,5]])
#	截取列向量的操作
>>> a[:, 1]
array([3, 2])
>>> a[:, -1]
array([5, 5])
>>> a[::-1, 1]#	截取列向量并且倒序
array([2, 3])
#	复制操作
>>> a[:, :]
array([[2, 3, 4, 5, 5],
       [1, 2, 3, 4, 5]])
>>> a[:]
array([[2, 3, 4, 5, 5],
       [1, 2, 3, 4, 5]])
# 将［２，５］变为［２，　１　，５］
>>> a[:, None, :]
array([[[2, 3, 4, 5, 5]],

       [[1, 2, 3, 4, 5]]])
#	将[2, 5]变为［２，５，　１］
>>> a[:,:, None]
array([[[2],
        [3],
        [4],
        [5],
        [5]],

       [[1],
        [2],
        [3],
        [4],
        [5]]])
>>> print(a[:,:,None].shape)
(2, 5, 1)
>>> print(a[:,None, :].shape)
(2, 1, 5)

#	获取最外层分组
>>> a[1,:]
array([1, 2, 3, 4, 5])
>>> a[2,:]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 2 is out of bounds for axis 0 with size 2
>>> a[0,:]
array([2, 3, 4, 5, 5])
>>> a[1]
array([1, 2, 3, 4, 5])
>>> a[0]
array([2, 3, 4, 5, 5])


```

