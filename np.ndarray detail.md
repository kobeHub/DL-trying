# numpy.ndarray 详解

## １．内存结构　

```python
numpy.ndarray
numpy中用于表示一个多维数组的对象，包含了一些固定大小的同质元素的数组
```

> 为了提高科学计算的性能，`ndarray` 是可以用于存储单一数据类型的多维数据结构，它采用了预先编译好的ｃ语言代码，具有较好的性能表现
>
> **如果没有ndarray:**
>
> + 为了表示矩阵，就必须采用list或者array，但是list可以保存任何对象，所以实际上保存的是对象的额指针，这样据需要与数据空间数量相同的指针空间，浪费内存以及cpu计算时间
> + array可以直接保存数值，与c语言的数组相似，但是结构简陋，无法满足需求

其在内存中的结构如下：

![numpy_ndarray](http://danzhuibing.github.io/images/python_numpy_ndarray.png)

数据存储区域保存着数组中所有元素的二进制数据，dtype对象则知道如何将元素的二进制数据转换为可用的值。数组的维数、大小等信息都保存在ndarray数组对象的数据结构中。

strides中保存的是当每个轴的下标增加1时，数据存储区中的指针所增加的字节数。例如图中的strides为12,4，即第0轴的下标增加1时，数据的地址增加12个字节：即a[1,0]的地址比a[0,0]的地址要高12个字节，正好是3个单精度浮点数的总字节数；第1轴下标增加1时，数据的地址增加4个字节，正好是单精度浮点数的字节数。

## numpy的基本数据类型

NumPy内置了24种基本类型，基本上可以和C语言的数据类型对应上，其中部分类型对应为Python内置的类型。下表列举了常用NumPy基本类型。

| 类型      | 注释                      |
| --------- | ------------------------- |
| `bool_`   | 兼容Python内置的bool类型  |
| `bool8`   | 8位布尔                   |
| `int_`    | 兼容Python内置的int类型   |
| `int8`    | 8位整数                   |
| `int16`   | 16位整数                  |
| `int32`   | 32位整数                  |
| `int64`   | 64位整数                  |
| `uint8`   | 无符号8位整数             |
| `uint16`  | 无符号16位整数            |
| `uint32`  | 无符号32位整数            |
| `uint64`  | 无符号64位整数            |
| `float_`  | 兼容Python内置的float类型 |
| `float16` | 16位浮点数                |
| `float32` | 32位浮点数                |
| `float64` | 64位浮点数                |
| `str_`    | 兼容Python内置的str类型   |

24个scalar types并不是dtype，但是可以作为参数传递给`np.dtype()`构造函数产生一个dtype对象，如`np.dtype(np.int32)`。在NumPy中所有需要dtype作为参数的函数都可以使用scalar types代替，会自动转化为对应的dtype类型。

## Structured Array

由于**numpy**只支持单一数据类型，对于常见的表格型数据，我们需要通过**numpy**提供的Structrued Array机制自定义`dtype`。

定义结构化数组有四种方式：1) string, 2) tuple, 3) list, or 4) dictionary。推荐使用后两种方式。

```python
import numpy as np

# list方式：a list of tuples. Each tuple has 2 or 3 elements specifying: 1) The name of the field (‘’ is permitted), 2) the type of the field, and 3) the shape (optional)
persontype = [('name', np.str_), ('age', np.int16), ('weight', np.float32)]
  
# dict方式：需要指定的键值有names和formats
persontype = np.dtype({
        'names': ['name', 'age', 'weight'], 
        'formats': [np.str_, np.int16, np.float32]
    })
a = np.array([("Zhang", 32, 75.5), ("Wang", 24, 65.2)], dtype=persontype)
```

## 创建ndarray

共有三类基本方法：一是从Python内置的array-like数据结构转化得到；二是利用**numpy**提供的创建函数直接生成；三是使用`genfromtxt()`方法生成。

```python
# -*- coding: utf-8 -*-
import numpy as np

# 从python的list转换
x = np.array([[1,2.0],[0,0],(1+1j,3.)])

np.zeros((2, 3))
np.arange(2, 3, 0.1) # start, end, step
np.linspace(1., 4., 6) # start, end, num
np.indices((3, 3)) # 返回一个array，元素0是行下标，元素1是列下标；行下标为一个3*3二维array，对应3*3矩阵的行下标；列下标为一个3*3二维array，对应3*3矩阵的列下标

ndtype=[('a',int), ('b', float), ('c', int)]
names = ["A", "B", "C"]
np.genfromtxt("file_name.txt", 
  delimiter=",",
  names=names,
  dtype=ndtype, 
  autostrip=True,
  comments="#",
  skip_header=3, 
  skip_footer=5,
  usecols=(0, -1))

# ndarray to list
a = np.array([[1, 2], [3, 4]])
a.tolist()
```

关于分片不会引起copy操作：

  ```python
# 使用ｃopy显式克隆
>>> e = c.copy()
>>> e
array([[ 1. ,  1.6,  2.2],
       [-1. , -1. , -1. ]])
>>> c
array([[ 1. ,  1.6,  2.2],
       [-1. , -1. , -1. ]])
>>> e[-1] = 0
>>> e
array([[1. , 1.6, 2.2],
       [0. , 0. , 0. ]])
>>> c
array([[ 1. ,  1.6,  2.2],
       [-1. , -1. , -1. ]])

# 使用分片操作，虽然重新分配了内存，但是会同步更新各自的值
>>> e
array([[1. , 1.6, 2.2],
       [0. , 0. , 0. ]])
>>> f = e[:]
array([[1. , 1.6, 2.2],
       [0. , 0. , 0. ]])
>>> f[0, 0] = 5
>>> f
array([[5. , 1.6, 2.2],
       [0. , 0. , 0. ]])
>>> e
array([[5. , 1.6, 2.2],
       [0. , 0. , 0. ]])

  ```

