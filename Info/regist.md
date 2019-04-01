# 基于tensorflow的基本逻辑回归

## 一．建造出生率与寿命预测模型

```SHEL
Dataset description:
Name: Birth rate-life expectancy in 2010
X = birth rate . Type float32h
Number of the datapoint:190
```

首先假设寿命和某地区的出生率呈线性关系，我们可以构建参数模型，使之满足`Y = w*X +b`

从而找到最适合于该模型的参数，达到预测的目的我们使用单层神经网络进行反向传播（backpropagation）,以方差作为损失函数，每一个时间段后，我们计算预测值与真实值之间的方差．

![img](https://lh4.googleusercontent.com/ip6bvydv2qMdP7FVYmf2A4fDn8OBaE5UgSu6bDCkGfDRunUzeTYwTJ083E-XU0TtpwLpAY6nqp3LANXzc_YE3vRwoacAs04MolXva4GoKcUY0aK-mfgGxE4aLSGVWg-4d1-mJxo1rdE)



```python
import tensorflow as tf

import utils

DATA_FILE = "data/birth_life_2010.txt"

# Step 1: read in data from the .txt file
# data is a numpy array of shape (190, 2), each row is a datapoint
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: construct model to predict Y (life expectancy from birth rate)
Y_predicted = w * X + b 

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
 
with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer()) 
	
	# Step 8: train the model
	for i in range(100): # run 100 epochs
		for x, y in data:
			# Session runs train_op to minimize loss
			sess.run(optimizer, feed_dict={X: x, Y:y}) 
	
	# Step 9: output the values of w and b
	w_out, b_out = sess.run([w, b]) 
```

如果待预测的关系是非线性的，比如二次函数，可以采用的损失函数可以是`y_ = wx*x +ux+b`

## 二．Huber Loss

根据以下数据分布图可知，有部分边缘数据，即底部低出生率，低平均寿命的数据，这些离群者将适配直线拉向他们，从而使模型表现很糟糕

+ 一个解决方法就是使用胡伯损失函数，使得离群者在模型中的作用权重远远减低
+ 以方差作为损失函数会使得离群者的偏离效果增大，因为使用了平方

****

![img](https://lh6.googleusercontent.com/HV1hB0PpKm_YsPKaDJVmBaGVq1weXTo2JDYV7FybckLF-lnfVV1a7Xzcki-1ryv5YPqv6JUvGJ-4WEmRZh6w6lYGXr9Z7SsINbKmM0MyBfRWHSAyWwYXUKYyu1kWyq94iN_XHmKV)

**Huber Loss:**



![img](https://lh6.googleusercontent.com/DMqLW7iyWVabRWqyu8sDNY7pE2pGtt5b58pn22qQZS6DCvuf_LvVOH_GLYuwtzu6oy3lY2EW6_ZK3Rk0NTPsDPLsrz35raSBNs-DjLFbNZbyTFeXPHsTxOMfx7y4SGsY9duGvuLe)



**使用方式:**

```python
if tf.abs(Y_predicted - Y) <= delta:
     # do something
        
### 这种方法只有当tensorflow的eager execution 起作用时才可以使用
### 否则会报TypeError

可以使用control flow 来操作：

tf.cond(
    pred,
    true_fn=None,
    false_fn=None,
    ...)


def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

```

## 三．　tf.data

### 1.	创建对象

> 使用feed_dict, placeholder 传递参数的好处在于，可以在tf外处理数据，使得数据更易分组，摆动，并且在python中生成任意数据
>
> 但是这种机制会降低程序性能

> Tensrflow 也提供了队列来处理数据，可以减少进行管道，线程操作的加载数据的时间，但是这中队列的易用性较低

> tf.data模块提供了比占位符更快，比队列更稳定的数据处理机制

使用tf.data数据存在一个tf对象中，一个`tf.data.Dataset`对象可以这样构建：

传入的参数应该是tensor对象，但是tf和numpy高度聚合，也可以传入ｎｐ对象

`tf.data.Dataset.from_tensor_slices((features, labels))`

```python

dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

print(dataset.output_types)			# >> (tf.float32, tf.float32)
print(dataset.output_shapes)		       # >> (TensorShape([]),TensorShape([]))



print(dataset.output_types)			# >> (tf.float32, tf.float32)
print(dataset.output_shapes)		       # >> (TensorShape([]), TensorShape([]))

dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

```

**You can also create a tf.data.Dataset from files using one of TensorFlow’s file format parsers, all of them have striking similarity to the old DataReader.tf.data.**

+ TextLineDataset(filenames): each of the line in those files will become one entry. It’s good for datasets whose entries are delimited by newlines such as data used for machine translation or data in csv files.
+ tf.data.FixedLengthRecordDataset(filenames): each of the data point in this dataset is of the same length. It’s good for datasets whose entries are of a fixed length, such as CIFAR or ImageNet.
+ tf.data.TFRecordDataset(filenames): it’s good to use if your data is stored in tfrecord format.

###　２．生成遍历器

```python
iterator = dataset.make_one_shot_iterator()
X, Y = iterator.get_next()         # X is the birth rate, Y is the life expectancy

with tf.Session() as sess:
	print(sess.run([X, Y]))		# >> [1.822, 74.82825]
	print(sess.run([X, Y]))		# >> [3.869, 70.81949]
	print(sess.run([X, Y]))		# >> [3.911, 72.15066]


for i in range(100): # train the model 100 epochs
        total_loss = 0
        try:
            while True:
                sess.run([optimizer]) 
        except tf.errors.OutOfRangeError:
            pass
```

树只可以提供一个时间段（epoch）的遍历作用，不需要实例化，是最简单易用的遍历器生成函数，时间段过后，达到数据的最后，无法重新实例化重复使用

为了在多个时段均可以使用可以采用`make_initializable_iterator()` 

**注意：调用初始化全局所有变量时并不会初始化遍历器，所以需要初始化**

`sess.run(iterator.initilizer`

`sess.run(tf.global_variable_initializer()` 

```python
iterator = dataset.make_initializable_iterator()
...
for i in range(100): 
        sess.run(iterator.initializer) 
        total_loss = 0
        try:
            while True:
                sess.run([optimizer]) 
        except tf.errors.OutOfRangeError:
            pass
```

## Optimizer

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
sess.run([optimizer]) 
```

optimizer 是一个使得损失函数最小的的操作，所以可以放在sess.run()的fetches参数中,当tf 的一个会话执行一个操作时．它会执行和这个操作相关的所有操作

默认情况下，优化器会训练模型中所有的可训练对象，可以再声明一个变量时确定其是否可以训练

```
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)
```

