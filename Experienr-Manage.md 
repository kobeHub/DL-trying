# 实验管理与可视化

## 1.共享变量与非共享变量

在训练深度网络时，为了减少训练参数的数量，或者是多机多卡并行化训练大数据模型往往需要共享变量．与此同时当一个模型十分复杂时，要保证这些变量名和操作名唯一不重复，同事维护一个条理清晰的graph十分重要

**变量共享的函数：**

+ `tf.get_variable()`: 拥有变量检查机制，会检测已经存在的变量是否设置为共享变量，如果已存在的变量未设置为共享变量则会报错
+ `tf.Variable()`：每次都会新建变量,底层实现引入别名机制

```python
>>> with tf.name_scope('con'):
...     v1 = tf.get_variable(name='weight', initializer=tf.zeros([5,6]))
...     v1 = tf.get_variable(name='weight', initializer=tf.zeros([3,4]))
... 
ValueError: Variable weight already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
r
  File "<stdin>", line 1, in <module>

####################################################################
def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights,
        strides=[1, 1, 1, 1], padding='SAME＇）
    return  tf.nn.relu(conv1 + conv1_biases)

# First call creates one set of 2 variables.
result1 = my_image_filter(image1)
# Another set of 2 variables is created in the second call.
result2 = my_image_filter(image2)
```

对于第二个实例，直接调用两次，每次都会产生两个变量，需要使用共享变量，但一定要保证共享变量的可共享性此时就需要`tf.variable_scope()`

## 2.tf.name_scope   and    tf.variable_scope

**name_scope 作用于操作，variable_scope 可以通过设置reuse 标志以及初始化方式来影响域下的变量。**

+ **name_scope:**  用于管理一个图中的各种操作，返回值是一个以scope_name(<scope_name>)命名的context manager ．一个graph会维护一个名称作用域的堆，避免各个op之间的冲突
+ **variable_scope:**  与name_scope配合使用，用于管理一个graph中的变量名称，允许共享变量

一般建立一个新的variable_scope时不需要设置reuse为真，只需要在使用时设置为true即可

```python
with tf.variable_scope("image_filters") as scope:
    result1 = my_image_filter(image1)
    scope.reuse_variables() # or 
    #tf.get_variable_scope().reuse_variables()
    result2 = my_image_filter(image2)
    
with tf.variable_scope("image_filters1") as scope1:
    result1 = my_image_filter(image1)
with tf.variable_scope(scope1, reuse = True)
    result2 = my_image_filter(image2)
```

通常情况下，tf.variable_scope 和 tf.name_scope 配合，能画出非常漂亮的流程图，但是他们两个之间又有着细微的差别，那就是 name_scope 只能管住操作 Ops 的名字，而管不住变量 Variables 的名字，看下例：

```ｐｙｔｈｏｎ
with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
assert v.name == "foo/v:0"
assert x.op.name == "foo/bar/add"
```

## 3. 保存和恢复模型

如果需要在一次训练结束后保存训练结果即训练得到的参数，可以使用tf的Saver类将数据保存为checkpoint文件，再次需要使用时，既可以读出．

+ Saver类添加操作从一个checkpoint*中保存和恢复变量，同时提供了方便的方法去执行操作
+ Checkpoint文件是一个特质形式的二进制文件将变量名映射到张量
+ 只要提供一个计数器，当计数器触发时，Saver类可以自动的生成checkpoint文件。这让我们可以在训练过程中保存多个中间结果。例如，我们可以保存每一步训练的结果。
+ 为了避免填满整个磁盘，Saver可以自动的管理Checkpoints文件。例如，我们可以指定保存最近的N个Checkpoints文件。

```python
saver.save(sess, 'my-model', global_step=0)  # 只保存初始状态的值　　'my-model-0'
...
# Create a saver.
saver = tf.train.Saver(...variables...)
# Launch the graph and train, saving the model every 1,000 steps.
sess = tf.Session()
for step in xrange(1000000):
    sess.run(..training_op..)
    if step % 1000 == 0:
        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=step)
```

### Saver类的基本操作

####１．`__init__`

```python
__init__(
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None
)
```

var_list确定了将会被存储的变量可以以列表（键值作为checkpoint的名字）或者字典（checkpoint的名字以操作名）的形式传入

```python
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
```

####2.`save`

```
save(
    sess,
    save_path,
    global_step=None,
    latest_filename=None,
    meta_graph_suffix='meta',
    write_meta_graph=True,
    write_state=True,
    strip_default_attrs=False
)
```

保存变量

该方法执行的操作可以存储变量，需要图中一个会话已经开启，待保存的变量已经经过初始化

The method returns the path prefix of the newly created checkpoint files. This string can be passed directly to a call to `restore()`.

#### 3.`restore`

```
restore(
    sess,
    save_path
)
```

Restores previously saved variables.

This method runs the ops added by the constructor for restoring variables. It requires a session in which the graph was launched. The variables to restore do not have to have been initialized, as restoring is itself a way to initialize variables.

The `save_path` argument is typically a value previously returned from a `save()` call, or a call to `latest_checkpoint()`.

#### Args:

- **sess**: A `Session` to use to restore the parameters. None in eager mode.
- **save_path**: Path where parameters were previously saved.

#### 4. 获取checkpoint文件状态

```python
tf.train.get_checkpoint_state(
    checkpoint_dir,　　　＃　指定文件夹
    latest_filename=None　＃　checkpoint文件的可选名
)
```