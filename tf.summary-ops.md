#　tf.summary 使用以及可视化

## 1. tf.summary概括

该模块可以提供张量的总结，并且提取信息到特定文件中，在训练中可以保存训练工程以及参数并且导出，配合tensorboard进行可视化展示

+ **`tf.summary.FileWriter()`类的操作:** 

  ```python
  __init__(
      logdir,
      graph=None,
      max_queue=10,
      flush_secs=120,
      graph_def=None,
      filename_suffix=None
  )
  logdir:   导出文件的文件夹
  graph:	要总结的图
  flush_secs: How often, in seconds, to flush the added summaries and events to disk.
  max_queue: Maximum number of summaries or events pending to be written to disk before one of the 'add' calls block.
  graph_def:	已废弃的参数改用为graph
  filename_suffix:	每个事件文件的后缀名称
  ```

  > FileWriter提供一种机制可以在特定文件夹创建event文件，并向其中添加event以及summaries．这个类可以提供自动更新，允许训练程序调用其方法在训练loop中向文件中直接添加数据而不需要减低速度

+ **`tf.sunmmary.FileWriterCache(logdir)`缓存类:**

  提供FileWriter的缓存操作，每个文件夹需要一个

## 2.常用函数

+ `tf.summary.scalar(name, tensor, collections=None, family=None)`:

  显示标量信息

  > - **name**: 即将产生的节点的名字，在TB上展示的名字
  > - **tensor**: 只包含一个值的实数张量
  > - **collections**: 图形集合键的可选列表。新的摘要操作被添加到这些集合中。默认为[GraphKeys.SUMMARIES]
  > - **family**: 可选的;如果提供，则用作汇总标签名称的前缀，该名称控制用于在Tensorboard上显示的标签名称。

  ```python
  #ops
  loss = ...
  tf.summary.scalar("loss", loss)
  merged_summary = tf.summary.merge_all()

  init = tf.global_variable_initializer()
  with tf.Session() as sess:
    writer = tf.summary.FileWriter(your_dir, sess.graph)
    sess.run(init)
    for i in xrange(100):
      _,summary = sess.run([train_op,merged_summary], feed_dict)
      writer.add_summary(summary, i)
  ```

  ​

+ `tf.summary.histogram(name, values, collections=None, family=None)`

  使用直方图输出到协议缓冲区，即以直方图的形式进行的summary

  > + name:upper
  > + values: 实数张量可以是任一形状，构建直方图
  > + collections, family：upper

+ `tf.summary.image(name, tensor, max_outputs=3, collections=None, family=None)`

  > 生成图片信息，产生相应的summary，summary最多有max_output的输出，图像由tensor构建，是一个四维张量[batch_size, height, width, channels]
  >
  > 还有一点就是，`TensorBord`中看到的`image summary`永远是最后一个`global step`的
  >
  > channels可以是：
  >
  > １　张量被解释为灰度
  >
  > ３　张量被解释为ＲＧＢ
  >
  > ４　张量被解释为ＲＧＢＡ

  > **参数：**
  >
  > + name: upper
  >
  > + tensor: 4-D tensor 可以是`uint8`或者`float32`   如果是无符号８位整数范围为[0, 255]
  >
  >   如果是浮点数３２位则将０．０转化为１２７　使得最小数字为０，最大为２５５
  >
  >   channels可以取值１　３　４代表不同的颜色表示方式

+ `tf.summary.sudio(name, tensor, sample_rate, max_outputs=3, collections=None, family=None)`

  生成训练过程中的音频信息，音频由tensor参数提供，是一个3-d张量，［batch_sizes, frames, channels］或者是一个二维张量［batch_sizes, frames］

   * If `max_outputs` is 1, the summary value tag is '*name*/audio'.

   * If `max_outputs` is greater than 1, the summary value tags are

      generated sequentially as '*name*/audio/0', '*name*/audio/1', etc

  > + max_outputs: 生成音频的最大批量的音频数量
  > + 一个类型为float32的标量指示信号的采集率，单位为Hz

+ `tf.summary.merge(inputs, collections=None, name=None).`

  将输入的summaries合并为一个文件

  #### Args:

  - **inputs**: A list of `string` `Tensor` objects containing serialized `Summary` protocol buffers.
  - **collections**: Optional list of graph collections keys. The new summary op is added to these collections. Defaults to `[]`.
  - **name**: A name for the operation (optional).

+ `tf.summary.merge_all(key='summaries', scope=None)`

  合并在默认图中收集到的所有Summaries

  ​

