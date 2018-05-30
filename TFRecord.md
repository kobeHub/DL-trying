# TFRecord 进行流数据的读写

## What's TFRecord?

TFRecord可以允许你将任意的数据转换为tensorflow 支持的格式，这种方法使得tensorflow 的数据集更容易与网络应用架构相匹配.这种建议的方法是使用TFRecord文件．

TFRecord文件包含了［tf.train.Example］协议缓存块所定义的类，［protocol buffer］,protobuf将其序列化为一个字符串，并且通过[tf.python_io.TFRecordWriter]　写入到FTRecords文件中．

Binary file format :

​	A serialized tf.train.Example protobuf object   序列化的协议缓存对象，是一种google定义的类似xml格式

FTRecord文件格式在图像识别中有很好的作用，其可以将二进制文件和标签数据存储在同一个文件中．可以在模型进行训练之前通过预处理步骤将图像进行格式转换．因为其本质是二进制文件，不对数据进行压缩，可以快速的将数据加载到内存中，不支持随机访问，因此它适合大量的数据流，但是不适合快速分片或其它非连续存储的数据．

## Why Binary

+ 更好的利用磁盘缓存
+ 更快的进行转移
+ 可以同时处理不同格式的对象（可以把标签和图像放在一起）

## Convert to TFRcords

+ Feature: an inage
+ Label:a number

## TFRecordReader

从一个TFRecord文件中进行读取，不可以通过eager execution 计算，但是可以使用tf.data进行数据获取，本类继承自 ReaderBase

```python
__init__(self, name=None, options=None)
	name: 名称
    options: 可选参数，一个FTRecordOption对象
####################继承自BaseReader的方法###########################
num_records_produced(self, name=None)
	返回该阅读器生成的记录的数量(int64 tensor)，这与已成功执行读取操作的数量相同
num_records_units_completed(self, name=None)
	返回该阅读器完成处理的工作元数量(int64 tensor)
tf.TFRecordReader.read(queue, name=None)
	返回阅读器生成的下一条记录（键值对，一个元组（key, value））
    如果必要将从队列中的对一个工作单元进行排序（例：当读者需要从一个新的文件开始读取时，因为他已经完成了前面文件的操作）
    + queue: 表示句柄的队列或可变字符串张量到队列，带有字符串工作项目
reset()
	将一个文件阅读器置空
restore_state(state, name)
	恢复阅读器至先前保存的状态
serialize_state()
	产生一个字符串张量，它可以对一个阅读器状态进行编码
supports_serialize()
	是否可以对当前状态进行编码
```

## tf.TFRecordWriter 进行张量记录的写入

```python
# 创建一个writer写入tfr文件
writer = tf.python_io.TFRecordWriter(out_file)

#获取图片的序列化的shape以及value
shape, binary_image = get_image_binary(image_file)

# 创建特征对象
features = tf.train.Features(feature={'label': _int64_feature(label),
                                    'shape': _bytes_feature(shape),
                                    'image': _bytes_feature(binary_image)})

# 创建一个包含上文定义的特征的样例
sample = tf.train.Example(features=features)

# 把样例写入ftr文件
writer.write(sample.SerializeToString())   #  写入的是字符串对象
writer.close()
```

i为tfr格式：

```python
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _byte_feature(value):
	return tf.train.Feature(byte_list=tf.train.ByteList(value = [value]))

```

## 使用FTRecordDataset

```python
dataset = tf.data.TFRecordDataset(tfrecord_files)
dataset =  dataset.map(_parase_fun)    #将tfrecord转换为所需要的格式
#即一个三元组（label, shape, image）

def _parase_fun(tfrecord_serialized):
	features={'label': tf.FixedLenFeature([], tf.int64),
              'shape': tf.FixedLenFeature([], tf.string),
              'image': tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(tfrecord_serialized, features)

    return parsed_features['label'], parsed_features['shape'],                               parsed_features['image']

```

