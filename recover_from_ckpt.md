```python
>>> import os
>>> os.chdir('Python/learn/DL/checkpoints')
>>> import tensorflow as tf
>>> from tensorflow.python import pywrap_tensorflow
>>> ckpt = tf.train.get_checkpoint_state('convnet_layers')
>>> ckpt_p = ckpt.model_checkpoint_path
>>> ckpt_p
'convnet_layers/mnist-convnet-12900'
>>> reader = pywrap_tensorflow.NewCheckpointReader(ckpt_p)
>>> var = reader.get_variable_to_shape_map()
>>> for k in var:
...     print(k)
... 
logits/kernel/Adam_1
logits/kernel
logits/bias/Adam_1
conv2/bias/Adam
conv1/bias/Adam
conv2/kernel/Adam_1
conv2/bias
conv1/kernel
conv1/bias/Adam_1
conv1/kernel/Adam_1
beta2_power
conv1/bias
conv1/kernel/Adam
conv2/bias/Adam_1
conv2/kernel
conv2/kernel/Adam
logits/kernel/Adam
beta1_power
fc/kernel/Adam
global_step
fc/bias
fc/kernel
logits/bias
logits/bias/Adam
fc/bias/Adam
fc/bias/Adam_1
fc/kernel/Adam_1
>>> log = tf.get_variable('logits/kernel/Adam_1', shape=(1024, 10))
>>> with tf.Session() as sess:
...     saver = tf.train.Saver()
...     saver.restore(sess, ckpt_p)
... 

>>> with tf.Session() as sess:
...     sess.run(log.initializer)
...     sess.run(log)
... 
```

