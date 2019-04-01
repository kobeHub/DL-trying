import os
os.environ['TF_CPP_MIN_LOG_LEVAL'] = '2'
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
DATA_FILE = 'data/birth_life_2010.txt'
epoches = 100

data, n_samples = utils.read_birth_life_data(DATA_FILE)
X, Y = tf.placeholder(name='X', dtype=tf.float32), tf.placeholder(name='Y', dtype=tf.float32)

w = tf.get_variable('weight', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

y_pre = w * X + b

loss = tf.square(Y - y_pre, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

star = time.time()
writer = tf.summary.FileWriter('./graphs/liners_reg', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoches):
        total_loss = 0
        for x,y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X:x, Y: y})
            total_loss += l
        print(f'Epoch {i}: {total_loss/n_samples}')
                
    writer.close()
    w_out, b_out = sess.run([w, b])
print('Took {time.time() - star} seconds')
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
