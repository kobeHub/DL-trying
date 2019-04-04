"""An eager mod of linear regression
"""
import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe
#import matplotlib.pyplot as plt

import time
import utils

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_FILE = 'data/birth_life_2010.txt'

tfe.enable_eager_execution()

data, n_sample = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

w = tfe.Variable(0.)
b = tfe.Variable(0.)

# Define the mod
def prediction(x):
    return x * w + b

def squard_loss(y, y_prediction):
    return (y - y_prediction) ** 2

def huber_loss(y, y_prediction, m=1.0):
    t = y - y_prediction
    return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

def train(loss_fn):
    print('Train loss fuction: {}'.format(loss_fn.__name__))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # Define the function through which to differentiate
    def loss_for_example(x, y):
        return loss_fn(y, prediction(x))

    # `grad_fn(x_i, y_i)` returns (1) the value of `loss_for_example`
    # evaluated at `x_i`, `y_i` and (2) the gradients of any variables used in
    # calculating it.
    grad_fn = tfe.implicit_value_and_gradients(loss_for_example)

    start = time.time()
    for epoch in range(100):
        total_loss = 0.0
        for x_i, y_i in tfe.Iterator(dataset):
            loss, gradients = grad_fn(x_i, y_i)
            optimizer.apply_gradients(gradients)
            total_loss += loss
        if epoch % 10 == 0:
            print('Epoch {}: {:.5f}'.format(epoch, total_loss/n_sample))

    print('Took {:.2f} seconds..'.format(time.time() - start))
    print('Eager execution exhibits significant overhead per operation. '
        'As you increase your batch size, the impact of the overhead will '
        'become less noticeable. Eager execution is under active development: '
        'expect performance to increase substantially in the near future!')

train(huber_loss)
plt.plot(data[:,0], data[:,1], 'bo')
# The `.numpy()` method of a tensor retrieves the NumPy array backing it.
# In future versions of eager, you won't need to call `.numpy()` and will
# instead be able to, in most cases, pass Tensors wherever NumPy arrays are
# expected.
#plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r',
#         label="huber regression")
#plt.legend()
#plt.show()
