import os
os.environ['TF_MIN_CPP_LOG_LEVEL'] = '2'

import tensorflow as tf
import time
import utils

# Super 
batch_size = 128
epoches = 30
lr = 0.01
n_train = 60_000
n_test = 10_000

# Get the mnist dataset
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Create datasets
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)  # random shuffle
test_data = tf.data.Dataset.from_tensor_slices(test)

train_data = train_data.batch(batch_size)
test_data = test_data.batch(batch_size)

# Builds an iterator for both the train and test data
iterator = tf.data.Iterator.from_structure(train_data.output_types,
        train_data.output_shapes)
img, label = iterator.get_next()

print(f'{img.shape} {label.shape}')
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# Build the computed graph
w = tf.get_variable('weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable('bias', shape=(1, 10), initializer=tf.zeros_initializer())

logits = tf.matmul(img, w) + b
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')

# Define train ops
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# Define test ops 
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

with tf.Session() as sess:
    start = time.time()
    sess.run(tf.global_variables_initializer())

    # Train model for epoches times
    for i in range(epoches):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print(f'Average loss epoch {i}: {total_loss / n_batches}')

    print(f'Train time: {time.time() - start}')

    # Test model 
    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print(f'Accuracy {total_correct_preds / n_test}')
writer.close()
