"""
Word2vec skip-gram model with NCE loss
and visualize embedding in tfboard
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

import utils
import word2vec_utils

# Model hyperparameters
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128            # dimension of the word embedding vectors
SKIP_WINDOW = 1             # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000        # number of tokens to visualize


class SkipGramModel:
    """Build the graph for word2vec"""
    def __init__(self, dataset, vocab_size, embed_size, batch_size, num_sampled, lr):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = lr
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.skip_step = SKIP_STEP
        
    def _import_data(self):
        """Step 1:import data from dataset
        """
        with tf.name_scope('get_data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()

    def _create_embedding(self):
        """Step 2 + 3: define weights and embedding lookup
        The actually weights we care about
        """
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable('embed_matrix',
                    shape=[self.vocab_size, self.embed_size],
                    initializer=tf.random_normal_initializer())
            self.embed = tf.nn.embedding_lookup(self.embed_matrix,
                    self.center_words,
                    name='embedding')

    def _create_loss(self):
        """construct weights and loss functions for NCE"""
        with tf.name_scope('loss'):
            self.nce_weights = tf.get_variable('nce_weight',
                    shape=[self.vocab_size, self.embed_size],
                    initializer=tf.truncated_normal_initializer(stddev=1. / (self.embed_size ** 0.5)))
            self.nce_bias = tf.get_variable('nce_bias',
                    initializer=tf.zeros([self.vocab_size]))
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,
                    biases=self.nce_bias,
                    labels=self.target_words,
                    inputs=self.embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            # Merge all the summaries
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_train_steps):
        saver = tf.train.Saver()    ## The Saver to save variables including: embed_matrix, nce_weight, nce_bias

        initial_step = 0
        utils.safe_mkdir('checkpoints')

        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # if ckpt exist, restore from ckpt
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Calculate the last recent SKIP_STEP average loss
            total_loss = 0.0
            writer = tf.summary.FileWriter(f'graphs/word2vec/lr_{str(self.lr)}', sess.graph)
            initial_step = self.global_step.eval()

            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print(f'Average loss at step {index}: {total_loss/self.skip_step}')
                        total_loss = 0.0
                        saver.save(sess, 'checkpoints/skip-gram', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)   # run out of iterator

            writer.close()

    def visualize(self, visual_fd, num_visualize):
        """Run `tensorboard --logdir visualization` to get embeddings"""
        # ceate a list of most common words to visualize
        word2vec_utils.most_common_words(visual_fd, num_visualize)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)

            # Must store embedding in a new variable
            embeddings = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embeddings.initializer)

            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_fd)

            # add embedding to config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embeddings.name

            # link this tensor to its metadata file
            embedding_metadata_file = f'vocab_{str(num_visualize)}.tsv'

            # save the config file that the tensorboard will read
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embeddings])
            saver_embed.save(sess, os.path.join(visual_fd, 'model.ckpt'), 1)
         


def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, 
                                        BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)

def main():
    dataset = tf.data.Dataset.from_generator(gen, 
                                (tf.int32, tf.int32), 
                                (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    model = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    model.visualize(VISUAL_FLD, NUM_VISUALIZE)

if __name__ == '__main__':
    main()
