"""A char level generative language model
to generate Trump's style tweet"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import time
import sys
sys.path.append('..')

import tensorflow as tf
import keras

import utils


# Use one-hot encode rule
def encoder(text, vocab):
    """encode every character of a word according to
    vocab.
    :text input word
    """
    return [vocab.index(x) + 1 for x in text if x in vocab]

def decoder(array, vocab):
    """
    :array The code of the word
    """
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window, overlap):
    """将text划分为相同长度的序列，每一个序列的长度都为`window`,
    对于小于window的text填充`0`.进行划分每次移动的步长为overlap
    """
    lines  = [line.strip() for line in open(filename, 'r').readlines()]

    while True:
        random.shuffle(lines)

        for text in lines:
            text = encoder(text, vocab)
            for start in range(0, len(text) - window, overlap):
                chunk = text[start:start+window]
                chunk += [0] * (window - len(chunk))
                yield chunk

def read_batch(stream, batch_size):
    batch = []
    for ele in stream:
        batch.append(ele)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch


## Define super arg of the model
HIDDEN_SIZE = [128, 256]
BATCH_SIZE = 64
LR = 0.0003
SKIP_STEP = 1000
NUM_STEP = 50    #用于RNN展开
LENGTH = 200     # length of the generated text



class CharRNN:
    def __init__(self, model, hidden_size, batch_size, skip_step, length, 
            num_step, lr):
        self.model = model
        self.path = f'data/{model}.txt'
        if 'trump' in model:
            self.vocab = ("$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    " '\"_abcdefghijklmnopqrstuvwxyz{|}@#➡📈")
        else:
            self.vocab = (" $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "\\^_abcdefghijklmnopqrstuvwxyz{|}")

        self.seq = tf.placeholder(tf.int32, [None, None])    # input sequence
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.skip_step = skip_step
        self.num_step = num_step
        self.length = length
        self.temp = tf.constant(1.5)
        self.gstep = tf.get_variable('global', dtype=tf.int32, trainable=False, 
                initializer=tf.constant(0))

    def create_rnn(self, seq):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_size]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers) 
        batch = tf.shape(seq)[0]
        zero_states = cells.zero_state(batch, dtype=tf.float32)
        self.in_state = tuple([tf.placeholder_with_default(state,
                [None, state.shape[1]]) for state in zero_states])

        # All the seq were packed to the same length
        # now get the real length of the seq
        length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
        self.output, self.out_state = tf.nn.dynamic_rnn(cells, seq, length, self.in_state)

    def create_mode(self):
        seq = tf.one_hot(self.seq, len(self.vocab))
        self.create_rnn(seq)
        self.logits = tf.layers.dense(self.output, len(self.vocab), None)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits[:, :-1],
                labels=seq[:, 1:])
        self.loss = tf.reduce_sum(loss)

        # sample the next character from MaxWell-Boltzman Distribution
        # with temperature temp. It workd equally well without exp
        self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0]
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                global_step=self.gstep)


    def _create_summaries(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_ops = tf.summary.merge_all()


    def train(self):
        self._create_summaries()
        saver = tf.train.Saver()
        start = time.time()
        min_loss = None

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs/char_rnn', sess.graph)
            sess.run(tf.global_variables_initializer())

            # restore ckpt 
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(f'checkpoints/{self.model}/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()
            stream = read_data(self.path, self.vocab, self.num_step, self.num_step//2)
            data = read_batch(stream, self.batch_size)

            while True:
                batch = next(data)
            
                batch_loss, _, summary = sess.run([self.loss, self.opt, self.summary_ops], {self.seq: batch})
                writer.add_summary(summary, global_step=step)

                if (step + 1) % self.skip_step == 0:
                    print(f'Step {step} loss:{batch_loss}, time:{time.time() - start}')
                    self.online_infer(sess)
                    start = time.time()
                    ckpt_name = f'checkpoints/{self.model}/char_rnn'
                    if min_loss is None:
                        saver.save(sess, ckpt_name, step)
                    elif batch_loss < min_loss:
                        saver.save(sess, ckpt_name, step)
                        min_loss = batch_loss
                step += 1

    def online_infer(self, sess):
        for seed in ['Hillary', 'I', 'R', 'T', '@', 'N', 'M', '.', 'G', 'A', 'W']:
            sentence = seed
            state = None
            for _ in range(self.length):
                batch = [encoder(sentence[-1], self.vocab)]
                feed = {self.seq: batch}
                if state is not None: # for the first decoder step, the state is None
                    for i in range(len(state)):
                        feed.update({self.in_state[i]: state[i]})
                index, state = sess.run([self.sample, self.out_state], feed)
                sentence += decoder(index, self.vocab)
            print('\t' + sentence)



def main():
    model = 'trump_tweets'
    utils.safe_mkdir('checkpoints/' + model)

    lm = CharRNN(model, HIDDEN_SIZE, BATCH_SIZE, SKIP_STEP, LENGTH, NUM_STEP, LR)
    lm.create_mode()
    lm.train()

if __name__ == '__main__':
    main()



