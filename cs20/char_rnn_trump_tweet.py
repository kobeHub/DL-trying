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


## Define super arg of the model
HIDDEN_SIZE = [128, 256]
BATCH_SIZE = 64
LR = 0.0003
SKIP_STEP = 1
NUM_STEP = 50    #用于RNN展开
LENGTH = 200     # length of the generated text



class CharRNN:
    def __init__(self, model, hidden_size, barch_size, skip_step, length, 
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
        self.gstep = tf.get_variable('global', dtype=tf.int32, trainable=False, 
                initializer=tf.constant(0))

    def create_rnn(self, seq):
        layers = [keras.layers.GRUCell(size) for size in self.hidden_size]
        

