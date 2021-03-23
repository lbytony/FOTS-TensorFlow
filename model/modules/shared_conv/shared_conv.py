#!-- encoding: utf-8 --
import os
import tensorflow as tf
from tensorflow import keras
from .res_block import ResBlock

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


class SharedConv(keras.Model):
    def __init__(self):
        super(SharedConv, self).__init__()
        pass

    def call(self, inputs, training=None):
        pass
