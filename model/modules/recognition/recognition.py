import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# TODO check is correct
class BidirectionalLSTM(keras.Model):
    def __init__(self, hidden_shape, out_shape):
        super(BidirectionalLSTM, self).__init__()
        self.lstm_cell_fwd = layers.LSTMCell(hidden_shape)
        self.lstm_cell_bwd = layers.LSTMCell(hidden_shape)
        self.bidir = layers.Bidirectional(self.lstm_cell_fwd, backward_layer=self.lstm_cell_bwd)
        self.embedding = layers.Dense(out_shape)

    def call(self, inputs, training=None, mask=None):
        out = self.bidir(inputs)
        out = self.embedding(out)
        out = tf.compat.v1.nn.bidirectional_dynamic_rnn
        out = tf.compat.v1.nn.rnn_cell.LSTMCell
        return out


class HeightMaxPool(keras.Model):
    def __init__(self, size=(2, 1), stride=(2, 1)):
        super(HeightMaxPool, self).__init__()
        self.pooling = layers.MaxPooling2D(pool_size=size, strides=stride)

    def call(self, inputs, training=None, mask=None):
        return self.pooling(inputs)


class BasicBlock(keras.Model):
    def __init__(self, filter_num):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.hmp = HeightMaxPool()

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.hmp(out)
        return out


class CNN(keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = BasicBlock(64)
        self.block2 = BasicBlock(128)
        self.block3 = BasicBlock(256)
        self.bilstm = BidirectionalLSTM(256, 256)

    def call(self, inputs, training=None, mask=None):
        out = self.block1(inputs)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bilstm(out)
        return out
