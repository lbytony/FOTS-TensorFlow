from tensorflow import keras
from tensorflow.keras import layers

# TODO check is correct
from base import BaseModel


class BidirectionalLSTM(keras.Model):
    def __init__(self, hidden_shape, out_shape):
        super(BidirectionalLSTM, self).__init__()
        self.lstm_cell_fwd = layers.LSTM(hidden_shape, return_sequences=True, go_backwards=False,
                                         dropout=0.2, name="fwd_lstm")
        self.lstm_cell_bwd = layers.LSTM(hidden_shape, return_sequences=True, go_backwards=True,
                                         dropout=0.2, name="bwd_lstm")
        self.bilstm = layers.Bidirectional(layer=self.lstm_cell_fwd, merge_mode="concat",
                                           backward_layer=self.lstm_cell_bwd, name="bilstm")
        self.embedding = layers.Dense(out_shape)

    def call(self, inputs, training=None, mask=None):
        out = self.bilstm(inputs)
        out = self.embedding(out)
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


class CRNN(keras.Model):
    def __init__(self, n_class, n_hidden):
        super(CRNN, self).__init__()
        self.block1 = BasicBlock(64)
        self.block2 = BasicBlock(128)
        self.block3 = BasicBlock(256)
        self.bilstm = BidirectionalLSTM(n_hidden, n_class)

    def call(self, inputs, training=None, mask=None):
        out = self.block1(inputs)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bilstm(out)
        return out


class Recognizer(BaseModel):
    def __init__(self, n_class, config):
        super().__init__(config)
        self.crnn = CRNN(n_class, 256)

    def call(self, rois):
        return self.crnn(rois)
