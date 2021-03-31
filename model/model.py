import math

import tensorflow as tf
from tensorflow.keras import layers

from base.base_model import BaseModel
from utils.target_keys import N_CLASS
from .modules.recognition.recognition import CNN
from .modules.shared_conv.shared_conv import SharedConv


class FOTSModel:
    def __init__(self, config):
        self.sharedConv = SharedConv(config)
        self.detector = Detector(config)
        self.recognizer = Recognizer(N_CLASS, config)
        pass

    def summary(self):
        self.sharedConv.summary()
        self.detector.summary()
        self.recognizer.summary()


class Detector(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.scoreMap = layers.Conv2D(32, 1, kernel_size=1)
        self.geoMap = layers.Conv2D(32, 4, kernel_size=1)
        self.angleMap = layers.Conv2D(32, 1, kernel_size=1)

    def call(self, *input):
        final, = input

        score = self.scoreMap(final)
        score = tf.nn.sigmoid(score)

        geoMap = self.geoMap(final)
        # 出来的是 normalise 到 0 -1 的值是到上下左右的距离，但是图像他都缩放到  512 * 512 了，但是 gt 里是算的绝对数值来的
        geoMap = tf.nn.sigmoid(geoMap) * 512

        angleMap = self.angleMap(final)
        angleMap = (tf.nn.sigmoid(angleMap) - 0.5) * math.pi / 2

        geometry = tf.concat([geoMap, angleMap], axis=1)

        return score, geometry


class Recognizer(BaseModel):
    def __init__(self, n_class, config):
        super().__init__(config)
        self.cnn = CNN(8, 32, n_class, 256)

    def call(self, rois, lengths):
        return self.crnn(rois, lengths)
