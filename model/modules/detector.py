import math

import tensorflow as tf
from tensorflow.keras import layers

from base import BaseModel


class Detector(BaseModel):
    """
    文字检测模块负责将128x128x32的共享特征图进行文本检测操作
    """

    def __init__(self, config):
        super(Detector, self).__init__(config)
        self.scoreMap = layers.Conv2D(1, 1)
        self.geoMap = layers.Conv2D(4, 1)
        self.angleMap = layers.Conv2D(1, 1)

    def call(self, inputs, training=None, mask=None):
        score = self.scoreMap(inputs)
        score = tf.nn.sigmoid(score)

        geo = self.geoMap(inputs)
        geo = tf.nn.sigmoid(geo) * 512

        angle = self.angleMap(inputs)
        angle = (tf.nn.sigmoid(angle) - 0.5) * math.pi / 2

        geometry = tf.concat([geo, angle], axis=1)

        return score, geometry
