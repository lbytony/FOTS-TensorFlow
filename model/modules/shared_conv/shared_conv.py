#!-- encoding: utf-8 --
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from base import BaseModel
from model.modules.shared_conv.resnet import FOTS_bbNet


def unpool(inputs):
    _, _, h, w = inputs.shape
    return tf.compat.v1.image.resize_bilinear(inputs, size=[h * 2, w * 2], align_corners=True)


def mean_image_subtraction(images, means=None):
    """
    image normalization
    :param images: bs * w * h * channel
    :param means:
    :return:
    """
    if means is None:
        means = [123.68, 116.78, 103.94]
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


class SharedConv(BaseModel):

    def __init__(self, config):
        super(SharedConv, self).__init__(config)
        self.backbone = FOTS_bbNet()

        self.merge1 = BasicUpLayer(128)
        self.merge2 = BasicUpLayer(64)
        self.merge3 = BasicUpLayer(32)
        self.merge4 = layers.Conv2D(32, (3, 3), padding='same')
        self.bn5 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

    def call(self, inputs, training=False, **kwargs):
        inputs = mean_image_subtraction(inputs)
        out, conv1, conv2, conv3 = self.backbone(inputs)
        # out = out
        out = unpool(out)
        out = self.merge1([conv1, out])
        out = unpool(out)
        out = self.merge2([conv2, out])
        out = unpool(out)
        out = self.merge3([conv3, out])

        out = self.merge4(out)
        out = self.bn5(out)
        out = self.relu(out)

        return out


class BasicUpLayer(keras.Model):

    def __init__(self, output_shape):
        super(BasicUpLayer, self).__init__()
        self.conv1 = layers.Conv2D(output_shape, (1, 1), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(output_shape, (3, 3), padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs: list, **kwargs):
        inputs = tf.concat(inputs, axis=1)
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
