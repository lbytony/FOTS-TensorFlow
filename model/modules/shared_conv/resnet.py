#!-- encoding: utf-8 --
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers


class BasicResBlock(layers.Layer):
    # 残差模块
    def __init__(self, filter_num: list, stride=1):
        super(BasicResBlock, self).__init__()
        # 第一个卷积单元
        self.conv1 = layers.Conv2D(filter_num, (1, 1), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 第二个卷积单元
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        # 第三个卷积单元
        self.conv3 = layers.Conv2D(filter_num, (1, 1), strides=1, padding='same')
        self.bn3 = layers.BatchNormalization()
        # 用于短接层对齐
        if stride == 1:
            self.downsample = lambda x: x
        else:
            self.downsample = Sequential([
                layers.Conv2D(filter_num, (1, 1), strides=stride)
            ])

    def call(self, inputs, training=None):
        # [b, h, w, c]，通过第一个卷积单元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 通过第三个卷积单元
        out = self.conv3(out)
        out = self.bn3(out)
        # 2条路径输出直接相加
        downsample = self.downsample(inputs)
        output = layers.add([out, downsample])
        output = tf.nn.relu(output)  # 激活函数
        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, mode):
        super(ResNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64, (7, 7), strides=2),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D((3, 3), strides=2, padding='same')
        ])
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(10, activation='softmax')
        self.mode = True if mode == 'regular' else False

    def call(self, inputs, training=None, mask=None):
        # 通过根网络
        x = self.stem(inputs)
        # 一次通过4个模块
        x = self.layer1(x)
        res2 = x
        x = self.layer2(x)
        res3 = x
        x = self.layer3(x)
        res4 = x
        x = self.layer4(x)

        if self.mode:
            # 通过池化层
            x = self.avgpool(x)
            # 通过全连接层
            x = self.fc(x)
        print(x.shape, res2.shape, res3.shape, res4.shape)
        return x, res2, res3, res4

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicResBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicResBlock(filter_num, stride=1))
        return res_blocks


def ResNet50():
    return ResNet([3, 4, 6, 3], 'regular')


def FOTS_bbNet():
    return ResNet([3, 4, 6, 3], 'FOTS')
