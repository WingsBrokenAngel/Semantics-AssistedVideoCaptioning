# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-4-28

import tensorflow as tf
from tensorflow import placeholder, glorot_normal_initializer, zeros_initializer
from tensorflow.nn import dropout
import numpy as np

n_z = 3584
n_y = 300
MSVD_PATH = None
MSRVTT_PATH = None
MSVD_GT_PATH = None
MSRVTT_GT_PATH = None
max_epochs = 1000
lr = 0.0002
batch_size = 128
keep_prob = 1.0
batch_size = 64



class TagNet():
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.y = placeholder(tf.float32, [None, n_y])
            self.z = placeholder(tf.float32, [None, n_z])
            self.keep_prob = placeholder(tf.float32, [])
            self.Wy1 = tf.get_variable('Wy1', [n_z, 512], tf.float32, glorot_normal_initializer())
            self.by1 = tf.get_variable('by1', [512], tf.float32, zeros_initializer())
            self.Wy2 = tf.get_variable('Wy2', [512, 512], tf.float32, glorot_normal_initializer())
            self.by2 = tf.get_variable('by2', [512], tf.float32, zeros_initializer())
            self.Wy3 = tf.get_variable('Wy3', [512, n_y], tf.float32, glorot_normal_initializer())
            self.by3 = tf.get_variable('by3', [n_y], tf.float32, zeros_initializer())

            z = dropout(self.z, self.keep_prob)
            h = tf.nn.relu(tf.matmul(z, self.Wy1) + self.by1)
            h = dropout(h, self.keep_prob)
            h = tf.nn.relu(tf.matmul(h, self.Wy2) + self.by2)
            h = dropout(h, self.keep_prob)

            self.pred = tf.sigmoid(tf.matmul(h, self.Wy3) + self.by3)

            cost = -self.y * tf.log(self.pred + 1e-6) - (1. - self.y) * tf.log(1. - self.pred + 1e-6)
            self.cost = tf.reduce_mean(tf.reduce_sum(cost, 1))

            self.pred_mask = tf.cast(self.pred >= 0.5, tf.int32)
            self.tmp = tf.cast(self.y, tf.int32)
            self.acc_mask = tf.cast(tf.equal(self.tmp, self.pred_mask), tf.float32)
            self.acc = tf.reduce_mean(self.acc_mask)






