# 本文件定义了神经网络的卷积函数、转置卷积函数、全链结函数、leak relu激活函数、batch normalization等
# 可以在此处添加卷积神经网络的常用基础重复模块
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import tensorflow as tf

from utils.visualization import *


class BatchNorm(object):
    """
    batch normalization
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def mean_sigmoid_cross_entropy_with_logits(logit, truth):
    """
    交叉熵
    :param logit:   神经网络判断结果
    :param truth:   真实值
    :return:        交叉熵
    """
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logit,
            labels=truth * tf.ones_like(logit)))


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    """
    卷积神经网络
    :param input_:      输入tensor
    :param output_dim:  输出通道
    :param k_h:         卷积核高度
    :param k_w:         卷积核宽度
    :param d_h:         垂直步长
    :param d_w:         水平步长
    :param stddev:      初始化标准差
    :param name:        tensorflow变量域名
    :return:            卷积神经网络输出
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    """
    卷积神经网络
    :param input_:      输入tensor
    :param output_shape:输出图像大小
    :param k_h:         卷积核高度
    :param k_w:         卷积核宽度
    :param d_h:         垂直步长
    :param d_w:         水平步长
    :param stddev:      初始化标准差
    :param name:        tensorflow变量域名
    :param with_w       是否输出中间变量
    :return:            卷积神经网络输出
    """
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    """
    leak relu
    :param x:       输入tensor
    :param leak:    leak参数
    :param name:    tensorflow变量域名
    :return:        激活结果
    """
    with tf.name_scope(name):
        return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """
    全连接神经网络，无激活函数
    :param input_:          输入tensor
    :param output_size:     输出tensor大小
    :param scope:           tensorflow变量域名
    :param stddev:          初始化标准差
    :param bias_start:      bias初始化值
    :param with_w:          是否输出中间变量
    :return:                全链接结果
    """
    input_size = int(np.prod(input_.get_shape()[1:]))
    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("Matrix", [input_size, output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  " \
                  "Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        input_flat = tf.reshape(input_, [-1, input_size])
        if with_w:
            return tf.matmul(input_flat, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_flat, matrix) + bias


def gaussian_kld(mu1, lv1, mu2, lv2):
    """
    求两个高斯分布间的KL散度
    :param mu1: 第一个分布的平均值
    :param lv1: 第一个分布的方差
    :param mu2: 第二个分布的平均值
    :param lv2: 第二个分布的方差
    :return: KL散度
    """
    with tf.name_scope('GaussianKLD'):
        v1 = tf.exp(lv1)
        v2 = tf.exp(lv2)
        mu_diff_sq = tf.square(mu1 - mu2)
        dimwise_kld = .5 * (
                (lv2 - lv1) + tf.div(v1, v2) + tf.div(mu_diff_sq, v2) - 1.)
        return tf.reduce_sum(dimwise_kld, -1)


def gaussian_sample_layer(z_mu, z_lv, name='GaussianSampleLayer'):
    """
    高斯样本生成
    :param z_mu:    均值
    :param z_lv:    方差
    :param name:    变量域名
    :return:        输出N(z_mu, z_lv)的高斯分布
    """
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.multiply(eps, std))
