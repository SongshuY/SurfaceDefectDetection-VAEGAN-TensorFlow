# 本文件包含数据集的解码和数据图片切割修整
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import tensorflow as tf
from skimage import transform

def parse_record(serialized_example):
    """
    tensorflow数据集解码
    解码训练数据集，没有标签，仅有神经网络训练的输入图片
    """
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 注意如果修改神经网络训练的输入大小需要修改大小
    img = tf.reshape(img, [64, 64, 3])
    # 转换为float32类型，并做归一化处理
    img = tf.cast(img, tf.float32) * (1. / 255)
    return img

def train_dis_parse_record(serialized_example):
    """
    tensorflow数据集解码
    解码测试数据集，包含标签，仅有神经网络训练的输入图片
    """
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 注意如果修改神经网络训练的输入大小需要修改大小
    img = tf.reshape(img, [64, 64, 3])
    # 转换为float32类型，并做归一化处理
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int64)
    return img,label

def test_parse_record(serialized_example):
    """
    tensorflow数据集解码
    解码demo数据集，没有标签，仅包含未切割过的大图，一定注意修改第本函数下三行代码处图像大小
    """
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 注意要修改此处，与图片大小需要一致
    img = tf.reshape(img, [600, 820, 3])
    # 转换为float32类型，并做归一化处理
    img = tf.cast(img, tf.float32) * (1. / 255)
    return img

def reshape_input_height_image(img_input, input_height=64):
    """
    修改图片大小使长宽使param input_height的整数倍
    :param img_input:       输入图片
    :param input_height:    神经网络训练的输入图片的高度
    :return:                改变大小的图片
    """
    m, n, _ = img_input.shape
    m_ = np.floor(m / input_height) * input_height
    n_ = np.floor(n / input_height) * input_height
    img_output = transform.resize(img_input, (m_, n_))
    return img_output

def cut_image2batch_ind(img_input,input_height=64):
    """
    生成滑动切割图片的切割点
    :param img_input:       输入图片
    :param input_height:    神经网络训练的输入图片的高度
    :return:                滑动切割图片的切割点
    """
    m, n, _ = img_input.shape
    img_row = np.arange(0, m, input_height // 2)[:-1]
    img_col = np.arange(0, n, input_height // 2)[:-1]
    assert m % input_height == 0 and n % input_height == 0, \
        'height and width of the input image must be multiple of var: input_height'
    return img_row, img_col

def cut_image2batch(img_input,input_height=64):
    """
    生成滑动切割图片的切割集
    :param img_input:       输入图片
    :param input_height:    神经网络训练的输入图片的高度
    :return:                滑动切割图片的切割结果
    """
    img_row, img_col = cut_image2batch_ind(img_input, input_height)
    num_of_image = img_row.size * img_col.size
    image_batch = np.zeros((num_of_image, input_height, input_height, 3))
    count = 0
    for i in img_row:
        for j in img_col:
            temp = img_input[i:i + input_height, :, :]
            img_cut = temp[:, j:j + input_height, :]
            if img_cut.shape == (input_height, input_height, 3):
                image_batch[count] = img_cut
                count += 1
    return image_batch

def deresolution(img, scale):
    """
    降低图像分辨率
    :param img:     输入图像
    :param scale:   降低倍率
    :return:        降低分辨率的图像
    """
    m, n, _ = img.shape
    img_out = cv2.resize(img, (int(n * scale), int(m * scale)))
    return img_out