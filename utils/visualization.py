# 本文件包含项目的可视化代码，用以生成和保存训练中需要可视化的部分
from __future__ import division
from __future__ import print_function
import math
import pprint
import cv2
import numpy as np
import os
import time
import datetime
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.data_set import *

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return cv2.imwrite(path, (image * 255).astype(np.uint8))


def save_images(images, size, image_path):
    """
    组装并保存param size个数的图片，param size是完全平方数
    :param images:      所有的图片
    :param size:        图片个数
    :param image_path:  保存图片地址
    :return:
    """
    return imsave(images, size, image_path)

def save_single_image(image, path):
    """
    仅保存一张图片
    """
    return cv2.imwrite(path, (image * 255).astype(np.uint8))


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def inverse_transform(images):
    return (images + 1.) / 2.


def visualize(sess, dcgan, config, option, sample_dir='samples'):
    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim],
                    os.path.join(sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime())))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            save_images(samples, [image_frame_dim, image_frame_dim],
                        os.path.join(sample_dir, 'test_arange_%s.png' % (idx)))


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

def make_heat_image(img_input_resize, height, scores):
    """
    制作和原图相等比例的检测结果热图
    使用了滑动窗口，步长为param height的一半
    因此会造成重叠，重叠部分使用平均值
    :param img_input_resize:    输入图像，长宽必须是param height的倍数
    :param height:              截取的高度
    :param scores:              每个小图的分数
    :return:                    输出热图
    """
    m, n, _ = img_input_resize.shape
    assert m % height == 0 and n % height == 0, \
        'height and width of the input image must be multiple of param: height'
    img_row, img_col = cut_image2batch_ind(img_input_resize)
    img_output = np.zeros((img_input_resize.shape[0], img_input_resize.shape[1]))
    half_height = height // 2
    count = 0
    for i_bat in img_row:
        for j_bat in img_col:
            if i_bat == 0 and j_bat == 0:
                img_output[i_bat:i_bat + height, j_bat:j_bat + height] = scores[count]
            elif i_bat != 0 and j_bat == 0:
                origin = img_output[i_bat:i_bat + half_height, j_bat:j_bat + height]
                img_output[i_bat:i_bat + half_height, j_bat:j_bat + height] = (scores[count] + origin) // 2
                img_output[i_bat + half_height:i_bat + height, j_bat:j_bat + height] = scores[count]
            elif i_bat == 0 and j_bat != 0:
                origin = img_output[i_bat:i_bat + height, j_bat:j_bat + half_height]
                img_output[i_bat:i_bat + height, j_bat:j_bat + half_height] = (scores[count] + origin) // 2
                img_output[i_bat:i_bat + height, j_bat + half_height:j_bat + height] = scores[count]
            else:
                origin = img_output[i_bat:i_bat + height, j_bat:j_bat + half_height]
                img_output[i_bat:i_bat + height, j_bat:j_bat + half_height] = (scores[count] + origin) // 2
                origin = img_output[i_bat:i_bat + half_height, j_bat + half_height:j_bat + height]
                img_output[i_bat:i_bat + half_height, j_bat + half_height:j_bat + height] = (scores[
                                                                                                 count] + origin) // 2
                img_output[i_bat + half_height:i_bat + height, j_bat + half_height:j_bat + height] = scores[count]
            count += 1
    img_output = cv2.applyColorMap(img_output.astype(np.uint8), cv2.COLORMAP_JET)
    img_output = deresolution(img_output, 0.5)
    return img_output
