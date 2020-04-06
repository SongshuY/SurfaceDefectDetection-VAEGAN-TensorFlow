# coding=utf-8
import os
import numpy as np
import numpy.random as random
import tensorflow as tf
import cv2
import argparse
import re

def cmp_file_by_num(file):
    return int(re.sub("\D", "", file))

def tfwrite(writer, img, ispure):
    img = img.astype(np.uint8)
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[ispure])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())
    return writer

def creat_tf(imgpath):
    tfout_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],'..\\data')
    tfout_name = os.path.join(tfout_path, "test_encoder_pndata.tfrecords")
    writer = tf.python_io.TFRecordWriter(tfout_name)
    count = 0
    impure_imgpath = os.path.join(imgpath, 'impure')
    impure_picset = os.listdir(impure_imgpath)
    try:
        impure_picset = sorted(impure_picset, key=cmp_file_by_num)
    except:
        pass
    impure_num = len(impure_picset)
    for ind in range(impure_num):
        img_name = impure_picset[ind]
        img_name = os.path.join(impure_imgpath, img_name)
        img = cv2.imread(img_name)
        writer = tfwrite(writer, img, 0)
        print(count)
        count += 1

    pure_imgpath = os.path.join(imgpath, 'pure')
    pure_picset = os.listdir(pure_imgpath)
    try:
        pure_picset = sorted(pure_picset, key=cmp_file_by_num)
    except:
        pass
    pure_num = len(pure_picset)
    for ind in range(pure_num):
        img_name = pure_picset[ind]
        img_name = os.path.join(pure_imgpath, img_name)
        img = cv2.imread(img_name)
        writer = tfwrite(writer, img, 1)
        print(count)
        count += 1
    writer.close()


if __name__ == '__main__':
    pic_dir_name = os.path.join(os.path.split(os.path.realpath(__file__))[0],'out')
    creat_tf(imgpath=pic_dir_name)