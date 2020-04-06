# coding=utf-8
import os
import numpy as np
import tensorflow as tf
import cv2
import argparse

def deresolution(img, scale):
    m,n,_ = img.shape
    img_out = cv2.resize(img, (int(n*scale), int(m*scale)))
    # print(img_out[0,0:10,:].astype(np.float)/255)
    return img_out

def crop(img, row, column):
    m, n, _= img.shape
    begin = int(row*m)
    end = m-begin
    temp = img[begin:end,:, :]
    begin = int(column*n)
    end = n-begin
    img_out = temp[:,begin:end,:]
    return img_out

def tfwrite(writer, img):
    img = img.astype(np.uint8)
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())
    return writer

def creat_tf(imgpath):
    tfout_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..\\data')
    tfout_name = os.path.join(tfout_path, "test_impurity_detection.tfrecords")
    writer = tf.python_io.TFRecordWriter(tfout_name)
    picset = os.listdir(imgpath)
    img_num = len(picset)
    for ind in range(img_num):
        img_name = picset[ind]
        img_name = os.path.join(imgpath, img_name)
        img = cv2.imread(img_name)
        img = crop(img, 0.3, 0.3)
        writer = tfwrite(writer, deresolution(img, 0.5))
        print(ind)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="path of input data")
    args = parser.parse_args()
    pic_dir_name = args.input_path
    # pic_dir_name = "D:\\JiachenLu\\PycharmProjects\\impurity-detection\\defect3"
    creat_tf(imgpath=pic_dir_name)