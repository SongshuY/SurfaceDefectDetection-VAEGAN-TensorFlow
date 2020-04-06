# coding=utf-8
import os
import numpy as np
import tensorflow as tf
import cv2
import argparse


def reshape64_image(img_input):
    m, n, _ = img_input.shape
    m_ = np.int(np.floor(m / 64) * 64)
    n_ = np.int(np.floor(n / 64) * 64)
    img_output = cv2.resize(img_input, (n_, m_))
    return img_output


def crop(img, row, column):
    m, n, _ = img.shape
    begin = int(row*m)
    end = m-begin
    temp = img[begin:end, :, :]
    begin = int(column*n)
    end = n-begin
    img_out = temp[:, begin:end, :]
    return img_out


def cut(writer, img):
    m, n, _ = img.shape
    img_row = np.arange(0, m, 64)
    img_col = np.arange(0, n, 64)
    counter = 0
    for i in img_row:
        for j in img_col:
            temp = img[i:i+64, :, :]
            img_cut = temp[:, j:j+64, :]
            if img_cut.shape == (64, 64, 3):
                writer = tfwrite(writer, img_cut)
                counter += 1
            else:
                pass
    return writer, counter


def deresolution(img, scale):
    m, n, _ = img.shape
    img_out = cv2.resize(img, (int(n*scale), int(m*scale)))
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
    tfout_name = os.path.join(tfout_path, "train.tfrecords")
    writer = tf.python_io.TFRecordWriter(tfout_name)
    bigcounter = 0
    picset = os.listdir(imgpath)
    img_num = len(picset)
    for ind in range(img_num):
        img_name = picset[ind]
        img_name = os.path.join(imgpath, img_name)
        img = cv2.imread(img_name)
        img = crop(img, 0.3, 0.3)
        writer, smallcounter = cut(writer=writer, img=reshape64_image(deresolution(img, 0.5)))
        bigcounter += smallcounter
        print(ind)
    writer.close()
    print("Train data has ", bigcounter, " samples.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="path of input data")
    args = parser.parse_args()
    pic_dir_name = args.input_path
    # pic_dir_name = "D:\\JiachenLu\\Matlab\\tobacoo\\picture\\none"
    creat_tf(imgpath=pic_dir_name)
