# 本程序用来将npy转换为csv
# --path 为某次训练的输出文件夹，比如20200402.141647-data-x64.z1024.VAE-GAN.y64.b64
# 输入npy文件的名字为encoder_pndata_1024.npy
# 输出文件夹和npy在同一个文件夹下
import numpy as np
import os
import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path of input npy")
    args = parser.parse_args()
    data_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0],'..\\out')
    data_dir = os.path.join(data_dir, args.path)
    data_dir = os.path.join(data_dir, 'encoder_code')
    out = np.load(os.path.join(data_dir, 'encoder_pndata_1024.npy'))
    out_name = os.path.join(data_dir,'latent_space1024.csv')
    with open(out_name,'w',newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerows(out)