import cv2
import os
import argparse
import numpy as np
import re


def cmp_file_by_num(file):
    return int(re.sub("\D", "", file))


def deresolution(img, scale):
    m,n,_ = img.shape
    img_out = cv2.resize(img, (int(n*scale), int(m*scale)))
    return img_out


def reshape_input_height_image(img_input, height):
    m, n, _ = img_input.shape
    m_ = np.int(np.floor(m / height) * height)
    n_ = np.int(np.floor(n / height) * height)
    img_output = cv2.resize(img_input, (n_, m_))
    return img_output


def crop(img, row, column):
    m, n, _= img.shape
    begin = int(row*m)
    end = m-begin
    temp = img[begin:end, :, :]
    begin = int(column*n)
    end = n-begin
    img_out = temp[:,begin:end,:]
    return img_out


def cut_dense_image(img_input, height):
    half_height= height // 2
    m, n, _ = img_input.shape
    img_row = np.arange(0, m, half_height)[:-1]
    img_col = np.arange(0, n, half_height)[:-1]
    num_of_image = img_row.size * img_col.size
    assert m % height == 0 and n % height == 0
    image_batch = np.zeros((num_of_image, height, height, 3))
    rect_batch = np.zeros((num_of_image, 4))
    count = 0
    for i in img_row:
        for j in img_col:
            temp = img_input[i:i + height, :, :]
            img_cut = temp[:, j:j + height, :]
            if img_cut.shape == (height, height, 3):
                rect_batch[count] = (j, i, j+height, i+height)
                image_batch[count] = img_cut
                count += 1
    return image_batch.astype(np.uint8), rect_batch.astype(np.int)

def instruction():
    print("说明：")
    print("本程序用以切割图片以制作数据集")
    print("采用滑动窗口的方法人工切割")
    print("以下是程序参数：")
    print("--input_path：string, 必选参数，输入图片的绝对位置")
    print("--output_path：string, 必选参数，输出图片的绝对位置")
    print("--data_size：int, 必选参数，输入图片个数")
    print("--height：int, 可选参数，切割后的图片大小 [64]")
    print("-p：代表是正样本，不用-p代表负样本")
    print("以下是操作说明：")
    print("z：保存当前滑窗内图片   q：退出")
    print("i：滑窗向上移动一格   j：滑窗向左移动一格  k：滑窗向下移动一格  l：滑窗向右移动一格")
    print("；：滑窗向右边移动五格   ,：滑窗向下移动五格")
    print("d：下一张图片   a：上一张图片    c：指定图片")


if __name__ == '__main__':
    instruction()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="path of input data")
    # D:\\JiachenLu\\PycharmProjects\\VAEGAN-impurity-detection\\prepare_data_set\\defect
    parser.add_argument("--height", default=64, type=int, help="height of out data", action="store")
    parser.add_argument("-p", "--pure", action="store_true",
                        help="-p for pure, no -p for impure")
    args = parser.parse_args()
    in_name = args.input_path
    height = args.height
    out_name = os.path.join(os.path.split(os.path.realpath(__file__))[0],'out')
    print(args.pure)
    if args.pure:
        out_name = os.path.join(out_name, "pure")
    else:
        out_name = os.path.join(out_name, "impure")
    if not os.path.exists(in_name):
        assert False, "\"" + in_name + "\" dosen't exits!"
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    input_set = os.listdir(in_name)
    try:
        input_set = sorted(input_set, key=cmp_file_by_num)
    except:
        pass
    num_picture = len(input_set)
    defect_num = len(os.listdir(out_name))
    ind = 0
    exit_flag = False

    while ind < num_picture and not exit_flag:
        file_name = input_set[ind]
        img = cv2.imread(os.path.join(in_name, file_name))
        img_height = reshape_input_height_image(deresolution(crop(img, 0.3, 0.3), 0.5), height)
        m, n, _ = img_height.shape
        img_col = np.arange(0, n, height//2)[:-1]
        col_size = img_col.size
        img_bat, rect_bat = cut_dense_image(img_height, height)
        num_bat = img_bat.shape[0]
        jnd = 0
        next_flag = False
        while jnd < num_bat and not next_flag:
            bat = rect_bat[jnd]
            img_show = img_height.copy()
            cv2.rectangle(img_show, pt1=(bat[0], bat[1]),
                          pt2=(bat[2], bat[3]), color=(0, 255, 0), thickness=2)
            cv2.namedWindow(file_name)
            cv2.imshow(file_name, deresolution(img_show, 0.8))
            cv2.namedWindow('image')
            cv2.imshow('image', img_bat[jnd])
            wrong_flag = False
            key = cv2.waitKey(0)
            while not wrong_flag:
                if key == ord('z'):                                                     # 1号签：小
                    img_name = os.path.join(out_name, 'defect ('+str(defect_num)+').bmp')
                    defect_num += 1
                    wrong_flag = True
                    next_flag = False
                    cv2.imwrite(img_name, img_bat[jnd])
                    print('save at:', img_name)
                elif key == ord('j') and jnd != 0:
                    wrong_flag = True
                    next_flag = False
                    jnd -= 1
                elif key == ord('l') and jnd < num_bat:
                    wrong_flag = True
                    next_flag = False
                    jnd += 1
                elif key == ord(';') and jnd < num_bat-5:
                    wrong_flag = True
                    next_flag = False
                    jnd += 5
                elif key == ord('k') and jnd < num_bat-col_size:
                    wrong_flag = True
                    next_flag = False
                    jnd += col_size
                elif key == ord(',') and jnd < num_bat-col_size*5:
                    wrong_flag = True
                    next_flag = False
                    jnd += col_size*5
                elif key == ord('i') and jnd > col_size-1:                                                   # 2号签：中
                    wrong_flag = True
                    next_flag = False
                    jnd -= col_size
                elif key == ord('a') and ind != 0:                                      # 前一张
                    wrong_flag = True
                    ind -= 1
                    next_flag = True
                elif key == ord('d') and ind < num_picture:      # 后一张
                    wrong_flag = True
                    ind += 1
                    next_flag = True
                elif key == ord('c'):
                    print('jump to: ')
                    wrong_flag = True
                    try:
                        ind = int(input())
                    except:
                        pass
                    next_flag = True
                elif key == ord('q'):
                    wrong_flag = True
                    next_flag = True
                    exit_flag = True
                else:
                    wrong_flag = False
                    key = cv2.waitKey(0)
            cv2.destroyAllWindows()
        if jnd == num_bat:
            ind += 1
