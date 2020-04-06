from __future__ import division
from __future__ import print_function
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import cv2
from ops.tensor import *
from utils.visualization import *
from utils.data_set import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class VAEGAN(object):
    def __init__(self, sess, input_height=64, input_width=64,
                 batch_size=64, output_height=64, output_width=64,
                 z_dim=1024, gf_dim=64, df_dim=64, reconst_v_gan=1e-1, c_dim=3, max_to_keep=1,
                 checkpoint_dir='ckpts', sample_dir='samples', out_dir='out'):
        """
        Args:
            sess: TensorFlow session
            batch_size:     批处理量
            input_height：   输入图像的高度 [64]
            input_width：    输入图像的宽度 [64]
            output_height：  输出图像的高度 [64]
            output_width：   输出图像的宽度 [64]
            z_dim：          编码空间的向量维度 [1024]
            gf_dim：         （不用改）生成器第一层输出的通道数 [64]
            df_dim：         （不用改）判别器第一层输出的通道数 [64]
            reconst_v_gan：  优化生成器的loss function中用来平衡生成器和判别器结果的参数和参数 [1e-1]
            c_dim：          生成的图片的通道数，彩色图是3，灰度图是1 [3]
            max_to_keep：    checkpoint保存回溯的最多次数 [1]
            out_dir：        checkpoint_dir和sample_dir在本文件夹下的位置 [out]
            checkpoint_dir： checkpoint在./out下保存的地址 [ckpts]
            sample_dir：     检查generator在训练中生成图像在./out中的位置 [samples]
        """
        self.sess = sess

        self.batch_size = batch_size
        self.reconst_v_gan = reconst_v_gan

        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.output_height = int(output_height)
        self.output_width = int(output_width)

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.ef_dim = df_dim

        self.checkpoint_dir = checkpoint_dir
        self.out_dir = out_dir
        self.max_to_keep = max_to_keep
        self.c_dim = c_dim
        self.sample_dir = sample_dir

        # Batch Normalization
        # discriminator部分
        self.d_bn1 = BatchNorm(name='d_bn1')
        self.d_bn2 = BatchNorm(name='d_bn2')
        self.d_bn3 = BatchNorm(name='d_bn3')
        # encoder部分
        self.e_bn0 = BatchNorm(name='e_bn0')
        self.e_bn1 = BatchNorm(name='e_bn1')
        self.e_bn2 = BatchNorm(name='e_bn2')
        self.e_bn3 = BatchNorm(name='e_bn3')
        self.e_bn_mu = BatchNorm(name='e_bn_mu')
        self.e_bn_lv = BatchNorm(name='e_bn_lv')
        # generator部分
        self.g_bn0 = BatchNorm(name='g_bn0')
        self.g_bn1 = BatchNorm(name='g_bn1')
        self.g_bn2 = BatchNorm(name='g_bn2')
        self.g_bn3 = BatchNorm(name='g_bn3')

        # 搭建神经网络结构，包括连接方式，loss function的定义，划定训练参数等
        # 输入图片
        self.inputs = tf.placeholder(
            tf.float32, [None, self.input_height, self.input_width, self.c_dim], name='real_images')
        # 隐空间向量
        self.z_direct = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')

        inputs = self.inputs
        self.z_mu, self.z_lv = self.encoder(inputs)
        # z满足N(0,1)
        self.z = gaussian_sample_layer(self.z_mu, self.z_lv)
        # xz从N(0,1)生成的图片
        self.xz = self.generator(self.z_direct)
        # logit_fake_xz是假图片的判断结果
        self.logit_fake_xz = self.discriminator(self.xz)
        # xz从真实图片经过编码之后在解码的图片
        self.xh = self.generator(self.z, reuse=True)
        # sample，注意设置reuse为True，train为False
        self.z_mu_sample, self.z_lv_sample = self.encoder(inputs, reuse=True, train=False)
        self.z_sample = gaussian_sample_layer(self.z_mu_sample, self.z_lv_sample)
        self.x_sample = self.generator(self.z_sample, reuse=True, train=False)

        # 对真实图片的判断
        self.logit_true = self.discriminator(inputs, reuse=True)
        # 对生成图片的判断
        self.logit_fake = self.discriminator(self.xh, reuse=True)

        # 真实图片要让判别器尽可能判断成真
        self.d_real_loss = mean_sigmoid_cross_entropy_with_logits(self.logit_true, 1.)
        # 生成图片要让判别器尽可能判断成假
        self.d_fake_loss = 0.5 * (
                mean_sigmoid_cross_entropy_with_logits(self.logit_fake, 0.) +
                mean_sigmoid_cross_entropy_with_logits(self.logit_fake_xz, 0.))
        # 生成器尽可能让图片被判别器判断为真
        self.g_fake_loss = 0.5 * (
                mean_sigmoid_cross_entropy_with_logits(self.logit_fake, 1.) +
                mean_sigmoid_cross_entropy_with_logits(self.logit_fake_xz, 1.))
        # 保证先验假设成立P(z|x) ~ N(0,1)
        self.kl_z_loss = tf.reduce_mean(
            gaussian_kld(self.z_mu, self.z_lv,
                         tf.zeros_like(self.z_mu), tf.zeros_like(self.z_lv)))
        # 生成图片和原图相似
        self.dis_loss = tf.reduce_mean(tf.square(self.xz - inputs))

        # 记录训练loss
        self.z_sum = tf.summary.histogram('z', self.z)
        self.logit_true_sum = tf.summary.histogram('D(true)', tf.nn.sigmoid(self.logit_true))
        self.logit_fake_sum = tf.summary.histogram('ED(Fake)', tf.nn.sigmoid(self.logit_fake))
        self.logit_fake_xz_sum = tf.summary.histogram('D(Fake)', tf.nn.sigmoid(self.logit_fake_xz))
        self.d_real_loss_sum = tf.summary.scalar('d_real_loss', self.d_real_loss)
        self.d_fake_loss_sum = tf.summary.scalar('d_fake_loss', self.d_fake_loss)
        self.g_fake_loss_sum = tf.summary.scalar('g_fake_loss', self.g_fake_loss)
        self.kl_z_loss_sum = tf.summary.scalar('kl_z_loss', self.kl_z_loss)
        self.dis_loss_sum = tf.summary.scalar('dis_loss', self.dis_loss)

        t_vars = tf.trainable_variables()
        # 挑选生成器所要训练的参数
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        # 挑选编码器所要训练的参数
        self.e_vars = [var for var in t_vars if 'e_' in var.name]
        # 挑选判别器所要训练的参数
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        # 判别器的loss
        self.obj_d = self.d_fake_loss + self.d_real_loss
        # 生成器的loss
        self.obj_g = self.g_fake_loss + self.dis_loss * self.reconst_v_gan
        # 编码器的loss
        self.obj_e = self.kl_z_loss + self.dis_loss
        # 保存参数用
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        self.writer = None
        self.summary_op = None

    def train(self, config):
        """
        :param config:  训练所需参数
        :return:        无
        """
        # 读取训练集
        filename = os.path.join(config.data_dir, config.data_set)
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parse_record)
        # 循环训练epoch下，shuffle打乱训练集，batch是minibatch的batchsize
        dataset = dataset.repeat(config.epoch).shuffle(config.batch_size*5).batch(config.batch_size)
        iterator = dataset.make_one_shot_iterator()
        img_batch = iterator.get_next()

        # 优化算法使用adam
        # 判别器的优化
        opt_d = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.obj_d, var_list=self.d_vars)
        # 编码器的优化
        opt_e = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.obj_e, var_list=self.e_vars)
        # 生成器的优化
        opt_g = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.obj_g, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        # 记录
        self.summary_op = tf.summary.merge([self.z_sum,
                                            self.logit_true_sum,
                                            self.logit_fake_sum,
                                            self.logit_fake_xz_sum,
                                            self.d_real_loss_sum,
                                            self.d_fake_loss_sum,
                                            self.g_fake_loss_sum,
                                            self.kl_z_loss_sum,
                                            self.dis_loss_sum])
        self.writer = tf.summary.FileWriter(os.path.join(self.out_dir, "logs"), self.sess.graph)

        counter = 1
        # 总共需要迭代的次数等于总的训练数据量除以批大小
        n_iter_per_epoch = config.data_size//config.batch_size
        start_time = time.time()
        # 断点重启
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        for epoch in xrange(config.epoch):
            # 每一个epoch
            for batch_it in xrange(n_iter_per_epoch):
                # 每一个batch
                # 读取batch内所有训练数据
                batch_x = self.sess.run(img_batch)
                # 生成正态分布
                batch_z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))

                # 训练判别器
                _, l_df, l_dr = self.sess.run(
                    [opt_d, self.d_fake_loss, self.d_real_loss],
                    feed_dict={self.inputs: batch_x, self.z_direct: batch_z})

                # 训练生成器两次
                _, l_g = self.sess.run(
                    [opt_g, self.g_fake_loss],
                    feed_dict={self.inputs: batch_x, self.z_direct: batch_z})
                _, l_g = self.sess.run(
                    [opt_g, self.g_fake_loss],
                    feed_dict={self.inputs: batch_x, self.z_direct: batch_z})

                # 训练编码器
                _, l_e, l_dis = self.sess.run(
                    [opt_e, self.kl_z_loss, self.dis_loss],
                    feed_dict={self.inputs: batch_x, self.z_direct: batch_z})

                msg = 'Time: {:.2f} '.format(time.time() - start_time)\
                      + 'Epoch [{:3d}/{:3d}] '.format(epoch + 1, config.epoch) \
                      + '[{:4d}/{:4d}] '.format(batch_it + 1, n_iter_per_epoch) \
                      + 'd_loss={:6.3f}+{:6.3f}, '.format(l_df, l_dr) \
                      + 'g_loss={:5.2f}, '.format(l_g) \
                      + 'KLD={:6.3f}, DIS={:6.3f}, '.format(l_e, l_dis)\

                print(msg)

                if np.mod(counter, config.sample_freq) == 0:
                    # 输出当前时刻训练好的图片
                    samples = self.sess.run(
                            self.x_sample,
                            feed_dict={
                                self.inputs: batch_x
                            }
                        )
                    # 保存两张图片，一张是原图，一张是生成图
                    print('./{}/{:08d}_inverter.png'.format(config.sample_dir, counter))
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/{:08d}_inverter.png'.format(config.sample_dir, counter))
                    print('./{}/{:08d}_origin.png'.format(config.sample_dir, counter))
                    save_images(batch_x, image_manifold_size(batch_x.shape[0]),
                                './{}/{:08d}_origin.png'.format(config.sample_dir, counter))
                    # 记录log
                    summary_str = self.sess.run(self.summary_op,
                                                feed_dict={self.inputs: batch_x, self.z_direct: batch_z})
                    self.writer.add_summary(summary_str, counter)

                if np.mod(counter, config.ckpt_freq) == 0:
                    self.save(config.checkpoint_dir, counter)
                counter += 1
        coord.request_stop()
        coord.join(threads)

    def test_encoder_pndata(self, config):
        """
        测试encoder对正负样本的编码情况，取名原因为：测试-编码器-p(ositive)n(egativ)data
        但是允许可以只有正样本或者只有负样本，仅用一种标签即可
        :param config:      必要的参数
        :return:            输出每张图像用encoder编码后的编码
                            numpy.array类型，size为N x z_dim，N为输入样本数
        """
        # 读取样本
        test_filename = os.path.join(config.data_dir, config.data_set)
        test_dataset = tf.data.TFRecordDataset(test_filename)
        test_dataset = test_dataset.map(train_dis_parse_record)
        # 仅循环一次，且不用打乱
        test_dataset = test_dataset.repeat(1).batch(config.batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()
        test_img_batch, test_label_batch = test_iterator.get_next()

        # 读取断点
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        # 总共需要迭代的次数等于总的数据量除以批大小
        n_iter_per_epoch = config.data_size // config.batch_size
        # 读取第一个batch，不写入循环是因为输出的out都是可变长度数组，需要初始化
        test_batch_x, test_batch_y = self.sess.run([test_img_batch, test_label_batch])
        print('[{:4d}/{:4d}] '.format(1, n_iter_per_epoch))
        z_out = self.sess.run(self.z_mu_sample, feed_dict={self.inputs: test_batch_x})
        label = test_batch_y
        for batch_it in xrange(n_iter_per_epoch-1):
            test_batch_x, test_batch_y = self.sess.run([test_img_batch, test_label_batch])
            z_t = self.sess.run(self.z_mu_sample, feed_dict={self.inputs: test_batch_x})
            print('[{:4d}/{:4d}] '.format(batch_it + 2, n_iter_per_epoch))
            z_out = np.vstack((z_out, z_t))
            label = np.vstack((label, test_batch_y))
        # np.save('impure_z_out_'+str(self.z_dim)+'.npy',z_out)
        # np.save('impure_z_out_label_'+str(self.z_dim)+'.npy',label)
        coord.request_stop()
        coord.join(threads)
        # 输出的格式为
        # [编码 标签]
        # e.g.
        # 0.1 0.2 0.3 ...... 0.2 | 1
        label = np.reshape(label, (label.size, 1))
        out = np.hstack((z_out, label))
        return out

    def test_impurity_detection(self, config):
        """
        图像异常点检测的一个demo，输入正常的拍摄完整图像，程序保存每张图像对应的检测结果
        检测结果为热图，从冷色到暖色，色泽越暖，有异物的概率越高
        :param config:      测试必要的参数
        :return:            无
        """
        # 读取数据
        filename = os.path.join(config.data_dir, config.data_set)
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(test_parse_record)
        # 仅循环一次，不要打乱，这里注意，由于输入数据集是原图，因此batch_size不能过大，设为1，2，4即可
        dataset = dataset.repeat(1).batch(config.batch_size)
        iterator = dataset.make_one_shot_iterator()
        img_batch = iterator.get_next()

        # 断点重读
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        # 总共需要迭代的次数等于总的数据量除以批大小
        n_iter_per_epoch = config.data_size // config.batch_size
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        count_img = 0
        for batch_it in xrange(n_iter_per_epoch):
            print("batch number: ", batch_it+1)
            batch_x = self.sess.run(img_batch)
            for ind in range(batch_x.shape[0]):
                img_input = batch_x[ind]
                # 修改输入图像的长宽至input_height的倍数
                img_input_resize = reshape_input_height_image(img_input, self.input_height)
                save_single_image(img_input_resize,
                                  './{}/{:08d}_test_ori.bmp'.format(config.sample_dir, count_img))
                img_bat = cut_image2batch(img_input_resize, self.input_height)
                # result：N x z_dim N是一张图切割出的小图数量
                result = self.sess.run(self.z_mu_sample, feed_dict={self.inputs: img_bat})
                # score：一张图切割出的小图所有的评估分数
                scores = np.ones(result.shape[0]).astype(np.int)
                for ind_score in range(result.shape[0]):
                    this_result = result[ind_score, :]
                    # 第一评判标准是阈值检测
                    gray1 = sum((this_result > 0.02).astype(np.int)) / 350
                    # 第二评判标准是峰值检测
                    gray2 = (max(this_result) > 0.2).astype(np.int)
                    if gray1 > 1:
                        gray1 = 1
                    elif gray1 < 0.5:
                        gray1 = 0
                    # 取两个评判标准的最大值
                    gray = max(gray1, gray2)
                    gray = gray * 255
                    scores[ind_score] = gray
                # 制作原图对应的热图
                img_output = make_heat_image(img_input_resize, 64, scores)
                print('./{}/{:08d}_test.bmp'.format(config.sample_dir, count_img))
                save_single_image(img_output,
                                  './{}/{:08d}_test.bmp'.format(config.sample_dir, count_img))
                count_img += 1
        coord.request_stop()
        coord.join(threads)

    def discriminator(self, image, reuse=False, train=True):
        """
        判别器
        :param image:   输入图像
        :param reuse:   参数是否是第一次使用，默认为是第一次
        :param train:   是否是训练模式，默认为训练模式
        :return:        最后一层全链接的输出，没有通过激活函数
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            # 输出维度是 64， 卷积核是5*5, 步长为2
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # 输出维度是 128， 卷积核是5*5, 步长为2
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv'), train=train))
            # 输出维度是 256， 卷积核是5*5, 步长为2
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), train=train))
            # 输出维度是 512， 卷积核是5*5, 步长为2
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv'), train=train))
            h4 = linear(h3, 1, 'd_h4_lin')  # 全连接，输出维度1
            return h4

    def generator(self, z, reuse=False, train=True):
        """
        生成器
        :param z:       输入编码
        :param reuse:   参数是否是第一次使用，默认为是第一次
        :param train:   是否是训练模式，默认为训练模式
        :return:        生成的图像
        """
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            # project `z` and reshape
            # 全连接 输出是512维
            z_, h0_w, h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
            h0 = tf.reshape(
                z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=train))
            h1, h1_w, h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(h1, train=train))
            h2, h2_w, h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2, train=train))
            h3, h3_w, h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3, train=train))
            h4, h4_w, h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
            # 输出是64x64x3
            return tf.nn.tanh(h4)

    def encoder(self, image, reuse=False, train=True):
        """
        编码器
        :param image:   输入图像
        :param reuse:   参数是否是第一次使用，默认为是第一次
        :param train:   是否是训练模式，默认为训练模式
        :return:        均值和方差
        """
        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(self.e_bn0(conv2d(image, self.ef_dim, name='e_h0_conv'), train=train))
            h1 = lrelu(self.e_bn1(conv2d(h0, self.ef_dim * 2, name='e_h1_conv'), train=train))
            h2 = lrelu(self.e_bn2(conv2d(h1, self.ef_dim * 4, name='e_h2_conv'), train=train))
            h3 = lrelu(self.e_bn3(conv2d(h2, self.ef_dim * 8, name='e_h3_conv'), train=train))
            # 全连接，输出维度是z_dim
            z_mu = lrelu(self.e_bn_mu(linear(h3, self.z_dim, 'e_hmu_lin'), train=train))
            # 全连接，输出维度是z_dim
            z_lv = lrelu(self.e_bn_lv(linear(h3, self.z_dim, 'e_hlv_lin'), train=train))
            return z_mu, z_lv

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step, filename='model', ckpt=True, frozen=False):
        """
        保存模型
        :param checkpoint_dir:  存档点地址
        :param step:            保留存档个数
        :param filename:        存档名称
        :param ckpt:            是否保存为ckpt模式， 默认为ckpt模式
        :param frozen:          是否冻结模型，默认不冻结
        :return:                无
        """
        filename += '.b' + str(self.batch_size)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if ckpt:
            self.saver.save(self.sess,
                            os.path.join(checkpoint_dir, filename),
                            global_step=step)

        if frozen:
            tf.train.write_graph(
                tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["generator_1/Tanh"]),
                checkpoint_dir,
                '{}-{:06d}_frz.pb'.format(filename, step),
                as_text=False)

    def load(self, checkpoint_dir):
        """
        加载模型
        :param checkpoint_dir:  加载模型地址
        :return:                无
        """
        print(" [*] Reading checkpoints...", checkpoint_dir)
        print("     ->", checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
