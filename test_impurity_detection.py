import os
import numpy as np

from model.vaegan import VAEGAN
from utils.others import pp, show_all_variables, timestamp, expand_path

import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
tf.flags.DEFINE_float("data_size", np.inf, "The size of test images [np.inf]")
tf.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
tf.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.flags.DEFINE_float('reconst_v_gan', 1e-1, 'weight on DIS cost')
tf.flags.DEFINE_integer("batch_size", 4, "The size of batch images [4]")
tf.flags.DEFINE_integer("input_height", 64, "The size of image to use (will be center cropped). [108]")
tf.flags.DEFINE_integer("input_width", None,
                        "The size of image to use (will be center cropped). If None, same value as input_height [None]")
tf.flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
tf.flags.DEFINE_integer("output_width", None,
                        "The size of the output images to produce. If None, same value as output_height [None]")
tf.flags.DEFINE_integer("z_dim", 1024, "dimensions of z [1024]")
tf.flags.DEFINE_string("data_dir", "./data", "path to datasets [e.g. ./data]")
tf.flags.DEFINE_string("data_set", "test_impurity_detection.tfrecords", "name of data set [test_impurity_detection.tfrecords]")
tf.flags.DEFINE_string("out_dir", "./out", "Root directory for outputs [e.g. ./out]")
tf.flags.DEFINE_string("out_name", "",
                       "Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
tf.flags.DEFINE_string("checkpoint_dir", "checkpoint",
                       "Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
tf.flags.DEFINE_string("sample_dir", "detection_results", "Folder (under out_root_dir/out_name) to save samples [samples]")
tf.flags.DEFINE_boolean("export", False, "True for exporting with new batch size")
tf.flags.DEFINE_boolean("ispure", False, "True for exporting with new batch size")
tf.flags.DEFINE_boolean("freeze", False, "True for exporting with new batch size")
tf.flags.DEFINE_integer("max_to_keep", 1, "maximum number of checkpoints to keep")
tf.flags.DEFINE_integer("sample_freq", 200, "sample every this many iterations")
tf.flags.DEFINE_integer("ckpt_freq", 200, "save checkpoint every this many iterations")


def main(_):
    assert FLAGS.data_size is not np.inf, "please give the training or testing data size in arg"
    FLAGS.data_size = int(FLAGS.data_size)
    assert FLAGS.out_name is not "", "pleas give the out name to load"
    pp.pprint(FLAGS.__flags)

    # expand user name and environment variables
    FLAGS.data_dir = expand_path(FLAGS.data_dir)
    FLAGS.out_dir = expand_path(FLAGS.out_dir)
    FLAGS.out_name = expand_path(FLAGS.out_name)
    FLAGS.checkpoint_dir = expand_path(FLAGS.checkpoint_dir)
    FLAGS.sample_dir = expand_path(FLAGS.sample_dir)

    if FLAGS.output_height is None:
        FLAGS.output_height = FLAGS.input_height
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    # output folders
    if FLAGS.out_name == "":
        FLAGS.out_name = '{}-{}'.format(timestamp(), FLAGS.data_dir.split('/')[-1])  # penultimate folder of path
        if FLAGS.train:
            FLAGS.out_name += '-x{}.z{}.{}.y{}.b{}'.format(FLAGS.input_width, FLAGS.z_dim,
                                                           "VAE-GAN", FLAGS.output_width, FLAGS.batch_size)

    FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.out_name)
    FLAGS.checkpoint_dir = os.path.join(FLAGS.out_dir, FLAGS.checkpoint_dir)
    FLAGS.sample_dir = os.path.join(FLAGS.out_dir, FLAGS.sample_dir)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    print(FLAGS.checkpoint_dir)
    print(FLAGS.out_dir)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        vaegan = VAEGAN(
            sess,
            reconst_v_gan=FLAGS.reconst_v_gan,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            z_dim=FLAGS.z_dim,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            out_dir=FLAGS.out_dir,
            max_to_keep=FLAGS.max_to_keep)

        show_all_variables()
        vaegan.test_impurity_detection(FLAGS)


if __name__ == '__main__':
    tf.app.run()
