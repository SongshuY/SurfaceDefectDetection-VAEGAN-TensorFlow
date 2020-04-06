from __future__ import division
import pprint
import os
import time
import datetime

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()


def expand_path(path):
    return os.path.expanduser(os.path.expandvars(path))


def timestamp(s='%Y%m%d.%H%M%S', ts=None):
    if not ts:
        ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime(s)
    return st


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)