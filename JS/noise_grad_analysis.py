#coding: utf-8

from __future__ import division, print_function, unicode_literals
import tensorflow as tf 
import numpy as np 
import math
FLAGS = tf.flags.FLAGS


def compute_privacy_cost(epsilon, delta=0.0):

    epsilon = FLAGS.nb_labels * epsilon # true epsilon per batch
    # use amplify theroem
    # q = FLAGS.batch_size / FLAGS.stdnt_share
    # e_ = math.log(1.0 + q * (math.exp(epsilon) - 1.0))
    
    # assert e_ <= epsilon
    
    # e = min(epsilon, e_)
    # delta = q * delta
    e = epsilon
    delta = delta
    # use composition theroem
    T = FLAGS.max_steps
    total_e = T * e * e + e * math.sqrt(2*T*math.log(1/(delta + 1e-14)))
    total_delta = delta
    return total_e, total_delta