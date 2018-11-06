#coding: utf-8

from __future__ import division, print_function, unicode_literals

import tensorflow as tf 
import numpy as np 
from scipy.stats import entropy
import logging

log = logging.getLogger('tensorflow')
# log.setLevel(logging.DEBUG)
fh = logging.FileHandler('tensorflow.log')
# fh.setLevel(logging.DEBUG)
log.addHandler(fh)
FLAGS = tf.flags.FLAGS

def compute_loss(s, t, graph, name=None):
    '''
    s: student's probability vector, shape=(batch_size, 1, nb_labels)
    t: teacher's probability vector, shape=(batch_size, nb_teachers, nb_labels)
    '''
    with tf.name_scope(name, "ComputeLoss", [s, t]) as name:
        return py_func(forward_func,
                                [s, t],
                                [tf.float32],
                                graph,
                                name=name,
                                grad=backprop_func_batch_noise)
                                # grad=kl_backprop_func_noise)

def py_func(func, inp, Tout, graph, stateful=True, name=None, grad=None):
    
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    with graph.gradient_override_map({"PyFunc": rnd_name}):
        loss = tf.py_func(func, inp, Tout, stateful=stateful, name=name)[0]
        loss.set_shape([])
        return loss

def forward_func(s, t):
    '''
    compute Jenson-Shannon divergence of s and t
    s: shape=(batch_size, 1, nb_labels)
    t: shape=(batch_size, nb_teachers, nb_labels)
    '''
    m = 0.5 * (s + t)
    
    t = t.transpose((2,1,0))
    m = m.transpose((2,1,0))
    tm = entropy(t, m)
    t = t.transpose((2,1,0))

    s = s.transpose((2,1,0))
    sm = entropy(s, m)
    s = s.transpose((2,1,0))
    m = m.transpose((2,1,0))

    tm = tm.transpose((1,0))  # tm.shape=[batch_size, nb_teachers]
    sm = sm.transpose((1,0))

    tm = np.mean(tm)
    sm = np.mean(sm)

    loss = (0.5*tm + 0.5*sm)

    return loss.astype(np.float32)

def backprop_func(op, grad):
    
    s = op.inputs[0]
    t = op.inputs[1]

    denominator = tf.add(s, t, name='denominator')
    tmp = 2.0*s/(denominator+1e-14) + 1e-14
    d_ = tf.log(tmp, name='derivation')
    d_ = d_ / np.log(2.0)
    d = 0.5*tf.reduce_mean(d_, axis=1, name='reduce_mean')
    d = tf.expand_dims(d, 1)
    s_grad = d*grad/FLAGS.batch_size
    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())

    return s_grad, t_grad

def gaussian_noise(shape):
    '''
    produce gaussian noise, shape is tensor shape
    '''

    delta = FLAGS.delta 
    eps = FLAGS.epsilon 
    assert delta > 0, 'delta needs to be greater than 0'
    assert eps > 0, 'epsilon needs to be greater than 0'
    sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps
    stddev = sigma*FLAGS.dp_grad_C
    # print("stddev is ", stddev)
    # stddev = tf.Print(stddev, [stddev], "stddev: ")
    noise = tf.random_normal(shape=shape, mean=0.0, stddev=stddev)
    
    return noise

def laplace_noise(shape):
    '''
    produce laplace noise, shape is tensor shape
    '''

    lap_epsilon = FLAGS.lap_epsilon
    # delta_f = 1.0 + FLAGS.dp_grad_C # log(2s/(s+t)) <= 1.0
    delta_f = 0.5 + FLAGS.dp_grad_C # 0.5*log(2s/(s+t)) <= 0.5
    lap_scale = delta_f / lap_epsilon
    lap_dist = tf.distributions.Laplace(loc=0.0, scale= lap_scale)
    noise = lap_dist.sample(shape)
    return noise

def backprop_func_noise(op, grad):
    s = op.inputs[0]
    t = op.inputs[1]

    denominator = tf.add(s, t, name='denominator')
    tmp = 2.0*s/(denominator+1e-14) + 1e-14
    d_ = tf.log(tmp, name='derivation')
    d_ = d_ / np.log(2.0)

    # restrict d's element less or equal than dp_grad_C
    d_l2_norm = tf.sqrt(tf.reduce_sum(tf.multiply(d_, d_), axis=-1)) # [128, 100]
    d_max_l2norm = tf.reduce_max(d_l2_norm, axis=-1) # [128, 1]
    # d_max_l2norm = tf.Print(d_max_l2norm, [d_max_l2norm], message='d_max_l2norm: ', first_n=10, summarize=50)
    threshold = d_max_l2norm / FLAGS.dp_grad_C # [128,]
    threshold = tf.maximum(1.0, threshold) # [128,]
    threshold = tf.expand_dims(threshold, 1)
    threshold = tf.expand_dims(threshold, 2)
    threshold = tf.tile(threshold, [1, FLAGS.nb_teachers, FLAGS.nb_labels])
    d_ = d_ / threshold # [128, 100, 10]

    condition = tf.greater_equal(d_, -FLAGS.dp_grad_C)
    assert_op = tf.Assert(tf.reduce_any(condition), [condition])
    with tf.control_dependencies([assert_op]):
        # d_ = tf.Print(d_, [d_], message='d_: ', first_n=10, summarize=50)
        d = tf.reduce_sum(d_, axis=1, name='reduce_sum')
    if FLAGS.lap_epsilon:
        noise = laplace_noise(d.get_shape())
        # d = d + laplace_noise(d.get_shape())
        d = d + noise
    else:
        noise = gaussian_noise(d.get_shape())
        d = d + noise
        # d = d + gaussian_noise(d.get_shape())
    d = d / FLAGS.nb_teachers
    d = 0.5*d
    d = tf.expand_dims(d, 1)

    s_grad = d*grad/FLAGS.batch_size

    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())

    return s_grad, t_grad

def backprop_func_batch_noise(op, grad):
    s = op.inputs[0]
    t = op.inputs[1]

    denominator = tf.add(s, t, name='denominator')
    tmp = 2.0*s/(denominator+1e-14) + 1e-14
    d_ = tf.log(tmp, name='derivation')
    d_ = d_ / np.log(2.0)
    d_ = 0.5*d_

    # restrict d's element less or equal than dp_grad_C
    d_l2_norm = tf.sqrt(tf.reduce_sum(tf.multiply(d_, d_), axis=-1)) # [128, 100]
    d_max_l2norm = tf.reduce_max(d_l2_norm, axis=-1) # [128, 1]
    # d_max_l2norm = tf.Print(d_max_l2norm, [d_max_l2norm], message='d_max_l2norm: ', first_n=10, summarize=50)
    threshold = d_max_l2norm / FLAGS.dp_grad_C # [128,]
    threshold = tf.maximum(1.0, threshold) # [128,]
    threshold = tf.expand_dims(threshold, 1)
    threshold = tf.expand_dims(threshold, 2)
    threshold = tf.tile(threshold, [1, FLAGS.nb_teachers, FLAGS.nb_labels])
    d_ = d_ / threshold # [128, 100, 10]

    condition = tf.greater_equal(d_, -FLAGS.dp_grad_C)
    assert_op = tf.Assert(tf.reduce_any(condition), [condition])
    with tf.control_dependencies([assert_op]):
        # d_ = tf.Print(d_, [d_], message='d_: ', first_n=10, summarize=50)
        d = tf.reduce_mean(d_, axis=1, name='reduce_sum')
    if FLAGS.lap_epsilon:
        noise = laplace_noise(d.get_shape()[-1])
        # d = d + laplace_noise(d.get_shape())
    else:
        noise = gaussian_noise(d.get_shape()[-1])
        # d = d + gaussian_noise(d.get_shape())
    # d = d / FLAGS.nb_teachers
    # print(noise.get_shape())
    noise = noise / FLAGS.batch_size
    noise = tf.tile(noise, [FLAGS.batch_size])
    noise = tf.reshape(noise, [FLAGS.batch_size, -1])
    d = d + noise

    # d = 0.5*d #TODO
    d = tf.expand_dims(d, 1)

    s_grad = d*grad/FLAGS.batch_size

    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())

    return s_grad, t_grad

def backprop_func_random_response(op, grad):
    s = op.inputs[0]
    t = op.inputs[1]

    # random sampling
    means = tf.constant(0.0)
    sample = tf.where(tf.random_normal(t.get_shape()) - means < 0,
                    tf.ones(t.get_shape()), tf.zeros(t.get_shape()))
    
    sample = tf.Print(sample, [sample], "sample: ", first_n=10, summarize=100)

    denominator = tf.add(s, t, name='denominator')
    tmp = 2.0*s/(denominator+1e-14) + 1e-14
    
    tmp = tf.multiply(sample, tmp, name="multipy_sample")
    d_ = tf.log(tmp+1e-14, name='derivation')
    d_ = d_ / np.log(2.0)
    
    # sample = tf.ones(d_.get_shape())
    # d_ = tf.multiply(sample, d_, name="multiply_sample")
    d = 0.5*tf.reduce_mean(d_, axis=1, name='reduce_mean')
    d = tf.expand_dims(d, 1)
    s_grad = d * grad / FLAGS.batch_size
    
    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())
    return s_grad, t_grad
# def backprop_func(op, grad):
#     '''
#    wrong implementation, because use mean gradients over batch
#     compute student's probability vector derivation of loss
#     '''
#     s = op.inputs[0]
#     t = op.inputs[1]

#     s = tf.Print(s, [s], "s: ", first_n=10, summarize=10)
#     t = tf.Print(t, [t], "t: ", first_n=10, summarize=10)
#     # two = tf.constant(2.0, dtype=tf.float32)
#     denominator = tf.add(s, t, name='denominator')
#     # denominator = tf.Print(denominator, [denominator], "denominator: ")
#     tmp = 2.0*s/(denominator+1e-14) + 1e-14
#     tmp = tf.Print(tmp, [tmp], "tmp: ", first_n=10, summarize=10)
#     d_ = tf.log(tmp, name='derivation')
#     d_ = tf.Print(d_, [d_], "d_: ", first_n=10, summarize=10)
#     d_ = d_ / np.log(2.0)
#     # print(type(d_))
#     d = 0.5*tf.reduce_mean(d_, axis=[0, 1], name="reduce_mean")
#     # print(type(d))
#     # print(type(grad))

#     # print(d)
#     # grad = tf.Print(grad, [grad], "grad: ", first_n=5, summarize=10)
#     s_grad = d*grad
#     s_grad = tf.Print(s_grad, [s_grad], "s_grad: ", first_n=10, summarize=10)
#     # s_grad = tf.tile(s_grad, [s.get_shape()[0]])
#     # s_grad = tf.reshape(s_grad, s.get_shape())
#     s_grad = tf.tile(s_grad, [128]) / 128
#     s_grad = tf.reshape(s_grad, [128, 1, 10])
#     t_grad = tf.constant(np.zeros(10, dtype=np.float32))

#     # t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
#     # t_grad = tf.reshape(t_grad, t.get_shape())
#     t_grad = tf.tile(t_grad, [128*100])
#     t_grad = tf.reshape(t_grad, [128, 100, 10])
#     # s_grad = tf.Print(s_grad, [s_grad], "student grad: ")
#     return s_grad, t_grad

def kl_forward_func(s, t):
    t = t.transpose((2,1,0))
    s = s.transpose((2,1,0))
    kl_loss = entropy(t, s)
    kl_loss = kl_loss.transpose((1,0))
    t = t.transpose((2,1,0))
    s = s.transpose((2,1,0))
    kl_loss = np.mean(kl_loss)
    return kl_loss.astype(np.float32)

def kl_backprop_func_noise(op, grad):
    s = op.inputs[0]
    t = op.inputs[1]

    d_ = - t / (s+1e-14)

    # restrict d's element less or equal than dp_grad_C
    d_l2_norm = tf.sqrt(tf.reduce_sum(tf.multiply(d_, d_), axis=-1)) # [128, 100]
    d_max_l2norm = tf.reduce_max(d_l2_norm, axis=-1) # [128, 1]
    # d_max_l2norm = tf.Print(d_max_l2norm, [d_max_l2norm], message='d_max_l2norm: ', first_n=10, summarize=50)
    threshold = d_max_l2norm / FLAGS.dp_grad_C # [128,]
    threshold = tf.maximum(1.0, threshold) # [128,]
    threshold = tf.expand_dims(threshold, 1)
    threshold = tf.expand_dims(threshold, 2)
    threshold = tf.tile(threshold, [1, FLAGS.nb_teachers, FLAGS.nb_labels])
    d_ = d_ / threshold # [128, 100, 10]

    condition = tf.greater_equal(d_, -FLAGS.dp_grad_C)
    assert_op = tf.Assert(tf.reduce_any(condition), [condition])
    with tf.control_dependencies([assert_op]):
        # d_ = tf.Print(d_, [d_], message='d_: ', first_n=10, summarize=50)
        d = tf.reduce_mean(d_, axis=1, name='reduce_sum')
    if FLAGS.lap_epsilon:
        noise = laplace_noise(d.get_shape()[-1])
        # d = d + laplace_noise(d.get_shape())
    else:
        noise = gaussian_noise(d.get_shape()[-1])
        # d = d + gaussian_noise(d.get_shape())
    # d = d / FLAGS.nb_teachers
    # print(noise.get_shape())
    noise = noise / FLAGS.batch_size
    noise = tf.tile(noise, [FLAGS.batch_size])
    noise = tf.reshape(noise, [FLAGS.batch_size, -1])
    d = d + noise

    d = tf.expand_dims(d, 1)

    s_grad = d*grad/FLAGS.batch_size

    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())

    return s_grad, t_grad