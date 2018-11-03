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
                                grad=backprop_func_noise)

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
    s_grad = d*grad/FLAGS.nb_teachers
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

def backprop_func_noise(op, grad):
    s = op.inputs[0]
    t = op.inputs[1]

    denominator = tf.add(s, t, name='denominator')
    tmp = 2.0*s/(denominator+1e-14) + 1e-14
    d_ = tf.log(tmp, name='derivation')
    d_ = d_ / np.log(2.0)
    d = 0.5*tf.reduce_mean(d_, axis=1, name='reduce_mean')
    d = tf.expand_dims(d, 1)

    s_grad = d * grad
    s_grad = s_grad / tf.maximum(1.0, FLAGS.dp_grad_C)
    s_grad = s_grad + gaussian_noise(s_grad.get_shape())
    s_grad = s_grad/FLAGS.nb_teachers
    # stddev = FLAGS.noise_scale*FLAGS.dp_grad_C
    # s_grad = s_grad+tf.random_normal(shape=s_grad.get_shape, stddev=stddev)
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