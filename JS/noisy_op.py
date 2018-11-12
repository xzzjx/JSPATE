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
                                # grad=backprop_func_noise)
                                # grad=backprop_func_batch_noise)
                                # grad=kl_backprop_func_noise)
                                # grad=backprop_func_bt)
                                # grad = vote_gradient)
                                grad = batch_vote_gradient)

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
    produce gaussian noise
    '''

    delta = FLAGS.delta 
    eps = FLAGS.epsilon 
    # print(shape)
    assert delta > 0, 'delta needs to be greater than 0'
    assert eps > 0, 'epsilon needs to be greater than 0'
    sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps
    delta_f = 0.5 + FLAGS.dp_grad_C
    stddev = sigma*delta_f*FLAGS.batch_size/FLAGS.nb_teachers
    # print("stddev is ", stddev)
    # stddev = tf.Print(stddev, [stddev], "stddev: ")
    # noise = tf.random_normal(shape=[shape], mean=0.0, stddev=stddev)
    normal_dist = tf.distributions.Normal(loc=0.0, scale=stddev)
    noise = normal_dist.sample(shape)
    
    return noise

def laplace_noise(shape):
    '''
    produce laplace noise, shape is tensor shape
    '''

    lap_epsilon = FLAGS.lap_epsilon
    # delta_f = 1.0 + FLAGS.dp_grad_C # log(2s/(s+t)) <= 1.0
    delta_f = 0.5 + FLAGS.dp_grad_C # 0.5*log(2s/(s+t)) <= 0.5
    delta_f = delta_f * FLAGS.batch_size / FLAGS.nb_teachers
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
        # d = tf.reduce_sum(d_, axis=1, name='reduce_sum')
        d = tf.reduce_mean(d_, axis=1, name='reduce_mean')
    if FLAGS.lap_epsilon:
        noise = laplace_noise(d.get_shape())
        # d = d + laplace_noise(d.get_shape())
        d = d + noise
    else:
        noise = gaussian_noise(d.get_shape())
        d = d + noise
        # d = d + gaussian_noise(d.get_shape())
    # d = d / FLAGS.nb_teachers
    # d = 0.5*d
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
        # print(d.get_shape())
    if FLAGS.lap_epsilon > 0:
        noise = laplace_noise(d.get_shape()[-1])
        # d = d + laplace_noise(d.get_shape())
    else:
        noise = gaussian_noise(d.get_shape().as_list()[-1])
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
        d = tf.reduce_mean(d_, axis=1, name='reduce_mean')
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


def laplace_noise_bt(shape):
    '''
    batch_size=nb_teachers
    one sample only ask one teacher
    '''
    delta_f = 0.5 + FLAGS.dp_grad_C
    b = delta_f / FLAGS.lap_epsilon
    lap_dist = tf.distributions.Laplace(0.0, b)
    return lap_dist.sample(shape)

def backprop_func_bt(op, grad):
    '''
    batch_size=nb_teachers
    one sample only ask one teacher
    '''
    s = op.inputs[0]
    t = op.inputs[1]
    # t[i,i,:]
    # t = t.transpose(2,1,0)
    t = tf.transpose(t, [2,1,0])
    diags_t = tf.map_fn(tf.diag_part, t)
    diags_t = tf.transpose(diags_t, [1,0])
    # diags_t = diags_t.transpose(2,1,0)
    # t = t.transpose(2,1,0)
    t = tf.transpose(t, [2,1,0])
    s = tf.squeeze(s, [1])
    denominator = tf.add(s, diags_t, name='denominator')
    # print(denominator.get_shape())
    tmp = 2.0*s/(denominator+1e-14) + 1e-14
    d_ = tf.log(tmp, name='derivation')
    d_ = d_ / np.log(2.0)
    d_ = 0.5*d_

    # restrict d's element less or equal than dp_grad_C
    d_l2_norm = tf.sqrt(tf.reduce_sum(tf.multiply(d_, d_), axis=-1)) # [128,]
    # d_max_l2norm = tf.reduce_max(d_l2_norm, axis=-1) # [128, 1]
    # d_max_l2norm = tf.Print(d_max_l2norm, [d_max_l2norm], message='d_max_l2norm: ', first_n=10, summarize=50)
    threshold = d_l2_norm / FLAGS.dp_grad_C # [128,]
    threshold = tf.maximum(1.0, threshold) # [128,]
    threshold = tf.expand_dims(threshold, 1)
    # threshold = tf.expand_dims(threshold, 2)
    threshold = tf.tile(threshold, [1, FLAGS.nb_labels])
    d_ = d_ / threshold # [128, 100, 10]

    condition = tf.greater_equal(d_, -FLAGS.dp_grad_C)
    assert_op = tf.Assert(tf.reduce_any(condition), [condition])
    # with tf.control_dependencies([assert_op]):
    #     # d_ = tf.Print(d_, [d_], message='d_: ', first_n=10, summarize=50)
    #     d = tf.reduce_mean(d_, axis=1, name='reduce_sum')
        # print(d.get_shape())
    if FLAGS.lap_epsilon:
        # noise = laplace_noise(d_.get_shape())
        noise = laplace_noise_bt(d_.get_shape())
        # d = d + laplace_noise(d.get_shape())
    else:
        noise = gaussian_noise(d_.get_shape())
        # d = d + gaussian_noise(d.get_shape())
    # d = d / FLAGS.nb_teachers
    # print(noise.get_shape())
    # noise = noise / FLAGS.batch_size
    # noise = tf.tile(noise, [FLAGS.batch_size])
    # noise = tf.reshape(noise, [FLAGS.batch_size, -1])
    with tf.control_dependencies([assert_op]):
        d = d_ + noise

    # d = 0.5*d #TODO
    d = tf.expand_dims(d, 1)

    s_grad = d*grad/FLAGS.batch_size

    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())

    return s_grad, t_grad

def bincount_(x):
    return tf.bincount(x, minlength=12)
def bincount(x):
    # return tf.bincount(x, minlength=FLAGS.nb_labels)
    return tf.map_fn(bincount_, x)
# def bincount_pyfunc(x):
#     result = np.zeros((x.shape[0], x.shape[1]))
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             bins = np.bincount(x[i][j], minlength=11)
#             result[i][j] = np.argmax(bins)
#     return result
def lap_noise_vote(shape):
    delta_f = 1
    epsilon = FLAGS.lap_epsilon
    b = delta_f / epsilon
    lap_dist = tf.distributions.Laplace(loc=0.0, scale=b)
    lap_noise = lap_dist.sample(shape)
    return lap_noise

def vote_gradient(op, grad):
    '''
    将梯度划分成11段，分别是1, 0, -1, -2, -3...-10
    每个teacher的梯度为最近的段点投票
    加噪音后输出段点的平均值作为该样本的梯度
    一个batch的梯度取平均值
    '''
    s = op.inputs[0] # shape=(128, 1, 10)
    t = op.inputs[1] # shape=(128, 100, 10)

    denominator = tf.add(s, t, name='denominator')
    tmp = 2.0*s/(denominator+1e-14) + 1e-14
    d_ = tf.log(tmp, name='derivation')
    d_ = d_ / np.log(2.0)
    # d_ = 0.5*d_   # shape=(128, 100, 10) # TODO

    d_ = tf.clip_by_value(d_, clip_value_min=-10.0, clip_value_max=1.0)
    # d_ = tf.round(d_) # shape=(128, 100, 10)
    ones = tf.ones_like(d_, dtype=tf.float32)
    # ones = tf.constant(np.ones([FLAGS.batch_size, FLAGS.nb_teachers, FLAGS.nb_labels]))
    # zeros = tf.constant(np.zeros([FLAGS.batch_size, FLAGS.nb_teachers, FLAGS.nb_labels]))    
    zeros = tf.zeros_like(d_, dtype=tf.float32)
    twos = ones * 2
    # val_list = [0.0, 1.0, -10.0]
    d_1 = tf.where(tf.greater(d_, 0.2), x=ones, y=zeros)
    d_2 = tf.where(tf.less(d_, -5.0), x=twos, y=zeros)
    d_ = d_1 + d_2
    # t_grad = tf.Print(d_, [d_], message="d_: ", first_n=10, summarize=10)
    # d = tf.ones_like(s)
    d_ = tf.cast(d_, tf.int32)
    # d_ = d_ + 10 # TODO
    # t_grad = tf.Print(d_, [d_], message="d_: ", first_n=10, summarize=10)
    
    d_ = tf.transpose(d_, [0, 2, 1])
    print(d_.get_shape())
    bins = tf.map_fn(bincount, d_) # shape=(128, 10, 12)
    print(bins.get_shape())
    bins = tf.cast(bins, tf.float32)
    print(bins.dtype)
    bins = bins + lap_noise_vote([FLAGS.batch_size, FLAGS.nb_labels, 3])
    d = tf.argmax(bins, axis=-1) # shape=(128, 10)
    # print(bins.get_shape())
    # d = d - 10
    ones = tf.ones_like(d)
    tens = ones * (-10)
    zeros = tf.zeros_like(d)
    d1 = tf.where(tf.equal(d, 1), ones, zeros)
    d2 = tf.where(tf.equal(d, 2), tens, zeros)
    d = d1 + d2
    d = tf.cast(d, tf.float32)
    d = 0.5 * d
    d = tf.expand_dims(d, 1)
    s_grad = d * grad / FLAGS.batch_size

    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())

    return s_grad, t_grad

def batch_vote_gradient(op, grad):
    s = op.inputs[0] # shape=(128, 1, 10)
    t = op.inputs[1] # shape=(128, 100, 10)
    # print(t.dtype)
    t = t+1e-14
    s = s+1e-14
    denominator = tf.add(s, t, name='denominator')
    tmp = 2.0*(s)/(denominator)
    d_ = tf.log(tmp, name='derivation')
    d_ = d_ / np.log(2.0)
    # zeros = tf.zeros_like(d_)
    # ones = tf.ones_like(d_)
    # d_ = tf.where(tf.equal(t, zeros), ones, d_)
    d_ = tf.clip_by_value(d_, clip_value_min=-10.0, clip_value_max=1.0)
    
    # compute every teacher's predict on mean value of batch's gradient
    d_t = tf.transpose(d_, [2, 1, 0])
    # print(d_t.get_shape())
    d_batch_mean = tf.reduce_mean(d_t, axis=-1) # (10, 100)
    # print(d_batch_mean.get_shape())
    d_t = tf.round(d_batch_mean)
    # print(d_t.get_shape())
    d_t = d_t + 10 # (10, 100)
    d_t = tf.cast(d_t, tf.int32)
    # print(d_t.get_shape())
    bins = tf.map_fn(bincount_, d_t) # (10, 12)
    # print(bins.get_shape())
    bins = tf.cast(bins, tf.float32)
    bins = bins + lap_noise_vote([FLAGS.nb_labels, 12])
    d_mean = tf.argmax(bins, axis=-1)
    d_mean = d_mean - 10
    d_mean = tf.cast(d_mean, tf.float32) # (10,)
    d_true_mean = tf.reduce_mean(d_batch_mean, axis=-1) # (10,)
    d_gap = d_mean - d_true_mean # (10,)
    d_ = tf.reduce_mean(d_, axis=1) # (128, 10)
    # print(d_.get_shape())
    # d_ = tf.transpose(d_, [1, 0])
    # print(d_.get_shape())
    d_ = d_ + d_gap
    # d_ = tf.transpose(d_, [1, 0])

    d_ = tf.expand_dims(d_, axis=1)
    s_grad = d_ * grad / FLAGS.batch_size


    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())

    return s_grad, t_grad
    
def batch_vote_gradient_sparse(op, grad):
    
    s = op.inputs[0] # shape=(128, 1, 10)
    t = op.inputs[1] # shape=(128, 100, 10)
    # print(t.dtype)
    t = t+1e-14
    s = s+1e-14
    denominator = tf.add(s, t, name='denominator')
    tmp = 2.0*(s+1e-14)/(denominator) + 1e-14
    d_ = tf.log(tmp, name='derivation')
    d_ = d_ / np.log(2.0)
    # zeros = tf.zeros_like(d_)
    # ones = tf.ones_like(d_)
    # d_ = tf.where(tf.equal(t, zeros), ones, d_)
    d_ = tf.clip_by_value(d_, clip_value_min=-10.0, clip_value_max=1.0)
    
    # compute every teacher's predict on mean value of batch's gradient
    d_t = tf.transpose(d_, [2, 1, 0])
    # print(d_t.get_shape())
    d_batch_mean = tf.reduce_mean(d_t, axis=-1) # (10, 100)
    # print(d_batch_mean.get_shape())
    d_t = tf.round(d_batch_mean)
    # print(d_t.get_shape())
    d_t = d_t + 10 # (10, 100)
    d_t = tf.cast(d_t, tf.int32)
    # print(d_t.get_shape())
    bins = tf.map_fn(bincount_, d_t) # (10, 12)
    # print(bins.get_shape())
    bins = tf.cast(bins, tf.float32)
    bins = bins + lap_noise_vote([FLAGS.nb_labels, 12])
    d_mean = tf.argmax(bins, axis=-1)
    d_mean = d_mean - 10
    d_mean = tf.cast(d_mean, tf.float32) # (10,)
    d_true_mean = tf.reduce_mean(d_batch_mean, axis=-1) # (10,)
    d_gap = d_mean - d_true_mean # (10,)
    d_ = tf.reduce_mean(d_, axis=1) # (128, 10)
    # print(d_.get_shape())
    # d_ = tf.transpose(d_, [1, 0])
    # print(d_.get_shape())
    d_ = d_ + d_gap
    # d_ = tf.transpose(d_, [1, 0])

    d_ = tf.expand_dims(d_, axis=1)
    s_grad = d_ * grad / FLAGS.batch_size


    t_grad = tf.constant(np.zeros(10, dtype=np.float32))

    t_grad = tf.tile(t_grad, [t.get_shape()[0]*t.get_shape()[1]])
    t_grad = tf.reshape(t_grad, t.get_shape())

    return s_grad, t_grad