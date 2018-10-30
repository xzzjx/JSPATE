# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np 
from six.moves import xrange
import tensorflow as tf 
import math
import time
from datetime import datetime
import os

from differential_privacy.multiple_teachers import aggregation
from differential_privacy.multiple_teachers import deep_cnn
from differential_privacy.multiple_teachers import input
from differential_privacy.multiple_teachers import metrics
from differential_privacy.multiple_teachers import utils
# from differential_privacy.multiple_teachers import train_student
import gnjs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir','/tmp/train_dir','Where model chkpt are saved')
tf.flags.DEFINE_string('teachers_dir','/tmp/train_dir',
                       'Directory where teachers checkpoints are stored.')

tf.flags.DEFINE_integer('teachers_max_steps', 3000,
                        'Number of steps teachers were ran.')
tf.flags.DEFINE_integer('max_steps', 3000, 'Number of steps to run student.')
tf.flags.DEFINE_integer('nb_teachers', 10, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('stdnt_share', 1000,
                        'Student share (last index) of the test data')
tf.flags.DEFINE_integer('lap_scale', 10,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_boolean('save_labels', False,
                        'Dump numpy arrays of labels and clean teacher votes')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')
tf.flags.DEFINE_float('epsilon', 0.5, 'privacy cost of every student query')
tf.flags.DEFINE_float('delta', 0.1, 'loose param for privacy cost of every student query')
tf.flags.DEFINE_integer('heat', 1, 'distillation heat for computing teacher preds')
tf.flags.DEFINE_float('tm_coef', 0.5, 'coefficient of KL-divergence of teacher and m')
tf.flags.DEFINE_float('sm_coef', 0.5, 'coefficient of KL-divergence of student and m')


def prepare_student_data(dataset, nb_teachers):
    assert input.create_dir_if_needed(FLAGS.train_dir)

    # Load the dataset
    if dataset == 'svhn':
        test_data, test_labels = input.ld_svhn(test_only=True)
    elif dataset == 'cifar10':
        test_data, test_labels = input.ld_cifar10(test_only=True)
    elif dataset == 'mnist':
        test_data, test_labels = input.ld_mnist(test_only=True)
    else:
        print("Check value of dataset flag")
        return False

    # Make sure there is data leftover to be used as a test set
    assert FLAGS.stdnt_share < len(test_data)

    # Prepare [unlabeled] student training data (subset of test set)
    stdnt_data = test_data[:FLAGS.stdnt_share]
    stdnt_label = test_labels[:FLAGS.stdnt_share]
    stdnt_test_data = test_data[FLAGS.stdnt_share:]
    stdnt_test_labels = test_labels[FLAGS.stdnt_share:]

    return stdnt_data, stdnt_label, stdnt_test_data, stdnt_test_labels

def logit2prob(student_pred):
    '''
    change student logit to probability
    '''
    # batch_size = student_pred.shape[0]
    # nb_classes = student_pred.shape[1]
    # change logit to probability
    student_pred = tf.nn.softmax(student_pred)
    # make student_pred.shape consistant with teacher_preds.shape
    # student_pred = tf.reshape(student_pred, [batch_size, 1, FLAGS.nb_labels])
    student_pred = tf.expand_dims(student_pred, 1)
    
    return student_pred

def JS_part_fun(teacher_preds, student_pred):

    student_pred = logit2prob(student_pred)
    M = (teacher_preds+student_pred) / 2
    
    # compute JS divergence
    teacher_preds_dist = tf.distributions.Categorical(probs=teacher_preds)
    student_pred_dist = tf.distributions.Categorical(probs=student_pred)
    M_dist = tf.distributions.Categorical(probs=M)

    tm = tf.reduce_mean(tf.distributions.kl_divergence(teacher_preds_dist, M_dist))
    sm = tf.reduce_mean(tf.distributions.kl_divergence(student_pred_dist, M_dist))

    return tm, sm
def JS_loss_fun_util(teacher_preds, student_pred):
    '''
    使用JS散度计算teacher和student的JS散度
    Args:
        teacher_preds: numpy array, type=(batch_size, nb_teachers, nb_clssses)
        student_pred: tensor logit from inference, type=(batch_size, FLAG.nb_labels)
    '''
    # batch_size = student_pred.shape[0]
    # # nb_classes = student_pred.shape[1]
    # # change logit to probability
    # student_pred = tf.nn.softmax(student_pred)
    # # make student_pred.shape consistant with teacher_preds.shape
    # student_pred = tf.reshape(student_pred, [batch_size, 1, FLAGS.nb_labels])

    # student_pred = logit2prob(student_pred)
    # M = (teacher_preds+student_pred) / 2
    
    # # compute JS divergence
    # teacher_preds_dist = tf.distributions.Categorical(probs=teacher_preds)
    # student_pred_dist = tf.distributions.Categorical(probs=student_pred)
    # M_dist = tf.distributions.Categorical(probs=M)

    # pi1 = 0.99
    # pi2 = 0.01
    # tm = tf.distributions.kl_divergence(teacher_preds_dist, M_dist)
    # sm = tf.distributions.kl_divergence(student_pred_dist, M_dist)
    tm, sm = JS_part_fun(teacher_preds, student_pred)
    js_mean = FLAGS.tm_coef*tm + FLAGS.sm_coef*sm
    # js_mean = tf.reduce_mean(pi1*tf.distributions.kl_divergence(teacher_preds_dist, M_dist)  \
                    # + pi2*tf.distributions.kl_divergence(student_pred_dist, M_dist))
    tf.Assert(tf.greater_equal(js_mean, 0),
                    ['js divergence should be non-negative'])
    return js_mean
    # tf.add_to_collection('losses', js_mean)
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')
def JS_loss_fun(teacher_preds, student_pred):
    
    loss = JS_loss_fun_util(teacher_preds, student_pred)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def JS_loss_fun_noise(teacher_preds, student_pred):
    '''
    添加高斯噪音的teacher和student的JS散度
    '''
    loss = JS_loss_fun_util(teacher_preds, student_pred)
    delta = FLAGS.delta
    eps = FLAGS.epsilon
    assert delta > 0, 'delta needs to be greater than 0'
    assert eps > 0, 'epsilon needs to be greater than 0'


    # sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps
    # # TODO
    # # privacy_accum_op = gnjs.accumulate_privacy_spending(eps, delta, sigma)
    # loss = loss + tf.random_normal(tf.shape(loss), stddev=sigma)
    k = 1 / FLAGS.nb_teachers
    loss = loss + k*tf.constant(np.random.laplace(loc=0.0, scale=float(FLAGS.lap_scale)))
    
    tf.add_to_collection('losses', loss) # TODO noise too large, loss will be negative
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss_fun(logits, labels):
    '''
    compute loss fun without weight decay
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean

def ks_loss_fun(teacher_preds, student_pred):

    student_pred = logit2prob(student_pred)

    teacher_preds_dist = tf.distributions.Categorical(teacher_preds)
    student_pred_dist = tf.distributions.Categorical(student_pred)

    return tf.reduce_mean(tf.distributions.kl_divergence(teacher_preds_dist, student_pred_dist))

# def train_op_fun(total_loss, global_step):
#     '''

#     '''
def train(images, teacher_preds, labels, ckpt_path, dropout=False):
    '''
    This function contains the loop that actually trains the student model
    '''
    assert len(images) == len(labels)
    assert len(images) == len(teacher_preds)
    assert images.dtype == np.float32
    # assert labels.dtype == np.int32
    teacher_preds_shape = teacher_preds.shape

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        train_data_note = deep_cnn._input_placeholder()

        train_labels_shape = (FLAGS.batch_size,)
        train_labels_node = tf.placeholder(tf.int32, shape=train_labels_shape)

        teacher_preds_shape = (FLAGS.batch_size, FLAGS.nb_teachers, teacher_preds_shape[-1])
        teacher_preds_node = tf.placeholder(tf.float32, shape=teacher_preds_shape)
        print("Done Initializing Training Placeholders")

        if FLAGS.deeper:
            logits = deep_cnn.inference_deeper(train_data_note, dropout=dropout)
        else:
            logits = deep_cnn.inference(train_data_note, dropout=dropout)

        # teacher_preds = tf.constant(teacher_preds)
        ground_truth_loss = loss_fun(logits, train_labels_node)
        kl_loss = ks_loss_fun(teacher_preds_node, logits)
        loss = JS_loss_fun_noise(teacher_preds_node, logits)
        tm, sm = JS_part_fun(teacher_preds_node, logits)
        train_op = deep_cnn.train_op_fun(loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        print("Graph constructed and saver created")

        tf.summary.scalar('teacher_M', tm)
        tf.summary.scalar('student_M', sm)
        tf.summary.scalar('techear_student_kl', kl_loss)
        tf.summary.scalar('ground_truth_loss', ground_truth_loss)
        # summary_dir = FLAGS.train_dir + '/log_dir/run-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
        summary_dir = FLAGS.train_dir + '/log_dir/run-{}-{}-{}-{}-{}-{}' \
                    .format(FLAGS.lap_scale, FLAGS.tm_coef, FLAGS.sm_coef, FLAGS.heat, FLAGS.max_steps, FLAGS.learning_rate)
        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        data_length = len(images)
        nb_batches = math.ceil(data_length / FLAGS.batch_size)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            batch_nb = step % nb_batches

            start, end = utils.batch_indices(batch_nb, data_length, FLAGS.batch_size)

            feed_dict = {train_data_note: images[start:end],
                                teacher_preds_node: teacher_preds[start:end],
                                train_labels_node: labels[start:end]}
            
            _, loss_value, gt_loss_value, kl_loss_value, summary = sess.run([train_op, loss,  
                                                                                     ground_truth_loss, 
                                                                                     kl_loss, merged], feed_dict=feed_dict)
            duration = time.time() - start_time
            train_writer.add_summary(summary, step)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size 
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f, gt_loss = %.2f, kl_loss = %.2f, (%.1f examples/sec; %.3f sec/batch)')

                print(format_str % (datetime.now(), step, loss_value, gt_loss_value, kl_loss_value, examples_per_sec, sec_per_batch))

            if step % 1000 == 0 or (step+1) == FLAGS.max_steps:
                saver.save(sess, ckpt_path, global_step=step)
    return True

def numpy_softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1, keepdims=True)
    # s = s[:, np.newaxis]
    e_x = np.exp(z-s)
    div = np.sum(e_x, axis=1, keepdims=True)
    # div = div[:, np.newaxis]
    return e_x / div

def ensemble_preds(dataset, nb_teachers, stdnt_data, heat):
    
    result_shape = (nb_teachers, len(stdnt_data), FLAGS.nb_labels)
    
    result = np.zeros(result_shape, dtype=np.float32)

    for teacher_id in xrange(nb_teachers):
        if FLAGS.deeper:
            ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt-' + str(FLAGS.teachers_max_steps - 1) #NOLINT(long-line)
        else:
            ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt-' + str(FLAGS.teachers_max_steps - 1)  # NOLINT(long-line)
    
        logits = deep_cnn.softmax_preds(stdnt_data, ckpt_path, return_logits=True)
        probs = numpy_softmax(logits / heat)
        result[teacher_id] = probs

        print("Computed Teacher " + str(teacher_id) + " softmax predictions")

    return result

def teacher_aggregation_acc(teacher_preds, labels):
    '''
    teacher_preds: shape=(stdnt_share, nb_teachers, nb_labels)
    labels: shape = (stdnt_share,)
    '''
    teacher_preds_labels = np.argmax(teacher_preds, axis=2)
    aggregate_labels = np.zeros((teacher_preds.shape[0]), dtype=np.int32)
    for i in range(teacher_preds.shape[0]):
        bin_ = np.bincount(teacher_preds_labels[i, :], minlength=FLAGS.nb_labels)
        aggregate_labels[i] = np.argmax(bin_)
    
    acc = metrics.accuracy(aggregate_labels, labels)
    return acc


def train_student_JS(dataset, nb_teachers):

    assert input.create_dir_if_needed(FLAGS.train_dir)

    stdnt_dataset = prepare_student_data(dataset, nb_teachers)

    stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset

    teacher_preds_filepath = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_' + str(FLAGS.heat) + '_teacher_preds.npy'
    
    if os.path.exists(teacher_preds_filepath):
        teacher_preds = np.load(teacher_preds_filepath)
    else:
        teacher_preds = ensemble_preds(dataset, nb_teachers, stdnt_data, FLAGS.heat)
        teacher_preds = np.transpose(teacher_preds, (1, 0, 2))
        np.save(teacher_preds_filepath, teacher_preds)
    if FLAGS.deeper:
        ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student_deeper.ckpt'
    else:
        ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.ckpt'

    assert train(stdnt_data, teacher_preds,  stdnt_labels, ckpt_path)

    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

    teacher_precision = teacher_aggregation_acc(teacher_preds, stdnt_labels)
    print("Precision of teacher aggregation vote: " + str(teacher_precision))

    student_train_pred = deep_cnn.softmax_preds(stdnt_data, ckpt_path_final)
    student_train_precision = metrics.accuracy(student_train_pred, stdnt_labels)
    print("Precision of student on training data: " + str(student_train_precision))
    
    student_pred = deep_cnn.softmax_preds(stdnt_test_data, ckpt_path_final)

    precision = metrics.accuracy(student_pred, stdnt_test_labels)
    print("Precision of student after training: " + str(precision))

    return True

def main(argv=None):
    
    assert train_student_JS(FLAGS.dataset, FLAGS.nb_teachers)

if __name__ == '__main__':
    tf.app.run()