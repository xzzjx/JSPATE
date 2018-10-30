#coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
from differential_privacy.multiple_teachers import aggregation

def noisy_max(teacher_preds, noise_scale, return_clean_votes=False):
    # TODO
    labels = aggregation.labels_from_probs(teacher_preds)
    labels_shape = np.shape(labels)
    labels = labels.reshape((labels_shape[0], labels_shapep1))