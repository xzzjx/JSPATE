#coding: utf-8

import os
import sys
n_teachers = 1
for i in range(n_teachers):
    os.system('python train_teachers.py --nb_teachers={0} --teacher_id={1} --dataset=mnist \
    --data_dir="/media/junxiang/Elements/mnist_dir" --train_dir="/media/junxiang/Elements/mnist_dir/train_dir_1" \
    '.format(n_teachers, i))