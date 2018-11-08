#coding: utf-8

import os
import sys
n_teachers = 600
teacher_data_share = 0
max_steps = 1000
batch_size = min(128, 60000//n_teachers)
for i in range(n_teachers):
    os.system('python train_teachers.py --nb_teachers={0} --teacher_id={1} --dataset=mnist \
    --data_dir="/media/junxiang/Elements/mnist_dir" --train_dir="/media/junxiang/Elements/mnist_dir/train_dir_60000_600" \
    --max_steps={2} --teacher_data_share={3} --batch_size={4}'.format(n_teachers, i, max_steps, teacher_data_share, batch_size))
