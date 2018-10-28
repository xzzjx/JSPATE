#coding: utf-8

import os
import sys
n_teachers = 100
for i in range(n_teachers):
    os.system('python train_teachers.py --nb_teachers={0} --teacher_id={1} --dataset=svhn \
    --datadir="/media/junxiang/Elements/svhn_dir" --train_dir="/media/junxiang/Elements/svhn_dir/train_dir" \
    '.format(n_teachers, i))