# coding: utf-8

from __future__ import division, print_function, unicode_literals
import os

def main():
    for max_steps in range(3000, 10001, 1000):
        for heat in range(1, 10):
            for tm_coef in range(10):
                # tm_coef: 0.1 -> 0.9
                tm_coef = tm_coef / 10
                sm_coef = 1 - tm_coef
                
                os.system("python train_student_JS.py --nb_teachers={0}  \
                                --dataset=svhn --stdnt_share={1} \
                                --data_dir='/media/junxiang/Elements/svhn_dir' \
                                --train_dir='/media/junxiang/Elements/svhn_dir/train_dir' \
                                --teachers_dir='/media/junxiang/Elements/svhn_dir/train_dir' \
                                --lap_scale={2} --heat={3} --max_steps={4}\
                                --tm_coef={5} --sm_coef={6}".format(100, 10000, 1, heat, max_steps, tm_coef, sm_coef))

if __name__ == '__main__':
    main()