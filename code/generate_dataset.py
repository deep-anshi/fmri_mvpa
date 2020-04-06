#!/usr/bin/env python
# encoding: utf-8

"""
generate_dataset.py

To be used as a function within searchlight_xhd
"""
import os
import numpy as np
import mvpa2
import pandas as pd
import nilearn
import nipy
import mvpa2.suite as mv
import nibabel
from numpy.testing.decorators import skipif

# main_dir = '/Users/h/Documents/projects_local/cluster_projects'
# main_dir = '/dartfs-hpc/scratch/psyc164/groupXHD'
# sub_name = 'sub-rid000001'
# task_list = ['beh', 'tax']
# data_set = []

def create_dataset(sub_name, main_dir, task_list, hemi):
    data_set = []
    for task_name in task_list:
        for run_num in range(1,6):
            ds=[]
            gifti_fname = os.path.join(main_dir,'analysis', sub_name,'func', sub_name + '_task-' + task_name + '_run-' + str(run_num) +'_rw-glm.' +hemi+'.coefs.gii')
            ds = mv.gifti_dataset(gifti_fname)

            # order in sub-rid000001_task-beh_run-5_rw-glm.lh.X.xmat.1D
            ds.sa['beh_tax'] = ["bird_eating", "bird_fighting", "bird_running", "bird_swimming",
                               "insect_eating", "insect_fighting", "insect_running", "insect_swimming",
                               "primate_eating","primate_fighting","primate_running","primate_swimming",
                               "reptile_eating","reptile_fighting", "reptile_running", "reptile_swimming",
                               "ungulate_eating","ungulate_fighting","ungulate_running","ungulate_swimming"]
            ds.sa['beh'] = np.tile(['eating','fighting','running','swimming'],5)
            ds.sa['tax'] = np.repeat(['bird','insect','primate', 'reptile', 'ungulate'],4)
            ds.fa['node_indices'] = range(0,ds.shape[1]) # 0 ~ 400000
            data_set.append(ds)

    # http://www.pymvpa.org/tutorial_mappers.html
    within_ds = mv.vstack(data_set)
    return within_ds
