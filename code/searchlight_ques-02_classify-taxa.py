#!/usr/bin/env python
"""
searchlight_ques-02_classify-taxa.py

Classify taxa (1 out of 5),
training on videos with 3 behaviors
and testing on videos with the left-out behavior.

step 1. This code first creates a dataset based on the glm surface outputs
step 2. 3. It creates cross validation classifiers and searchlight objects.

chunks: beh
targets: taxa
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

__author__ = "Heejung Jung, Xiaochun Han, Deepanshi Shokeen"
__version__ = "1.0.1"
__email__ = "heejung.jung@colorado.edu"
__status__ = "Production"




# 0. parameters ____________________________________________________________
# main_dir = '/Users/h/Documents/projects_local/cluster_projects'
main_dir = '/dartfs-hpc/scratch/psyc164/groupXHD'
sub_name = sys.argv[1]
hemisphere = sys.argv[2]
task_list = ['beh', 'tax']
radii = 10.0

# 1. create pymvpa dataset  ____________________________________________________________
ds_q2 = generate_dataset.create_dataset(sub_name, main_dir, task_list, hemisphere)
ds_q2.sa['chunks'] = ds_q2.sa['beh']
ds_q2.sa['targets'] = ds_q2.sa['tax']
del ds_q2.sa['intents']
mv.zscore(ds_q2, chunks_attr = 'chunks')

# 2. cross validation __________________________________________________________________
# setting up classifier
clf = mv.LinearCSVMC()
cv = mv.CrossValidation(clf, mv.NFoldPartitioner())
cv_within = cv(ds_q2)
cv_within
np.mean(cv_within)
# why is the mean lower?

# 3. searchlight _______________________________________________________________________
fsaverage_gii = os.path.join(main_dir , 'fs_templates', hemisphere+ '.pial.gii')
surf = mv.surf.read(fsaverage_gii)
# note: surf.vertices.shape (81920, 3) and surf.faces.shape (40962, 3) surface = surf,
qe = mv.SurfaceQueryEngine(surf, radius = radii,distance_metric='dijkstra')
sl = mv.Searchlight(cv, queryengine=qe)
sl_q2 = sl(ds_q2)

# 4. save output _______________________________________________________________________

# save as NIML dataset
niml_q1_filename = os.path.join(main_dir, 'analysis', 'searchlight', sub_name,
  sub_name+'_ques-02_'+hemisphere+ '_searchlight_radii-' +str(radii)+'.niml.dset')
niml.write(niml_q1_filename, sl_q2)

# save as GIFTI sub-rid000001_ques-01_lh_searchlight_radii-10
searchlight_q1_filename = os.path.join(main_dir, 'analysis', 'searchlight', sub_name,
sub_name+'_ques-02_'+hemisphere+ '_searchlight_radii-' +str(radii)+'.gii')
nimg = mv.map2gifti(sl_q2, filename, encoding='GIFTI_ENCODING_B64GZ', surface=surf)


# **** helpful resources _______________________________________________________________
# generating dataset:
# http://www.pymvpa.org/tutorial_mappers.html
# saving output:
# http://www.pymvpa.org/examples/searchlight_surf.html
#
