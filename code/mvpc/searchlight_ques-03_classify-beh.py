#!/usr/bin/env python
"""
searchlight_ques-03_classify-beh.py

Classify taxa (1 out of 5),
training on videos with 3 behaviors
and testing on videos with the left-out behavior.

step 1. This code first creates a dataset based on the glm surface outputs
step 2. 3. It creates cross validation classifiers and searchlight objects.

chunks: taxa
targets: beh
"""
import sys
import os
import numpy as np
import mvpa2
import pandas as pd
import generate_dataset
# import nilearn
# import nipy
import mvpa2.suite as mv
# import nibabel
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
task_name = sys.argv[3]
radii = 10.0

# 1. create pymvpa dataset  ____________________________________________________________
ds_q3 = generate_dataset.create_dataset(sub_name, main_dir, task_name, hemisphere)
ds_q3.sa['chunks'] = ds_q3.sa['tax']
ds_q3.sa['targets'] = ds_q3.sa['beh']
#del ds_q3.sa['intents']

mv.zscore(ds_q3, chunks_attr = 'chunks')

n_medial = {'lh': 3486, 'rh': 3491}
medial_wall = np.where(np.sum(ds_q3.samples == 0, axis=0) == 100)[0].tolist()
cortical_vertices = np.where(np.sum(ds_q3.samples == 0, axis=0) < 100)[0].tolist()
assert len(medial_wall) == n_medial[hemisphere]
n_vertices = ds_q3.fa.node_indices.shape[0]
assert len(medial_wall) + len(cortical_vertices) == n_vertices

# 2. cross validation __________________________________________________________________
# setting up classifier
clf = mv.LinearCSVMC()
cv = mv.CrossValidation(clf, mv.NFoldPartitioner())
cv_within = cv(ds_q3)
cv_within
np.mean(cv_within)
# why is the mean lower?

# 3. searchlight _______________________________________________________________________
fsaverage_gii = os.path.join(main_dir , 'fs_templates', hemisphere+ '.pial.gii')
surf = mv.surf.read(fsaverage_gii)
# note: surf.vertices.shape (81920, 3) and surf.faces.shape (40962, 3) surface = surf,
qe = mv.SurfaceQueryEngine(surf, radius = radii,distance_metric='dijkstra')
sl = mv.Searchlight(cv, queryengine=qe,roi_ids=cortical_vertices)
sl_q3 = sl(ds_q3)

# 3-1. reconstruct medial wall _________________________________________________________
assert sl_q3.shape[1] == len(cortical_vertices)
sl_final = np.zeros((1, n_vertices))
np.put(sl_final, cortical_vertices, np.mean(sl_q3, axis=0))
assert sl_final.shape == (1, n_vertices)

# 4. save output _______________________________________________________________________
if not os.path.exists(os.path.join(main_dir, 'analysis', 'searchlight', sub_name)):
    os.makedirs(os.path.join(main_dir, 'analysis', 'searchlight', sub_name))

# save as NIML dataset
niml_q3_filename = os.path.join(main_dir, 'analysis', 'searchlight', sub_name,
  sub_name+'_ques-03_task-'+task_name+'_'+hemisphere+ '_searchlight_radii-' +str(radii)+'.niml.dset')
mv.niml.write(niml_q3_filename, sl_final)

# save as GIFTI sub-rid000001_ques-01_task-beh_lh_searchlight_radii-10
searchlight_q3_filename = os.path.join(main_dir, 'analysis', 'searchlight', sub_name,
sub_name+'_ques-03_task-'+task_name+'_'+hemisphere+ '_searchlight_radii-' +str(radii)+'.gii')
nimg = mv.map2gifti(sl_final, searchlight_q3_filename, encoding='GIFTI_ENCODING_B64GZ', surface=surf)


# **** helpful resources _______________________________________________________________
# generating dataset:
# http://www.pymvpa.org/tutorial_mappers.html
# saving output:
# http://www.pymvpa.org/examples/searchlight_surf.html
#
