#!/bin/bash
#PBS -q default
#PBS -l nodes=1:ppn=4
#PBS -l walltime=00:10:00
#PBS -m bea
#PBS -M heejung.jung.gr@dartmouth.edu

SUBJ=${1}
HEMI=${2}
CODE_DIR=/dartfs-hpc/scratch/psyc164/groupXHD/code/mvpc
python ${CODE_DIR}/searchlight_ques-03_classify-beh.py ${SUBJ} ${HEMI}

sleep 1

