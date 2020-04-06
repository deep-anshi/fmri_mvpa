#!/bin/bash
#PBS -q default
#PBS -l nodes=1:ppn=4
#PBS -l walltime=00:10:00
#PBS -m 3
#PBS -M heejung.jung.gr@dartmouth.edu

SUBJ=${1}
HEMI=${2}
TASK=${3}
CODE_DIR=/dartfs-hpc/scratch/psyc164/groupXHD/code/mvpc
echo ${SUBJ} ${HEMI} ${TASK}
python ${CODE_DIR}/searchlight_ques-03_classify-beh.py ${SUBJ} ${HEMI} ${TASK}

sleep 1
