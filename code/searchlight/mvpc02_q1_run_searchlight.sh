#!/bin/bash -l
#PBS -q default
#PBS -l nodes=1:ppn=5
#PBS -l walltime=02:10:00
#PBS -m bea
#PBS -M heejung.jung.gr@dartmouth.edu

SUBJ=${1}
HEMI=${2}
module load python/2.7-Anaconda
CODE_DIR=/dartfs-hpc/scratch/psyc164/groupXHD/code/mvpc
echo ${SUBJ} ${HEMI}
python ${CODE_DIR}/searchlight_ques-01_classify-20.py ${SUBJ} ${HEMI}

sleep 1

