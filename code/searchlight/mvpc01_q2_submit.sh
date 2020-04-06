#!/bin/bash
#PBS -q default
#PBS -l nodes=1:ppn=10
#PBS -l walltime=12:00:00
#PBS -m e
#PBS -M heejung.jung.gr@dartmouth.edu

module load afni
module load python/2.7-Anaconda

subjects=("sub-rid000001" "sub-rid000012" "sub-rid000017" "sub-rid000024" "sub-rid000027" \
"sub-rid000031" "sub-rid000032" "sub-rid000033" "sub-rid000034" "sub-rid000036" \
"sub-rid000037" "sub-rid000041")
hemi=("lh" "rh")
CODE_DIR=/dartfs-hpc/scratch/psyc164/groupXHD/code
for SUBJ in ${subjects[*]}; do
  for HEMI in ${hemi[*]}; do
    ${CODE_DIR}/mvpc/mvpc02_q2_run_searchlight.sh ${SUBJ} ${HEMI}
  done
done

