#!/bin/bash
#PBS -q default
#PBS -l nodes=1:ppn=10
#PBS -l walltime=12:00:00
#PBS -m bea
#PBS -M heejung.jung.gr@dartmouth.edu


module load afni
module load python/2.7-Anaconda
subjects=( "rid000001" "rid000012" "rid000017" "rid000024" "rid000027" "rid000031" \
"rid000032" "rid000033" "rid000034" "rid000036" "rid000037" "rid000041")
trials=(  "tax" "beh")

for SUBJ in ${subjects[*]}; do
for TRIAL_SELECT_KEY_VAL in ${trials[*]}; do

CODE_DIR=/dartfs-hpc/scratch/psyc164/groupXHD/code
python ${CODE_DIR}/category_glm_runwise.py ${SUBJ} ${TRIAL_SELECT_KEY_VAL}

sleep 1

done
done
