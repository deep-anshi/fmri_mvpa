#!/bin/bash -l
#PBS -q default
#PBS -l nodes=1:ppn=4
#PBS -l walltime=00:10:00
#PBS -m bea
#PBS -M heejung.jung.gr@dartmouth.edu

SUBJ=${1}
TRIAL=${2}

echo ${SUBJ} ${TRIAL}

sleep 1

