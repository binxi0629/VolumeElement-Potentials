#!/bin/bash

#PBS -S /bin/bash
#PBS -o myjob.out
#PBS -e myjob.err
#PBS -l walltime=00:01:00:00
#PBS -l nodes=1:ppn=24
#PBS -q	normal

ulimit -s unlimited
csh
cd $PBS_O_WORKDIR
mpirun -hostfile $PBS_NODEFILE vasp-535-s > LOG

rm WAVECAR CHG CHGCAR