#!/bin/bash

#PBS -S /bin/bash
#PBS -o job.out
#PBS -e job.err
#PBS -l nodes=1:ppn=24
#PBS -q debug
#PBS -m e

ulimit -s unlimited
csh
cd $PBS_O_WORKDIR
mpirun -hostfile $PBS_NODEFILE vasp-544-n > LOG
