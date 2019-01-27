#!/bin/bash
qsub -I -V -q $1 -l nodes=$2 -l walltime=$3

