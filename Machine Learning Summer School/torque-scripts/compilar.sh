
#PBS -N compilar
DIR=~/torque-scripts

#alias
source ~/.bashrc

alias

echo 'compilando'
date

cd $DIR
mpicc.openmpi cpi.c

echo 'fin compilacion'
date

