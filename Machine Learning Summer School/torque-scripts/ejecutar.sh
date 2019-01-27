#PBS -N ejecutar
DIR=~/torque-scripts

#alias
source ~/.bashrc

cd $DIR
hostname --fqdn
mpiexec.openmpi --hostfile $PBS_NODEFILE -np 10 ./a.out
