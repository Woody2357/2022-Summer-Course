#!/bin/bash
#PBS -P volta_pilot
#PBS -j oe
#PBS -N rnn7071
#PBS -q volta_gpu
#PBS -l select=1:ncpus=5:mem=50gb:ngpus=1:mpiprocs=1
#PBS -l walltime=48:00:00
cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);
image="/app1/common/singularity-img/3.0.0/pytorch_1.0.0_nvcr_19.01_py3.simg"
singularity exec $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID
python RNNs_eval_samenoise.py --seq_length_test 20 --checkpoint RNN_train_last_20_same_noise
EOF