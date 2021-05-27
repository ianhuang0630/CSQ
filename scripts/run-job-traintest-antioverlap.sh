#!/bin/bash
#SBATCH --partition=orion --qos=normal
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:titanxp:1

#SBATCH --job-name="train_sq"
#SBATCH --output=/orion/u/ianhuang/train_sq_chair_traintest_antioverlap_v0.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

srun bash run_traintest_antioverlap.sh

# done
echo "Done"
