#!/bin/bash 
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:8
#SBATCH -J speedtest-dino-alternating-training-seed0
#SBATCH -o log.txt 
 
echo "Workingdir: $PWD" 
echo "Started at $(date)" 
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"
 
source ~/.profile 
conda activate torch 

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
code/speedtest.py --config 0

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
code/speedtest.py --config 1

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
code/speedtest.py --config 2

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
code/speedtest.py --config 3

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
code/speedtest.py --config 4
