#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --account=jessetho_1016

module purge 
module load gcc/8.3.0 
module load cuda/11.1-1

python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        moco-v3-main/main_lincls.py \
        -a resnet50 \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --batch-size 32 \
        --epochs 150 \
        -j 4 \
        --pretrained output_imagenet100_base_augmix/checkpoint_0099.pth.tar \
        ../imagenet100

