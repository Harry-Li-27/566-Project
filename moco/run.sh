#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --account=jessetho_1016

module purge 
module load gcc/8.3.0 
module load cuda/11.1-1

python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        moco-v3-main/main_moco.py \
        --moco-m-cos --crop-min=.02 \
        --batch-size 32 \
	    --lr 0.3 \
        --wd 1e-6 \
        --output_dir ./output_imagenet100_base_augmix/ \
        --epochs 100 \
	    --optimizer sgd \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        -j 4 \
        ../imagenet100

