#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --account=jessetho_1016

module purge 
module load gcc/8.3.0 
module load cuda/11.1-1

python3 -m torch.distributed.launch \
        --nproc_per_node=2 \
        moco-v3-main/main_moco.py \
        --moco-m-cos --crop-min=.2 \
        --batch-size 32 \
        --epochs 60 \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        -j 4 \
        ../imagenet-mini

#run local
python -m torch.distributed.launch --nproc_per_node=1 moco-v3-main/main_moco.py --moco-m-cos --crop-min=.2 --batch-size 16 --epochs 60 --multiprocessing-distributed --world-size 1 --rank 0 -j 2 ../imagenet-mini --dist-url "file:///sharefile"