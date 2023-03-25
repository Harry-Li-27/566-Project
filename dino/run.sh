#!/bin.bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --account=jessetho_1016

module purge
module load gcc/9.2.0
module load cuda/11.0.2
module load nvhpc/22.11
 
python3 -m torch.distributed.launch \
        --nproc_per_node=2 \
        dino-main/main_dino.py \
        --arch resnet50 \
        --optimizer adamw \
        --batch_size_per_gpu 32 \
        --epochs 60 \
        --lr 0.01 \
        --weight_decay 1e-4 \
        --weight_decay_end 1e-4 \
        --global_crops_scale 0.14 1 \
        --local_crops_scale 0.05 0.14 \
        --data_path ../imagenet-mini/train \
        --output_dir ./output/

        python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir