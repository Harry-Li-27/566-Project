```
# login
ssh yihaozho@discovery.usc.edu

# host
hostname

# quota
myquota

# transfer files
scp <filename> yihaozho@hpc-transfer1.usc.edu:/home1/yihaozho

# load module
module list

module avail
module spider <module_name>

module load <module/name>/<version>
module unload <module_name>

module purge: unload all modules

# run job
# example run.sh
#!/bin.bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1 (2 each user)
#SBATCH --account=jessetho_1016

module purge
module load <>/<v>

python3 <program>

# run job
sbatch run.sh

# job status
squeue --me

# job info
sacct / seff

# simple debugging
salloc --time=2:00:00 --cpus-per-task=8 --mem=16GB
```
