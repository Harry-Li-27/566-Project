# 566-Project

## Usage

```bash
# transfer files, src to dst
scp -r <pathToFolder> <CARC>

# configure env, follow env_config.sh

# run model training; can modify hyperparams inside
sbatch run.sh

# env debug; alloc resources with options(#SBATCH) from run.sh
salloc

# conda setup
conda activate -n <name> python=3.8
conda deactivate
conda remove -m <name> --all

conda install <pkg>
conda uninstall <pkg>

pip install <pkg>
pip uninstall <pkg>
```

## Dataset Download - CUB

```
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
```
set the dataset path to 