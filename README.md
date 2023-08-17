# MinSimAugment

## Helix Commands
### Setup
```
module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh
conda create -n minsim python=3.10
conda activate minsim
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge tensorboard
pip install omegaconf
```
### Interactive jobs
```
Interactive jobs must NOT run on the logins nodes, however resources for interactive jobs can be requested using srun.
$ salloc --partition=single --ntasks=2 --time=2:00:00 --mem=16G --gres=gpu:1
# partition: https://wiki.bwhpc.de/e/Helix/Slurm#:~:text=the%20job%20script.-,3.1,-Partitions
# ntasks: number of cpus
# mem: memory in GB
# gres: number of gpus

It is also possible to attach an interactive shell to a running job with command:
$ srun --jobid=<jobid> --overlap --pty /bin/bash
```
### Non-interactive jobs
```
#!/bin/bash
#SBATCH -p single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:4
# SBATCH --mem=236G
#SBATCH -J JOBNAME
#SBATCH -o path to stdout file
#SBATCH -e path to stderr file
# SBATCH --array 0-X%1
```
