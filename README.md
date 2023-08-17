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
### Non-interactive jobs
