#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mgg17
#SBATCH --output=FedLrn
source /vol/bitbucket/mgg17/diss/venv/bin/activate

source /vol/cuda/10.0.130/setup.sh

TERM=vt100

/usr/bin/nvidia-smi

uptime

nohup python -u main.py > out/experimentAFA.log 2>&1
#echo $! > out/lastExperimentPID.txt

