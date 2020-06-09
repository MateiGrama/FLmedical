#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mgg17
#SBATCH --output=FedLrn
source /vol/bitbucket/mgg17/diss/venv/bin/activate

source /vol/cuda/10.0.130/setup.sh

TERM=vt100

/usr/bin/nvidia-smi

add-apt-repository ppa:openjdk-r/ppa
apt-get update -q
apt install -y openjdk-11-jdk

uptime

nohup python -u main.py > out/experimentAFAxis.log 2>&1
#nohup python -u main.py > out/experimentCOVIDx_resnet_5clients_2batch_10rounds_cpu_2ndAttempt.log 2>&1

#echo $! > out/lastExperimentPID.txt

