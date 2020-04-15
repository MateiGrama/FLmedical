#!/bin/bash

source /home/mgg17/diss/venv/bin/activate
nohup python -u main.py > out/experiment.log 2>&1 &
echo $! > out/lastExperimentPID.txt

