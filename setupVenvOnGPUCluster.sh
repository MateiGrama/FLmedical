#!/bin/bash

# Move to not-persistent workespace (project repo)
cd /vol/bitbucket/mgg17/diss

# Copy conda setup file for environment
cp /homes/mgg17/conda.sh conda.sh

# Running the setup script
bash conda.sh -b -p venv

rm conda.sh

source venv/bin/activate

# Installing required libraries
pip install torch
pip install torch.vision
pip install image
pip install scipy
pip install sklearn
pip install matplotlib
