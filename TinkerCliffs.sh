#!/bin/bash

#SBATCH --ntasks=128
#SBATCH -t 03-00:00:00
#SBATCH -p normal_q
#SBATCH --account=zelin1
#SBATCH --export=NONE # this makes sure the compute environment is clean

echo "TinkerCliffs Start"

echo "Core Number:"
nproc --all

pwd
ls

echo "load Anaconda"
module load Anaconda3

echo "activate environment: dronePytorch"
source activate dronePytorch

python --version

python A3C_train_agent_optuna_3.py


echo "Scrpt End"