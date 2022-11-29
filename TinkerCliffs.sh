#!/bin/bash

#SBATCH --ntasks=128
#SBATCH -t 00-00:10:00
#SBATCH -p dev_q
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

echo "set OMP_NUM_THREADS=1"
OMP_NUM_THREADS=1
export OMP_NUM_THREADS

python A3C_train_agent_optuna_4.py


echo "Scrpt End"