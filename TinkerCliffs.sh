#!/bin/bash

#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128
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
export OMP_NUM_THREADS=1

python A3C_train_agent_optuna_5.py


echo "Scrpt End"