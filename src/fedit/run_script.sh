#!/bin/bash

### Job Name
#SBATCH --job-name=exp

### Log Output
#SBATCH --output=output-generate-1.txt

### Log Errors
#SBATCH --error=error-generate-1.txt

### Partition
#SBATCH --partition=permanent

### Number of Nodes:
#SBATCH --nodes=1

### Number of Tasks (one for each GPU desired):
#SBATCH --ntasks=2

### Processors Per Task
#SBATCH --cpus-per-task=24

### Number of GPUs  (which GPU)
#SBATCH --gres=gpu:2

### Memory
#SBATCH --mem=64G

### Compute Time
#SBATCH --time=24:00:00

### Load Modules as Needed
eval "$(conda shell.bash hook)"
conda activate nerfstudio

### Run Job Script
python3 final_dataset_generation.py                       ## Pre-training
