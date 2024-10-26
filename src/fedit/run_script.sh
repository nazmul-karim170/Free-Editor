#!/bin/bash

### Job Name
#SBATCH --job-name=exp

### Log Output
#SBATCH --output=output-generate.txt

### Log Errors
#SBATCH --error=error-generate.txt

### Partition
#SBATCH --partition=defq

### Number of Nodes:
#SBATCH --nodes=1

### Number of Tasks (one for each GPU desired):
#SBATCH --ntasks=1

### Processors Per Task
#SBATCH --cpus-per-task=24

### Number of GPUs  (which GPU)
#SBATCH --gres=gpu:1

### Memory
#SBATCH --mem=64G

### Compute Time
#SBATCH --time=24:00:00

### Load Modules as Needed
export HTTPS_PROXY="http://StripDistrict:10132"
module load shared cuda11.8/toolkit/11.8.0
eval "$(conda shell.bash hook)"
conda activate nerfstudio

### Run Job Script
python3 final_dataset_generation.py                       ## Pre-training
# python3 train_classifier.py                             ## Fine-tuning on downstream task 