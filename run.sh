#!/bin/bash

#SBATCH --job-name=abme
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G


echo start time is "$(date)"
echo Slurm job ID is "${SLURM_JOBID}"

#module add languages/anaconda3/2020.02-tflow-2.2.0
source /mnt/storage/software/languages/anaconda/Anaconda3.8.5/etc/profile.d/conda.sh

conda activate stn
cd "${SLURM_SUBMIT_DIR}"

python -u evaluate.py --data_dir "/user/home/wg19671/stmfnet/stmfnet_data"

echo end time is "$(date)"
hostname
