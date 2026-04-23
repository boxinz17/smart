#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=target_only
#SBATCH --array=0-99

echo "Job ID: $SLURM_JOB_ID"
echo "Task ID (rd_id): $SLURM_ARRAY_TASK_ID"
echo "Running target-only baseline estimation for rd_id=$SLURM_ARRAY_TASK_ID"

# Load necessary modules
module load R/4.3
module load python/3.12

mkdir -p logs
Rscript run_baseline_target_only.R $SLURM_ARRAY_TASK_ID > logs/out_target_only_${SLURM_ARRAY_TASK_ID}.txt 2> logs/err_target_only_${SLURM_ARRAY_TASK_ID}.txt
