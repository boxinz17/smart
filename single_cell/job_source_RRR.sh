#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=source_RRR

echo "Job ID: $SLURM_JOB_ID"
echo "Running source estimation for RRR"

module load R/4.3
module load python/3.12
mkdir -p logs
Rscript run_source_estimation_RRR.R > logs/out_rrr.txt 2> logs/err_rrr.txt
