#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=source_SOFAR

echo "Job ID: $SLURM_JOB_ID"
echo "Running source estimation for SOFAR"

module load R/4.3
module load python/3.12
mkdir -p logs
Rscript run_source_estimation_SOFAR.R > logs/out_sofar.txt 2> logs/err_sofar.txt