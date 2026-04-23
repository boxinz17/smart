#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --account=YOUR_ACCOUNT

# Load Python and activate environment
module load python/3.12
source "${VENV:-$HOME/SMART_experiments}/bin/activate"

mkdir -p logs

# Run script and redirect output
python prepare_data.py > logs/out_prepare_data.txt 2> logs/err_prepare_data.txt
