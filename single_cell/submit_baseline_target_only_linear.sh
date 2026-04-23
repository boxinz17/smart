#!/bin/bash

# ------------------------------------------------------------------------
# Submission script for run_baseline_target_only_linear.py (Ridge/OLS/Lasso)
# ------------------------------------------------------------------------

echo "Submitting target-only linear baseline jobs (ridge, ols, lasso)..."

sbatch <<EOF
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=target_only_linear
#SBATCH --array=0-99

echo "Job ID: \$SLURM_JOB_ID"
echo "User: \$SLURM_JOB_USER"
echo "Running run_baseline_target_only_linear.py with rd_id=\${SLURM_ARRAY_TASK_ID}"

# Load Python and activate environment
module load python/3.12
source "${VENV:-$HOME/SMART_experiments}/bin/activate"

mkdir -p logs

# Run target-only linear model baseline job
python run_baseline_target_only_linear.py \${SLURM_ARRAY_TASK_ID} > logs/out_target_only_linear_s\${SLURM_ARRAY_TASK_ID}.txt 2> logs/err_target_only_linear_s\${SLURM_ARRAY_TASK_ID}.txt
EOF

echo "All target-only linear baseline jobs submitted."
