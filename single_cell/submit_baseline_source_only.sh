#!/bin/bash

# ---------------------------------------------------------------------
# Submission script for run_baseline_source_only.py (All source types)
# ---------------------------------------------------------------------

echo "Submitting source-only baseline jobs for all matrix types..."

for MATRIX_ID in {0..6}
do
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=src_only_m${MATRIX_ID}
#SBATCH --array=0-99

echo "Job ID: \$SLURM_JOB_ID"
echo "User: \$SLURM_JOB_USER"
echo "Running run_baseline_source_only.py with matrix_id=${MATRIX_ID}, rd_id=\${SLURM_ARRAY_TASK_ID}"

# Load Python and activate environment
module load python/3.12
source "${VENV:-$HOME/SMART_experiments}/bin/activate"

mkdir -p logs

# Run baseline source-only job
python run_baseline_source_only.py ${MATRIX_ID} \${SLURM_ARRAY_TASK_ID} > logs/out_source_only_m${MATRIX_ID}_s\${SLURM_ARRAY_TASK_ID}.txt 2> logs/err_source_only_m${MATRIX_ID}_s\${SLURM_ARRAY_TASK_ID}.txt
EOF
done

echo "All source-only baseline jobs submitted."
