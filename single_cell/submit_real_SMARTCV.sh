#!/bin/bash

# ---------------------------------------------------------------
# Submission script for run_real_SMARTCV.py across all methods
# ---------------------------------------------------------------

echo "Submitting SMARTCV jobs on real data for all methods (0-6)..."

for METHOD_ID in {0..6}
do
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=real_smartcv_m${METHOD_ID}
#SBATCH --array=0-99

echo "Job ID: \$SLURM_JOB_ID"
echo "User: \$SLURM_JOB_USER"
echo "Running run_real_SMARTCV.py with method_id=${METHOD_ID}, rd_id=\${SLURM_ARRAY_TASK_ID}"

# Load Python and activate environment
module load python/3.12
source "${VENV:-$HOME/SMART_experiments}/bin/activate"

mkdir -p logs

# Run SMARTCV job
python run_real_SMARTCV.py ${METHOD_ID} \${SLURM_ARRAY_TASK_ID} > logs/out_real_smartcv_m${METHOD_ID}_s\${SLURM_ARRAY_TASK_ID}.txt 2> logs/err_real_smartcv_m${METHOD_ID}_s\${SLURM_ARRAY_TASK_ID}.txt
EOF
done

echo "All SMARTCV real-data jobs submitted."
