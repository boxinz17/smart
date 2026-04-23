#!/bin/bash

#---------------------------------------------------------------------------------
# Submission script for SMARTCV with model_id=0 and exp_id=0 (exp1), 3 (exp4)
#---------------------------------------------------------------------------------

echo "Submitting SMARTCV jobs for model_id=0 and exp_id in {0,3}..."

for EXP_ID in 0 3
do
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=smartcv_m0e${EXP_ID}
#SBATCH --array=0-99

echo "Job ID: \$SLURM_JOB_ID"
echo "Job User: \$SLURM_JOB_USER"
echo "Running SMARTCV model_id=0, exp_id=${EXP_ID}, rd_seed_id=\${SLURM_ARRAY_TASK_ID}"

# Load Python module and activate virtual environment
module load python/3.12
source "${VENV:-$HOME/SMART_experiments}/bin/activate"

mkdir -p logs

MODEL_ID=0
SEED_ID=\${SLURM_ARRAY_TASK_ID}

# Run SMARTCV experiment
python run_SMARTCV.py \$MODEL_ID ${EXP_ID} \$SEED_ID > logs/out_smartcv_m\${MODEL_ID}_e${EXP_ID}_s\${SEED_ID}.txt 2> logs/err_smartcv_m\${MODEL_ID}_e${EXP_ID}_s\${SEED_ID}.txt
EOF
done

echo "All SMARTCV jobs submitted."