#!/bin/bash

#---------------------------------------------------------------------------------
# Combined submission script for SMART with model_id=2 and exp_id=0,1,2,3
#---------------------------------------------------------------------------------

echo "Submitting SMART jobs for model_id=2 and exp_id in {0,1,2,3}..."

for EXP_ID in 0 1 2 3
do
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=smart_m2e${EXP_ID}
#SBATCH --array=0-99

echo "Job ID: \$SLURM_JOB_ID"
echo "Job User: \$SLURM_JOB_USER"
echo "Running SMART model_id=2, exp_id=${EXP_ID}, rd_seed_id=\${SLURM_ARRAY_TASK_ID}"

# Load Python module and activate SMART virtual environment
module load python/3.12
source "${VENV:-$HOME/SMART_experiments}/bin/activate"

mkdir -p logs

MODEL_ID=2
SEED_ID=\${SLURM_ARRAY_TASK_ID}

# Run SMART experiment
python run_SMART.py \$MODEL_ID ${EXP_ID} \$SEED_ID > logs/out_smart_m\${MODEL_ID}_e${EXP_ID}_s\${SEED_ID}.txt 2> logs/err_smart_m\${MODEL_ID}_e${EXP_ID}_s\${SEED_ID}.txt
EOF
done

echo "All jobs submitted."
