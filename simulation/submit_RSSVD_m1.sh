#!/bin/bash

#---------------------------------------------------------------------------------
# Combined submission script for RSSVD with model_id=1 and exp_id=0,1,2,3
#---------------------------------------------------------------------------------

echo "Submitting RSSVD jobs for model_id=1 and exp_id in {0,1,2,3}..."

for EXP_ID in 0 1
do
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=rssvd_m1e${EXP_ID}
#SBATCH --array=0-99

echo "Job ID: \$SLURM_JOB_ID"
echo "Job User: \$SLURM_JOB_USER"
echo "Running RSSVD model_id=1, exp_id=${EXP_ID}, rd_seed_id=\${SLURM_ARRAY_TASK_ID}"

# Load R module
module load R/4.3

mkdir -p logs

MODEL_ID=1
SEED_ID=\${SLURM_ARRAY_TASK_ID}

# Run RSSVD experiment
Rscript run_RSSVD.R \$MODEL_ID ${EXP_ID} \$SEED_ID > logs/out_rssvd_m\${MODEL_ID}_e${EXP_ID}_s\${SEED_ID}.txt 2> logs/err_rssvd_m\${MODEL_ID}_e${EXP_ID}_s\${SEED_ID}.txt
EOF
done

echo "All RSSVD jobs for model_id=1 submitted."
