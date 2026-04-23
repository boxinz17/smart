#!/bin/bash

echo "Submitting all source estimation jobs..."

for METHOD in RRR SRRR RSSVD SOFAR
do
  echo "Submitting job_source_${METHOD}.sh"
  sbatch job_source_${METHOD}.sh
done

echo "All jobs submitted."