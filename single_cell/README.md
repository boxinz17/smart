# Real-data experiment (Section 6)

This folder reproduces the CITE-seq multi-task experiment in Section 6 of
*Zhao, Kolar, Lv — "SMART: A Spectral Transfer Approach to Multi-Task Learning",
arXiv preprint [arXiv:2604.20161](https://arxiv.org/abs/2604.20161) (2026).*
The data live in a dense `.h5ad` file (a few GB), so they are **not** checked into the
repo. All generated intermediates (`processed_data/`, `source_matrix/`), per-seed
results (`result/`), figures (`fig/`), and HPC logs (`logs/`) are ignored by git and
re-created by the scripts below.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ./smart
pip install -r requirements.txt
pip install scanpy anndata prettytable     # needed only by this folder
```

For the R baselines install `rrpack` and `reticulate` (see top-level `R_packages.txt`).
The R scripts call into NumPy through `reticulate`, so your R environment must be able
to see the Python environment above.

## 2. Raw data

The paper uses the NeurIPS 2021 Multimodal Single-Cell Integration challenge CITE-seq
dataset (GEO accession
[GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)).
Download
`GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad` (≈4.5 GB) and place it at

```
raw_data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad
```

The path is relative to this folder (`single_cell/raw_data/…`).

## 3. Preprocessing

`pre_processing/prepare_data.py` subsets the data to the NK (source) and ILC1 (target)
cell types, runs CLR normalisation on the ADT matrix, log-normalises and selects 3000
highly-variable genes on the GEX matrix, applies marginal-regression feature screening
to reduce to 150 features per task, and writes the four matrices plus intrinsic-rank
estimates. Run it once:

```bash
cd pre_processing
python prepare_data.py
cd ..
```

The outputs land in `processed_data/`:

```
processed_data/
    X_source.npy, Y_source.npy      # NK training data
    X_target.npy, Y_target.npy      # ILC1 training/evaluation data
    r_hat_source.csv, r_hat_target.csv   # rank estimates from RankSelectorRSC
```

## 4. Source matrices

SMART takes a source regression coefficient matrix `C0` as input. We estimate it seven
different ways on the large NK source. Run each once (order does not matter):

```bash
# Ridge / OLS / Lasso source estimators (Python)
python run_source_estimation_linear.py

# RRR / SRRR / RSSVD / SOFAR source estimators (R)
Rscript run_source_estimation_RRR.R
Rscript run_source_estimation_SRRR.R
Rscript run_source_estimation_RSSVD.R
Rscript run_source_estimation_SOFAR.R
```

The HPC helpers `job_source_<method>.sh` wrap each of the four R calls; they are
optional. Each writes `source_matrix/source_matrix_<method>.npy`.

## 5. Baselines

Two families of baselines are evaluated per random split:

Target-only estimators fit everything from the (small) ILC1 training set:

```bash
for rd_id in $(seq 0 99); do
    Rscript run_baseline_target_only.R       $rd_id
    python  run_baseline_target_only_linear.py $rd_id
done
```

Source-only estimators evaluate each source matrix on the ILC1 test set (no
target-side training):

```bash
for matrix_id in 0 1 2 3 4 5 6; do
    for rd_id in $(seq 0 99); do
        python run_baseline_source_only.py $matrix_id $rd_id
    done
done
```

Slurm equivalents are `submit_baseline_target_only.sh`,
`submit_baseline_target_only_linear.sh`, and `submit_baseline_source_only.sh`.

All Slurm helpers in this folder (including `submit_real_SMARTCV.sh`,
`submit_source_estimation.sh`, `job_source_*.sh`, and
`pre_processing/prepare_data.sh`) contain the same set of placeholders:

- `#SBATCH --account=YOUR_ACCOUNT` — replace with your Slurm account.
- `module load python/3.12` and `module load R/4.3` — adjust to your cluster's
  module names.
- `source "${VENV:-$HOME/SMART_experiments}/bin/activate"` — either place your
  Python virtual environment at `$HOME/SMART_experiments/` or
  `export VENV=/path/to/venv` before running `sbatch`.

## 6. SMART on ILC1

For every combination of source matrix (7 methods) and random split (100 seeds):

```bash
for method_id in 0 1 2 3 4 5 6; do
    for rd_id in $(seq 0 99); do
        python run_real_SMARTCV.py $method_id $rd_id
    done
done
```

Each invocation writes
`result/SMARTCV_realdata_<method>_rd_id=<k>.pkl`. The full sweep launches via
`submit_real_SMARTCV.sh`. `run_real_SMARTCV.py` uses the package default `gamma=2.0`,
which matches the paper's main real-data results.

## 7. Table and figure

Once all `.pkl` files are in `result/`, compile the summary:

```bash
python summarize_real_results_table.py    # fig/summary_statistics_table.txt
python visualize_real_results.py          # fig/smart_vs_baselines_boxplot.pdf
```

The text table is the one reported in Section 6; the boxplot is the corresponding
figure.

## 8. Random seeds

Train/test splits are determined by
[`random_seeds/realdata_seeds.csv`](random_seeds/realdata_seeds.csv) (100 seeds),
which is checked in. Every baseline and SMART script reads its seed for a given
`rd_id` directly from this CSV.

## 9. Expected runtime

Per random seed, each SMART run takes a few minutes on one CPU; the baselines are
orders of magnitude faster. A 100-seed sweep across all seven source-matrix choices
totals roughly one CPU-day and is trivial on a small Slurm array.
