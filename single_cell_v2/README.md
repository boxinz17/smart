# NK → ILC1 application, leave one donor out

Implements `response_r1/APPLICATION_ANALYSIS_PLAN.md`: SparseSMART and its
competitors on CITE-seq bone-marrow cells (GEO GSE194122), with NK cells as
the source and ILC1 cells as the target. Each fold holds out one donor; every
preprocessing statistic, the source fit, the ranks and all tuning use the
other donors only, and test cells are scored only after every method is
selected. The v1 scripts in `code/single_cell/` are left unchanged.

Data stay outside Dropbox, in `~/data/SMART_BoxinJinchi/single_cell/` locally
and `/shared/home/mladen.kolar/SMART_BoxinJinchi/data/single_cell/` on Titan.

## Files

| File | Role |
|---|---|
| `config.py` | Frozen protocol; every result records its SHA-256 |
| `inspect_metadata.py` | Step 0: cell and feature tables only |
| `extract_cells.py` | Raw counts of NK and ILC1 cells (row selection only, ~16 MB) |
| `folds.py` | Folds from per-donor cell counts; the task list (folds × variants) |
| `preprocess.py` | Per-cell transforms, then per-fold genes, scaling, screening, centering |
| `modeling.py` | Source fit, ranks, data scales, SparseSMART and every competitor |
| `diagnostics.py` | Low rank, containment angles, alignment concentration |
| `run_fold.py` | One task: one fold under one variant |
| `summarize.py` | Per-donor tables and paired comparisons across donors |
| `titan/` | Environment setup and the Slurm job for Titan |
| `tests/` | Synthetic checks, including a leakage test |

SparseSMART reuses the structural campaign's 211-candidate library and fitting
loop (`code/simulation/reviewer_revision_20260913/runner.py`); its penalties are
multiplied by σ̂/0.5 and its spectral margins by d̂₁/5, from the target-only RRR
fit on training cells. The comparators are the original-design ones
(`code/simulation/paper_source_comparators_20260913`). Park et al. runs through
the campaign's bridge to the authors' R code, in the main variant only (it uses
raw source cells and no rank, so no variant changes it).

## Run

Run Python from this folder or the repository root: from `code/`, the
`code/smart/` project folder shadows the installed `smart` package.

```bash
PY=code/.venv/bin/python
RAW=~/data/SMART_BoxinJinchi/single_cell/raw/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad
COUNTS=~/data/SMART_BoxinJinchi/single_cell/derived/nk_ilc1_counts.npz
FOLDS=response_r1/numerical_evidence/single_cell_v2/folds.json

$PY code/single_cell_v2/extract_cells.py --data $RAW --out $COUNTS
$PY code/single_cell_v2/folds.py --counts $COUNTS --out $FOLDS
# smoke: a few candidates, short trajectories, no test scoring
$PY code/single_cell_v2/run_fold.py --counts $COUNTS --folds $FOLDS --out RUN_DIR \
    --fold donor-15078 --variant main --mode smoke
(cd code/single_cell_v2 && ../.venv/bin/python -m pytest -q tests)
```

On Titan (see `.slurm/runs/` for the run record):

```bash
bash code/single_cell_v2/titan/setup_env.sh          # once, on the login node
sbatch --parsable --job-name=RUNTAG --comment=RUNTAG \
    --export=ALL,RUNTAG=RUNTAG,MODE=production code/single_cell_v2/titan/run_tasks.sbatch
```

Then copy `runs/RUNTAG` back and run
`summarize.py --run RUN_DIR --folds $FOLDS --out response_r1/numerical_evidence/single_cell_v2/RUNTAG`.
