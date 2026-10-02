"""Per-donor paired gaps of SparseSMART against chosen competitors, for exploration variants.

    python code/single_cell_v2/titan/paired_gaps.py RUN_DIR BASE_RUN_DIR FOLDS_JSON variant [variant ...]

Reads result.json files only. Gap = SparseSMART test RMSE minus competitor's;
negative favours SparseSMART. Intervals are t-intervals across held-out donors.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

run, base, folds_path, *variants = sys.argv[1:]
folds = json.loads(Path(folds_path).read_text())
labels = folds["donor_labels"]
names = [f["fold"] for f in folds["folds"]]
donors = [labels[str(f["test_donor"])] for f in folds["folds"]]
competitors = ["nuclear_contrast", "target_ridge", "target_ridge_rrr", "park", "initializer_only"]
for variant in variants:
    root = Path(base) if variant == "base" else Path(run) / variant
    results = [json.loads((root / "main" / n / "result.json").read_text())["methods"] for n in names]
    ours = np.array([r["sparse_smart"]["test"]["rmse"] for r in results])
    print(f"== {variant}: SparseSMART per donor " + " ".join(f"{d}={v:.3f}" for d, v in zip(donors, ours)))
    for c in competitors:
        gap = ours - np.array([r[c]["test"]["rmse"] for r in results])
        half = stats.t.ppf(0.975, len(gap) - 1) * gap.std(ddof=1) / np.sqrt(len(gap))
        print(f"   vs {c:18s} mean {gap.mean():+.4f}  95% CI ({gap.mean() - half:+.4f}, {gap.mean() + half:+.4f})"
              f"  SparseSMART better on {int(np.sum(gap < 0))}/8")
