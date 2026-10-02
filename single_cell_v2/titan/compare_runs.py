"""Check that a run reproduces earlier runs, method by method and fold by fold.

    python code/single_cell_v2/titan/compare_runs.py FOLDS NEW_DIR/VARIANT OLD_DIR/VARIANT [NEW OLD ...]

Each directory holds <fold>/result.json (and predictions.npz). Prints the largest
absolute difference in test RMSE, and whether the saved predictions load.
"""

import json
import sys
from pathlib import Path

import numpy as np

folds = [f["fold"] for f in json.loads(Path(sys.argv[1]).read_text())["folds"]]
pairs = sys.argv[2:]
for new, old in zip(pairs[::2], pairs[1::2]):
    worst, compared = 0.0, 0
    for fold in folds:
        a = json.loads((Path(new) / fold / "result.json").read_text())
        b = json.loads((Path(old) / fold / "result.json").read_text())
        assert a["complete"] and a["mode"] == "production"
        with np.load(Path(new) / fold / "predictions.npz") as saved:
            assert np.isfinite(saved["Y_test"]).all()
        for method, record in a["methods"].items():
            if record.get("success") and method in b["methods"]:
                worst = max(worst, abs(record["test"]["rmse"] - b["methods"][method]["test"]["rmse"]))
                compared += 1
    print(f"{new} vs {old}: {compared} method-fold pairs, max |test RMSE difference| = {worst:.3e}")
