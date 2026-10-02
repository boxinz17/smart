"""Leave-one-donor-out folds, built from cell counts only.

Each fold names its test donor, whole validation donors, target training
donors, the source donors (everyone except the test donor), and the
donor-blocked part of the source cells held out to tune the source fit.

    code/.venv/bin/python code/single_cell_v2/folds.py --counts <extract.npz> --out folds.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from config import CONFIG, fingerprint
from data import load_extract


def choose_holdout(counts, share):
    """Subset of donors whose cell share is closest to `share`.

    Ties go to fewer donors, then to the lexicographically smallest donor list.
    """
    donors = sorted(counts)
    total = sum(counts.values())
    if len(donors) < 2 or total == 0:
        raise ValueError("need at least two donors with cells")
    best = None
    for size in range(1, len(donors)):
        for subset in combinations(donors, size):
            key = (abs(sum(counts[d] for d in subset) / total - share), size, subset)
            if best is None or key < best:
                best = key
    return list(best[2])


def build_folds(extract, config=CONFIG):
    source, target = config["cell_types"]["source"], config["cell_types"]["target"]
    labels = {int(d): str(n) for d, n in zip(extract.donor_id, extract.donor_number)}
    target_counts = Counter(int(d) for d, t in zip(extract.donor_id, extract.cell_type) if t == target)
    source_counts = Counter(int(d) for d, t in zip(extract.donor_id, extract.cell_type) if t == source)
    donors = sorted(set(target_counts) | set(source_counts))
    eligible = [d for d in donors if target_counts[d] >= config["min_test_cells"]]
    folds = []
    for test in eligible:
        pool = {d: target_counts[d] for d in donors if d != test and target_counts[d] > 0}
        validation = choose_holdout(pool, config["validation_share"])
        training = [d for d in sorted(pool) if d not in validation]
        source_donors = [d for d in donors if d != test and source_counts[d] > 0]
        source_holdout = choose_holdout({d: source_counts[d] for d in source_donors},
                                        config["source_holdout_share"])
        folds.append(dict(
            fold=f"donor-{test}", test_donor=test, test_label=labels[test],
            validation_donors=validation, training_donors=training,
            source_donors=source_donors, source_holdout_donors=source_holdout,
            n_test=target_counts[test], n_validation=sum(pool[d] for d in validation),
            n_training=sum(pool[d] for d in training),
            n_source=sum(source_counts[d] for d in source_donors),
            n_source_holdout=sum(source_counts[d] for d in source_holdout)))
    return dict(config_sha256=fingerprint(config), donor_labels=labels,
                target_cells=dict(sorted(target_counts.items())),
                source_cells=dict(sorted(source_counts.items())),
                never_tested=[d for d in donors if d not in eligible], folds=folds)


def load_folds(path):
    folds = json.loads(Path(path).read_text())
    folds["donor_labels"] = {int(k): v for k, v in folds["donor_labels"].items()}
    return folds


def tasks(folds, config=CONFIG):
    """Slurm task list: every fold under every variant, main variant first."""
    return [dict(fold=f["fold"], variant=v) for v in config["variants"] for f in folds["folds"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    folds = build_folds(load_extract(args.counts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(folds, indent=1) + "\n")
    for fold in folds["folds"]:
        print(fold["fold"], fold["test_label"], "test", fold["n_test"], "validation",
              fold["validation_donors"], fold["n_validation"], "training", fold["n_training"],
              "source", fold["n_source"], "source holdout", fold["source_holdout_donors"])


if __name__ == "__main__":
    main()
