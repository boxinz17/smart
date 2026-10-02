"""Step 0 of the NK -> ILC1 application: read cell metadata only.

Reads the cell (obs) and feature (var) tables, and checks whether the expression
matrix holds raw counts or normalized values. It never reads protein values for
modeling and fits nothing. The expression matrix stays on disk; only the first
entries of its stored values are sampled for the format check.

    code/.venv/bin/python code/single_cell_v2/inspect_metadata.py --data <h5ad> --out <json>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

CELL_TYPES = ("NK", "ILC1")
DONOR_COLUMNS = ("DonorID", "DonorNumber", "Site", "batch", "Samplename")


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def read_frame(group):
    """Decode an AnnData (legacy layout) dataframe group into plain arrays."""
    categories = group.get("__categories", {})
    columns = {}
    for name, dataset in group.items():
        if name == "__categories" or not isinstance(dataset, h5py.Dataset):
            continue
        values = dataset[()]
        if name in categories:
            labels = np.array([_decode(v) for v in categories[name][()]], dtype=object)
            values = labels[values]
        elif values.dtype == object:
            values = np.array([_decode(v) for v in values], dtype=object)
        columns[name] = values
    return columns


def value_format(group, sample=200_000):
    data = group["data"][:sample]
    return dict(
        stored_entries=int(group["data"].shape[0]),
        sampled=int(len(data)),
        integer_valued=bool(np.all(data == np.round(data))),
        min=float(data.min()),
        max=float(data.max()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with h5py.File(args.data, "r") as f:
        obs = read_frame(f["obs"])
        var = read_frame(f["var"])
        x_format = value_format(f["X"])
        counts_format = value_format(f["layers"]["counts"])
        uns = {k: _decode(f["uns"][k][()]) for k in f["uns"]}

    cell_type = obs["cell_type"]
    summary = dict(
        source_file=args.data.name,
        n_cells=int(len(cell_type)),
        features=dict(Counter(var["feature_types"])),
        uns=uns,
        obs_columns=sorted(obs),
        X_format=x_format,
        counts_layer_format=counts_format,
        cell_type_counts=dict(Counter(cell_type).most_common()),
    )

    # Which sites and batches each donor appears in (all cells).
    donor_sites = defaultdict(set)
    donor_batches = defaultdict(set)
    for donor, site, batch in zip(obs["DonorID"], obs["Site"], obs["batch"]):
        donor_sites[int(donor)].add(str(site))
        donor_batches[int(donor)].add(str(batch))
    summary["donors"] = {
        str(d): dict(sites=sorted(donor_sites[d]), batches=sorted(donor_batches[d]))
        for d in sorted(donor_sites)
    }

    for kind in CELL_TYPES:
        mask = cell_type == kind
        block = dict(n_cells=int(mask.sum()))
        for column in DONOR_COLUMNS:
            block[f"by_{column}"] = {str(k): int(v) for k, v in sorted(Counter(obs[column][mask]).items(),
                                                                     key=lambda kv: str(kv[0]))}
        summary[kind] = block

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
