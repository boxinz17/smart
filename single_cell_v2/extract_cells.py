"""Extract raw counts and cell metadata for the source and target cell types.

Only rows are selected: no normalization or other statistic is computed here,
so the output carries no information across cells. All later preprocessing
starts from these raw counts inside each fold.

    code/.venv/bin/python code/single_cell_v2/extract_cells.py \
        --data ~/data/SMART_BoxinJinchi/single_cell/raw/<file>.h5ad \
        --out ~/data/SMART_BoxinJinchi/single_cell/derived/nk_ilc1_counts.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from config import CONFIG
from inspect_metadata import read_frame

OBS_FIELDS = dict(barcode="_index", cell_type="cell_type", donor_id="DonorID",
                  donor_number="DonorNumber", site="Site", batch="batch")


def sha256(path, block=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(h5ad, cell_types):
    with h5py.File(h5ad, "r") as f:
        obs, var = read_frame(f["obs"]), read_frame(f["var"])
        keep = np.flatnonzero(np.isin(obs["cell_type"], list(cell_types)))
        group = f["layers"]["counts"]
        shape = tuple(int(s) for s in group.attrs["shape"])
        counts = sp.csr_matrix((group["data"][()], group["indices"][()], group["indptr"][()]),
                               shape=shape)[keep]
    counts.sort_indices()
    arrays = {name: np.asarray(obs[column][keep]) for name, column in OBS_FIELDS.items()}
    arrays["donor_id"] = arrays["donor_id"].astype(np.int64)
    for name in ("barcode", "cell_type", "donor_number", "site", "batch"):
        arrays[name] = arrays[name].astype(str)
    arrays.update(
        counts_data=counts.data.astype(np.float32), counts_indices=counts.indices,
        counts_indptr=counts.indptr, counts_shape=np.array(counts.shape),
        feature_name=np.asarray(var["_index"]).astype(str),
        feature_type=np.asarray(var["feature_types"]).astype(str),
    )
    if not np.all(arrays["counts_data"] == np.round(arrays["counts_data"])):
        raise ValueError("counts layer is not integer-valued")
    return arrays


def extract_isotypes(h5ad, barcodes):
    """Isotype-control values for the given cells, in the given order.

    The file stores them already normalized (not raw counts); they serve only
    as a per-cell technical covariate.
    """
    with h5py.File(h5ad, "r") as f:
        group = f["obsm"]["ADT_isotype_controls"]
        names = [str(c) for c in group.attrs["column-order"]]
        index = {b.decode() if isinstance(b, bytes) else str(b): i for i, b in enumerate(group["_index"][()])}
        rows = np.array([index[b] for b in barcodes])
        values = np.column_stack([group[name][()][rows] for name in names]).astype(np.float64)
    return dict(barcode=np.asarray(barcodes).astype(str), isotypes=values, names=np.array(names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--isotypes-only", action="store_true",
                        help="write only the isotype file next to an existing --out extract")
    args = parser.parse_args()
    if args.isotypes_only:
        with np.load(args.out, allow_pickle=False) as existing:
            barcodes = existing["barcode"]
        target = args.out.with_name(args.out.stem + "_isotypes.npz")
        np.savez_compressed(target, **extract_isotypes(args.data, barcodes))
        print(f"wrote {target}")
        return
    cell_types = (CONFIG["cell_types"]["source"], CONFIG["cell_types"]["target"])
    arrays = extract(args.data, cell_types)
    provenance = dict(source_file=args.data.name, source_sha256=sha256(args.data),
                      cell_types=list(cell_types), n_cells=int(arrays["counts_shape"][0]),
                      n_features=int(arrays["counts_shape"][1]),
                      cells_by_type={t: int(np.sum(arrays["cell_type"] == t)) for t in cell_types})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, provenance=json.dumps(provenance), **arrays)
    print(json.dumps(provenance, indent=1))


if __name__ == "__main__":
    main()
