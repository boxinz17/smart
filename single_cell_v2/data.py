"""Load the extracted counts file written by extract_cells.py."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Extract:
    counts: sp.csr_matrix          # cells x features, raw counts
    cell_type: np.ndarray
    donor_id: np.ndarray
    donor_number: np.ndarray
    batch: np.ndarray
    feature_name: np.ndarray
    feature_type: np.ndarray       # "GEX" or "ADT"
    provenance: dict


def load_extract(path) -> Extract:
    with np.load(path, allow_pickle=False) as f:
        shape = tuple(int(s) for s in f["counts_shape"])
        counts = sp.csr_matrix((f["counts_data"], f["counts_indices"], f["counts_indptr"]), shape=shape)
        return Extract(counts=counts, cell_type=f["cell_type"], donor_id=f["donor_id"],
                       donor_number=f["donor_number"], batch=f["batch"],
                       feature_name=f["feature_name"], feature_type=f["feature_type"],
                       provenance=json.loads(str(f["provenance"])))


def from_arrays(counts, cell_type, donor_id, feature_type, *, feature_name=None, donor_number=None,
                batch=None) -> Extract:
    """Build an Extract in memory (synthetic tests)."""
    n, m = counts.shape
    return Extract(counts=sp.csr_matrix(counts, dtype=np.float32), cell_type=np.asarray(cell_type),
                   donor_id=np.asarray(donor_id, dtype=np.int64),
                   donor_number=np.asarray(donor_number if donor_number is not None
                                           else [f"donor{d}" for d in donor_id]),
                   batch=np.asarray(batch if batch is not None else ["b"] * n),
                   feature_name=np.asarray(feature_name if feature_name is not None
                                           else [f"f{j}" for j in range(m)]),
                   feature_type=np.asarray(feature_type), provenance=dict(source_file="synthetic"))
