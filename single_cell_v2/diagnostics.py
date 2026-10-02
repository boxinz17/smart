"""Diagnostics of the transfer mechanism (Reviewer 1, comment 6; Reviewer 2, (f)).

All quantities use training cells, plus validation cells for the rank curve;
test cells never enter. Containment is checked against the leading r0 source
directions, because against full source frames it holds trivially.
"""

from __future__ import annotations

import numpy as np

from folds import choose_holdout
from sparse_smart_v2 import target_rrr


def leading_subspaces(coefficient, rank):
    U, _, Vt = np.linalg.svd(coefficient, full_matrices=False)
    return U[:, :rank], Vt[:rank].T


def cosines(A, B):
    """Cosines of the principal angles between the column spans of orthonormal A and B."""
    return np.clip(np.linalg.svd(A.T @ B, compute_uv=False), 0.0, 1.0)


def alignment(frame, directions, rank0):
    """Energy of each target direction across the ordered source coordinates."""
    energy = (frame.T @ directions) ** 2
    energy = energy / energy.sum(axis=0, keepdims=True)
    ordered = -np.sort(-energy, axis=0)
    cumulative = np.cumsum(ordered, axis=0)
    return dict(
        in_leading_r0=energy[:rank0].sum(axis=0).tolist(),
        top1=cumulative[0].tolist(), top3=cumulative[min(2, len(cumulative) - 1)].tolist(),
        n_for_90_percent=(np.argmax(cumulative >= 0.9, axis=0) + 1).tolist(),
        participation=(1.0 / np.sum(energy ** 2, axis=0)).tolist(),
    )


def random_reference(directions, dimension, draws, rng):
    """Principal-angle cosines between `directions` and random `dimension`-dimensional spans."""
    values = np.array([cosines(directions, np.linalg.qr(rng.standard_normal((directions.shape[0], dimension)))[0])
                       for _ in range(draws)])
    return dict(mean=values.mean(axis=0).tolist(), q95=np.quantile(values, 0.95, axis=0).tolist(),
                expected_squared_cosine=dimension / directions.shape[0])


def half_split_floor(fd, rank):
    """Angles between target fits on two donor-disjoint halves of the training cells."""
    counts = {int(d): int(np.sum(fd.train_donor == d)) for d in np.unique(fd.train_donor)}
    if len(counts) < 2:
        return None
    first = np.isin(fd.train_donor, choose_holdout(counts, 0.5))
    halves = []
    for mask in (first, ~first):
        X, Y = fd.X_train[mask], fd.Y_train[mask]
        halves.append(leading_subspaces(target_rrr(X - X.mean(0), Y - Y.mean(0), rank).coefficient, rank))
    return dict(left=cosines(halves[0][0], halves[1][0]).tolist(), right=cosines(halves[0][1], halves[1][1]).tolist(),
                n_cells=[int(first.sum()), int((~first).sum())])


def run(fd, bases, rank, rank0, rsc_target, rsc_source, config):
    settings = config["diagnostics"]
    rng = np.random.default_rng(settings["seed"])
    target_fit = target_rrr(fd.X_train, fd.Y_train, rank).coefficient
    U, V = leading_subspaces(target_fit, rank)
    curve = []
    for k in range(1, min(settings["max_rank_curve"], fd.Y_train.shape[1]) + 1):
        coefficient = target_rrr(fd.X_train, fd.Y_train, k).coefficient
        curve.append(dict(rank=k, validation_mse=float(np.mean((fd.Y_val - fd.X_val @ coefficient) ** 2))))
    leading_left, leading_right = np.array(bases.leading_left), np.array(bases.leading_right)
    return dict(
        low_rank=dict(target_rsc=rsc_target, source_rsc=rsc_source, target_rrr_validation_curve=curve),
        containment=dict(
            left_cosines=cosines(U, leading_left).tolist(), right_cosines=cosines(V, leading_right).tolist(),
            random_left=random_reference(U, leading_left.shape[1], settings["random_reference_draws"], rng),
            random_right=random_reference(V, leading_right.shape[1], settings["random_reference_draws"], rng),
            half_split_floor=half_split_floor(fd, rank)),
        alignment=dict(left=alignment(np.array(bases.left), U, rank0),
                       right=alignment(np.array(bases.right), V, rank0)),
    )
