"""Reproducible practical fits; run after installing sparse-smart."""
import json
import numpy as np
from sparse_smart import ExactSource, NoisySource, Margins, PracticalCalibration, SparseSMART


def run(mode):
    rng = np.random.default_rng(2026)
    n, p, q, r0, r = 160, 10, 9, 7, 2
    U = np.linalg.qr(rng.normal(size=(p, r0)))[0]
    V = np.linalg.qr(rng.normal(size=(q, r0)))[0]
    P, Q = np.zeros((r0, r)), np.zeros((r0, r))
    # Multiple source directions per component, with overlapping row supports.
    P[:4] = np.linalg.qr(np.array([[1., .2], [.4, 1.], [.3, -.2], [.1, .3]]))[0]
    Q[:3] = np.linalg.qr(np.array([[1., .3], [.2, 1.], [.4, -.2]]))[0]
    truth = (U @ P * [3., 1.2]) @ (V @ Q).T
    X = rng.normal(size=(n, p))
    Y = X @ truth + .01 * rng.normal(size=(n, q))
    if mode == "exact":
        source = ExactSource(U, V)
        limits = (8, 8)
    else:
        spectrum = [9., 9., 6., 5., 4., 3., 2.] if mode == "cluster" else [9., 7., 6., 5., 4., 3., 2.]
        tau = 1e-5
        observed = (U * spectrum) @ V.T + tau * rng.normal(size=(p, q))
        source = NoisySource(observed, tau, gap_lower=1., cluster_size=2 if mode == "cluster" else 1)
        limits = (12, 12)
    model = SparseSMART(rank=r, source_rank=r0, sparsity=(8, 6),
        margins=Margins(.2, 6., .3, trial_radius=.5),
        calibration=PracticalCalibration(.003, .001, 10., limits), iterations=20)
    model.fit(X, Y, source=source)
    if not model.success_:
        raise RuntimeError(f"{mode}: {model.status_}: {model.message_}")
    return {"source_mode": mode, "status": model.status_, "iterations": model.n_iter_,
            "coefficient_error": float(np.linalg.norm(model.coefficient_ - truth)),
            "prediction_error": float(np.linalg.norm(X @ (model.coefficient_ - truth))**2 / n),
            "initial_objective": model.history_[0].objective,
            "final_objective": model.history_[-1].objective,
            "theorem_certified": model.diagnostics_["theorem_certified"]}


if __name__ == "__main__":
    print(json.dumps([run(mode) for mode in ("exact", "noisy", "cluster")], indent=2))

