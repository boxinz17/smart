"""Small deterministic local example, with no simulation-campaign integration."""

from itertools import product

import numpy as np

from sparse_smart_v2 import (
    Margins, ObservedSource, PracticalCalibration, SparseSMARTv2, SparseSMARTv2Tuner,
)


def main():
    rng = np.random.default_rng(2026)
    p, q = 5, 4
    coefficient = np.zeros((p, q))
    coefficient[0, 0] = 2.0
    X = rng.normal(size=(120, p))
    Y = X @ coefficient + 0.05 * rng.normal(size=(120, q))
    Xv = rng.normal(size=(40, p))
    Yv = Xv @ coefficient + 0.05 * rng.normal(size=(40, q))
    source_coefficient = np.zeros((p, q))
    source_coefficient[:3, :3] = np.diag([3.0, 2.0, 1.0])
    source = ObservedSource(source_coefficient)
    candidates = [
        SparseSMARTv2(
            rank=1, source_rank=3, free_directions=free,
            margins=Margins(0.01, 8.0, 0.005, anchor_min=0.005),
            calibration=PracticalCalibration(
                init_penalty=0.01, penalty=penalty,
                step_size_inverse=20.0, support_limits=(1, 1),
            ),
            iterations=30, validation_interval=5,
        )
        for free, penalty in product(((1, 1), (2, 2)), (0.0, 0.02))
    ]
    tuner = SparseSMARTv2Tuner(candidates).fit(X, Y, source=source, validation_data=(Xv, Yv))
    for result in tuner.results_:
        print(f"candidate={result['candidate_index']} status={result['status']} "
              f"eligible={result['eligible']} validation_mse={result['validation_mse']}")
    if not tuner.success_:
        raise RuntimeError(tuner.message_)
    print(f"selected_candidate={tuner.best_index_} selected_iteration={tuner.selected_iteration_}")
    print(f"validation_mse={tuner.best_score_:.6f}")
    print(f"prediction_shape={tuner.predict(Xv).shape}")
    print(f"theorem_certified={tuner.metadata_['theorem_certified']}")


if __name__ == "__main__":
    main()
