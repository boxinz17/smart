"""Independent tuning observations conditional on a legacy simulation model.

Generate the original training experiment first, preserving its fixed-seed
arrays exactly. A separate PCG64 stream generates validation X and noise using
the same coefficient, target noise scale, and AR(1) design covariance. Truth is
used here to simulate responses, never supplied to the tuning estimator.
"""
from numbers import Integral, Real

import numpy as np


VALIDATION_SEED_TAG = 1397970481  # ASCII SSV1, a fixed domain separator.


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def generate_external_validation(*, n_train, p, q, sigma0, random_seed,
                                 n_validation=100, sigma=.5, r_star=5, r0_star=10,
                                 seed_tag=VALIDATION_SEED_TAG, generate_data_fn=None):
    """Return unchanged training arrays plus independent validation observations.

    The same legacy seed and n_train are passed to generate_data. Calling that
    generator with n_train+n_validation, or calling it a second time, would
    change its coefficient draw and is deliberately avoided.
    """
    n_train, p, q, n_validation = (
        _integer(value, name, minimum=1) for value, name in
        ((n_train, "n_train"), (p, "p"), (q, "q"), (n_validation, "n_validation")))
    random_seed = _integer(random_seed, "random_seed")
    seed_tag = _integer(seed_tag, "seed_tag")
    if random_seed >= 2**32:
        raise ValueError("random_seed must fit the legacy generator's 32-bit seed range")
    for value, name in ((sigma, "sigma"), (sigma0, "sigma0")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if generate_data_fn is None:
        from smart import generate_data
        generate_data_fn = generate_data
    training = dict(generate_data_fn(n=n_train, p=p, q=q, sigma0=float(sigma0),
                                    sigma=float(sigma), r_star=r_star, r0_star=r0_star,
                                    random_seed=random_seed))
    for name, shape in (("X", (n_train, p)), ("Y", (n_train, q)),
                        ("C_star", (p, q)), ("C0", (p, q))):
        value = np.asarray(training[name])
        if np.iscomplexobj(value) or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Generator's {name} must be a finite real array of shape {shape}")
    covariance = .5 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([random_seed, seed_tag])))
    validation_X = rng.multivariate_normal(np.zeros(p), covariance,
                                           size=n_validation, method="cholesky")
    validation_noise = rng.normal(size=(n_validation, q)) * sigma
    validation_Y = validation_X @ training["C_star"] + validation_noise
    if not np.isfinite(validation_Y).all():
        raise FloatingPointError("Generated validation responses are nonfinite")
    training.update(X_validation=validation_X, Y_validation=validation_Y,
        validation_seed_metadata=dict(seed_sequence_entropy=[random_seed, seed_tag],
            bit_generator="PCG64", covariance="AR1", covariance_rho=.5,
            design_factorization="cholesky", noise_std=float(sigma),
            conditional_on_same_coefficient=True, source_reused=True))
    return training
