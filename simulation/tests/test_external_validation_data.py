import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION))
from external_validation_data import generate_external_validation, VALIDATION_SEED_TAG

SPEC = importlib.util.spec_from_file_location('external_test_legacy_data',
    SIMULATION.parent / 'smart' / 'smart' / 'utils.py')
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


def inputs(**overrides):
    values = dict(n_train=200, n_validation=100, p=8, q=7, sigma0=.1,
                  sigma=.5, random_seed=42, r_star=2, r0_star=4,
                  generate_data_fn=legacy.generate_data)
    values.update(overrides)
    return values


def test_training_data_model_and_global_rng_are_exactly_preserved():
    expected = legacy.generate_data(n=200,p=8,q=7,sigma0=.1,sigma=.5,
                                   random_seed=42,r_star=2,r0_star=4)
    expected_state = np.random.get_state()
    actual = generate_external_validation(**inputs())
    actual_state = np.random.get_state()
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value)
    assert actual_state[0] == expected_state[0] and actual_state[2:] == expected_state[2:]
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual['X'].shape == (200,8) and actual['X_validation'].shape == (100,8)
    assert actual['Y'].shape == (200,7) and actual['Y_validation'].shape == (100,7)
    assert not np.array_equal(actual['X'][:100],actual['X_validation'])


def test_validation_uses_same_coefficient_and_a_reproducible_independent_stream():
    first = generate_external_validation(**inputs(sigma=0.))
    again = generate_external_validation(**inputs(sigma=0.))
    np.testing.assert_array_equal(first['X_validation'],again['X_validation'])
    np.testing.assert_allclose(first['Y_validation'],first['X_validation'] @ first['C_star'])
    changed = generate_external_validation(**inputs(sigma=0.,seed_tag=VALIDATION_SEED_TAG+1))
    for name in ('X','Y','C_star','C0'):
        np.testing.assert_array_equal(first[name],changed[name])
    assert not np.array_equal(first['X_validation'],changed['X_validation'])


def test_design_covariance_and_independent_response_noise_match_model():
    data = generate_external_validation(**inputs(n_validation=20000,p=4,q=3,r_star=2,r0_star=3))
    X = data['X_validation']
    residual = data['Y_validation'] - X @ data['C_star']
    target = .5 ** np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])
    np.testing.assert_allclose(np.cov(X,rowvar=False),target,atol=.04)
    assert abs(residual.std()-.5) < .01
    assert abs(residual.mean()) < .01
    assert np.max(np.abs(X.T @ residual / len(X))) < .02


@pytest.mark.parametrize('overrides', [dict(n_validation=0),dict(n_validation=True),
    dict(n_train=-1),dict(random_seed=2**32),dict(random_seed=-1),dict(seed_tag=True),
    dict(sigma=-1),dict(sigma0=np.inf)])
def test_invalid_sizes_seeds_and_scales_rejected(overrides):
    with pytest.raises(ValueError):
        generate_external_validation(**inputs(**overrides))


def test_only_one_legacy_generation_is_performed_at_training_n():
    calls=[]
    def generator(**kwargs):
        calls.append(kwargs)
        return legacy.generate_data(**kwargs)
    generate_external_validation(**inputs(generate_data_fn=generator))
    assert len(calls)==1 and calls[0]['n']==200
