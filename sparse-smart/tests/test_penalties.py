import numpy as np
import pytest

from sparse_smart import Margins, PracticalCalibration, ResolvedCalibration, resolve_calibration


@pytest.mark.parametrize('penalty', [.02, (.02, .07), [0., .07]])
def test_scalar_and_separate_penalties_resolve(penalty):
    config = PracticalCalibration(.01, penalty, 1., (1, 1))
    resolved = resolve_calibration(config, n=40, p=4, q=3, rank=1, source_rank=2,
        sparsity=(1, 1), margins=Margins(.1, 5., .1), source_noise=0., source_gap=None)
    expected = (.02, .02) if np.isscalar(penalty) else tuple(penalty)
    assert resolved.penalties == expected
    assert resolved.diagnostics['penalties'] == expected
    if isinstance(penalty, list):
        penalty[0] = 8.
        assert config.penalty == (0., .07)


@pytest.mark.parametrize('penalty', [(-.1, .1), (0., np.inf), (True, .1),
                                   (.1,), (.1, .2, .3), '0.01'])
def test_malformed_separate_penalties_rejected(penalty):
    with pytest.raises(ValueError):
        PracticalCalibration(.01, penalty, 1., (1, 1))


def test_direct_resolved_penalty_validation():
    resolved = ResolvedCalibration(.01, (.01, .1), 1., (1, 1), 'practical', {})
    assert resolved.penalties == (.01, .1)
    with pytest.raises(ValueError):
        _ = ResolvedCalibration(.01, (np.nan, .1), 1., (1, 1), 'practical', {}).penalties
