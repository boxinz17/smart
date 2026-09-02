"""Regression tests for scale-local BI-SMART numerical policies."""

from __future__ import annotations

import numpy as np
import pytest

from bi_smart.refinement import (
    BISMARTState,
    RefinementError,
    block_diagonal,
    default_core_thresholds,
    fitted_source,
    fitted_target,
    solve_quotient_gauss_newton,
    strict_polar_factor,
)
from bi_smart.types import FailureReason, NumericalFailure


def _two_block_state(*, h_scale: float, g_scales: tuple[float, float]) -> BISMARTState:
    """Build a regular state whose three core scales can be varied separately."""

    return BISMARTState(
        U0=np.eye(2),
        V0=np.eye(2),
        A=np.array([[1.0], [0.0]]),
        B=np.array([[0.0], [1.0]]),
        H=np.array([[h_scale]]),
        G_blocks=(np.array([[g_scales[0]]]), np.array([[g_scales[1]]])),
    )


def test_default_core_thresholds_use_each_core_local_scale() -> None:
    """A huge target core cannot make small regular source cores look singular."""

    state = _two_block_state(h_scale=1e12, g_scales=(1e-4, 3e-4))
    thresholds = default_core_thresholds(
        state,
        radius=2.0,
        atol=0.0,
        rtol=1e-8,
    )

    assert thresholds.h_min == pytest.approx(5e11)
    assert thresholds.g_min == pytest.approx(5e-5)
    # Squared spectra are 1e-8 and 9e-8; their own local separation rule
    # certifies the 8e-8 gap independently of the 1e12 target-core scale.
    assert thresholds.separation_min == pytest.approx(4e-8)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message_name"),
    [
        ("U0", np.diag([1.0, 1.001]), "U0"),
        ("V0", np.diag([1.001, 1.0]), "V0"),
        ("A", np.array([[1.0], [0.1]]), "A_active"),
        ("B", np.array([[0.1], [1.0]]), "B_active"),
    ],
)
def test_public_state_requires_every_stiefel_factor_to_be_orthonormal(
    field: str,
    invalid_value: np.ndarray,
    message_name: str,
) -> None:
    """Direct public construction cannot bypass a product-manifold invariant."""

    values = {
        "U0": np.eye(2),
        "V0": np.eye(2),
        "A": np.array([[1.0], [0.0]]),
        "B": np.array([[0.0], [1.0]]),
        "H": np.array([[1.0]]),
        "G_blocks": (np.array([[2.0]]), np.array([[0.5]])),
    }
    values[field] = invalid_value
    with pytest.raises(ValueError, match=rf"{message_name} must have orthonormal"):
        BISMARTState(**values)


@pytest.mark.parametrize("invalid_mask", [np.array([1.0, 0.5]), np.array([1.0, np.nan])])
def test_public_state_rejects_nonboolean_support_values(
    invalid_mask: np.ndarray,
) -> None:
    """Support normalization cannot silently turn arbitrary numbers into True."""

    with pytest.raises(ValueError, match="boolean or exact zero/one"):
        BISMARTState(
            U0=np.eye(2),
            V0=np.eye(2),
            A=np.array([[1.0], [0.0]]),
            B=np.array([[1.0], [0.0]]),
            H=np.array([[1.0]]),
            G_blocks=(np.eye(2),),
            active_u=invalid_mask,
            active_v=np.ones(2, dtype=bool),
        )


def test_public_block_diagonal_rejects_complex_cores() -> None:
    """The standalone helper cannot silently assign complex data into float."""

    with pytest.raises(ValueError, match="real-valued"):
        block_diagonal((np.array([[1.0 + 2.0j]]),))


def test_each_polar_factor_uses_its_own_singular_value_scale() -> None:
    """Common conditioning receives the same decision at very different scales."""

    large = np.diag([1e12, 1e6])
    small = np.diag([1e-12, 1e-18])
    np.testing.assert_allclose(
        strict_polar_factor(large, atol=0.0, rtol=1e-8), np.eye(2)
    )
    np.testing.assert_allclose(
        strict_polar_factor(small, atol=0.0, rtol=1e-8), np.eye(2)
    )

    # Both matrices become numerically deficient under the same relative rule
    # once their condition ratio falls below that rule.
    with pytest.raises(RefinementError, match="rank deficient"):
        strict_polar_factor(np.diag([1e12, 1e2]), atol=0.0, rtol=1e-8)
    with pytest.raises(RefinementError, match="rank deficient"):
        strict_polar_factor(np.diag([1e-12, 1e-22]), atol=0.0, rtol=1e-8)

    # Both configured terms contribute to the threshold; max(atol, rtol*s)
    # would incorrectly accept this second singular value.
    with pytest.raises(RefinementError, match="rank deficient"):
        strict_polar_factor(
            np.diag([1.0, 1.5e-8]),
            atol=1e-8,
            rtol=1e-8,
        )


def test_legacy_zero_tolerance_still_controls_spectral_separation() -> None:
    """The pre-local-policy override retains its documented squared-scale use."""

    state = _two_block_state(h_scale=1.0, g_scales=(1e-4, 3e-4))
    with pytest.raises(RefinementError, match="spectra.*not separated"):
        default_core_thresholds(
            state,
            radius=1.0,
            zero_tolerance=1e-7,
        )


def test_vertical_rank_check_is_independent_across_core_scales() -> None:
    """A huge source block cannot erase a small block's independent gauges."""

    state = BISMARTState(
        U0=np.eye(4),
        V0=np.eye(4),
        A=np.array([[1.0], [0.0], [0.0], [0.0]]),
        B=np.array([[0.0], [0.0], [1.0], [0.0]]),
        H=np.array([[1.0]]),
        G_blocks=(np.diag([1e12, 2e12]), np.diag([1.0, 2.0])),
    )
    X = np.vstack((np.eye(4), np.array([[1.0, 2.0, 3.0, 4.0]])))

    # The later quotient Jacobian may legitimately be judged ill-conditioned
    # in the product metric.  It must not be misclassified earlier as an
    # irregular state merely because raw gauge columns have different units.
    try:
        result = solve_quotient_gauss_newton(
            state,
            X,
            X @ fitted_target(state),
            fitted_source(state),
            0.5,
        )
    except NumericalFailure as failure:
        assert failure.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    else:
        assert result.diagnostics.vertical_dimension == 4
