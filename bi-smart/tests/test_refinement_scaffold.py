"""Focused checks for the executable part of the refinement scaffold."""

from __future__ import annotations

import numpy as np
import pytest

from bi_smart.refinement import (
    BISMARTDirection,
    BISMARTState,
    TrustRegionThresholds,
    check_path_certificate,
    euclidean_gradients,
    fitted_source,
    fitted_target,
    joint_objective,
    product_inner_product,
    project_to_tangent,
    quotient_gauss_newton_direction,
    retract_state,
    stiefel_error,
)


def _tiny_problem() -> tuple[BISMARTState, np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct a deterministic, regular two-block/rank-one example."""

    # U0 and V0 have orthonormal columns, while A and B are unit vectors in
    # the full selected source-coordinate space.  Both one-dimensional source
    # cores are nonsingular and have separated squared spectra.
    state = BISMARTState(
        U0=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        V0=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        A=np.array([[1.0], [0.0]]),
        B=np.array([[0.0], [1.0]]),
        H=np.array([[1.5]]),
        G_blocks=(np.array([[2.0]]), np.array([[0.75]])),
    )
    X = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.25],
            [-1.0, 0.5, 1.0],
        ]
    )
    Y = np.array(
        [
            [0.5, 1.0, -0.5],
            [1.0, -0.5, 0.25],
            [0.0, 0.75, 1.0],
            [-0.25, 0.5, 0.0],
        ]
    )
    observed_source = np.array(
        [
            [1.8, 0.1, 0.0],
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.0],
        ]
    )
    return state, X, Y, observed_source, 0.4


def _shift_state(
    state: BISMARTState, direction: BISMARTDirection, scale: float
) -> BISMARTState:
    """Take an ambient finite-difference step without applying a retraction."""

    return BISMARTState(
        U0=state.U0 + scale * direction.U0,
        V0=state.V0 + scale * direction.V0,
        A=state.A + scale * direction.A,
        B=state.B + scale * direction.B,
        H=state.H + scale * direction.H,
        G_blocks=tuple(
            core + scale * core_direction
            for core, core_direction in zip(
                state.G_blocks, direction.G_blocks, strict=True
            )
        ),
        active_u=state.active_u,
        active_v=state.active_v,
    )


def test_joint_objective_and_explicit_gradients() -> None:
    """The objective matches its definition and gradients its differential."""

    state, X, Y, observed_source, omega = _tiny_problem()
    target = fitted_target(state)
    source = fitted_source(state)
    expected = (
        0.5 * np.linalg.norm(Y - X @ target, ord="fro") ** 2 / X.shape[0]
        + 0.5
        * omega
        * np.linalg.norm(observed_source - source, ord="fro") ** 2
    )
    assert joint_objective(state, X, Y, observed_source, omega) == pytest.approx(
        expected
    )

    gradients = euclidean_gradients(state, X, Y, observed_source, omega)
    direction = BISMARTDirection(
        U0=np.array(
            [
                [0.10, -0.20],
                [0.05, 0.15],
                [-0.10, 0.30],
            ]
        ),
        V0=np.array(
            [
                [-0.05, 0.20],
                [0.10, -0.10],
                [0.25, 0.05],
            ]
        ),
        A=np.array([[0.20], [-0.15]]),
        B=np.array([[-0.10], [0.25]]),
        H=np.array([[0.30]]),
        G_blocks=(np.array([[0.12]]), np.array([[-0.08]])),
    )

    # Central finite differences check all six displayed Euclidean gradient
    # blocks at once, including the diagonal-block projection of grad_G.
    epsilon = 1e-6
    plus = joint_objective(
        _shift_state(state, direction, epsilon), X, Y, observed_source, omega
    )
    minus = joint_objective(
        _shift_state(state, direction, -epsilon), X, Y, observed_source, omega
    )
    numerical_derivative = (plus - minus) / (2.0 * epsilon)
    analytic_derivative = product_inner_product(gradients, direction)
    assert numerical_derivative == pytest.approx(
        analytic_derivative, rel=2e-7, abs=2e-8
    )


def test_tangent_projection_and_polar_retraction_preserve_orthogonality() -> None:
    """Projected directions are tangent and their retractions are Stiefel."""

    state, X, Y, observed_source, omega = _tiny_problem()
    gradients = euclidean_gradients(state, X, Y, observed_source, omega)
    tangent = project_to_tangent(state, gradients)

    for factor, factor_direction in (
        (state.U0, tangent.U0),
        (state.V0, tangent.V0),
        (state.A[state.active_u], tangent.A[state.active_u]),
        (state.B[state.active_v], tangent.B[state.active_v]),
    ):
        tangent_identity = factor.T @ factor_direction + factor_direction.T @ factor
        np.testing.assert_allclose(tangent_identity, 0.0, atol=1e-12)

    retracted = retract_state(state, tangent, step_size=1e-2)
    assert stiefel_error(retracted.U0) < 1e-12
    assert stiefel_error(retracted.V0) < 1e-12
    assert stiefel_error(retracted.A[retracted.active_u]) < 1e-12
    assert stiefel_error(retracted.B[retracted.active_v]) < 1e-12


def test_quotient_direction_fails_with_actionable_pseudocode() -> None:
    """The missing quotient solve is explicit rather than silently approximated."""

    state, X, Y, observed_source, omega = _tiny_problem()
    with pytest.raises(NotImplementedError) as error:
        quotient_gauss_newton_direction(state, X, Y, observed_source, omega)

    message = str(error.value)
    assert "PSEUDOCODE" in message
    assert "vertical-space" in message
    assert "horizontal basis" in message
    assert "normal equations" in message
    assert "descent identity" in message


def test_path_certificate_rejects_step_larger_than_bar_eta() -> None:
    """The public checker enforces the domain used in the appendix proof."""

    state, X, Y, observed_source, omega = _tiny_problem()
    del X, Y, observed_source
    tangent = project_to_tangent(
        state,
        BISMARTDirection(
            U0=np.zeros_like(state.U0),
            V0=np.zeros_like(state.V0),
            A=np.zeros_like(state.A),
            B=np.zeros_like(state.B),
            H=np.zeros_like(state.H),
            G_blocks=tuple(np.zeros_like(core) for core in state.G_blocks),
        ),
    )
    thresholds = TrustRegionThresholds(
        radius=10.0,
        h_min=0.1,
        g_min=0.1,
        separation_min=0.0,
    )

    with pytest.raises(ValueError, match="eta <= bar_eta"):
        check_path_certificate(
            state,
            tangent,
            state,
            omega=omega,
            step_size=1.01,
            initial_step_size=1.0,
            thresholds=thresholds,
        )
