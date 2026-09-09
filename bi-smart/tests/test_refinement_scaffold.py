"""Focused checks for the executable part of the refinement scaffold."""

from __future__ import annotations

import numpy as np
import pytest

from bi_smart._quotient_geometry import (
    tangent_parseval_frame,
    vertical_gauge_generators,
)
from bi_smart.refinement import (
    BISMARTDirection,
    BISMARTState,
    TrustRegionThresholds,
    _JointJacobianOperator,
    check_path_certificate,
    euclidean_gradients,
    fitted_source,
    fitted_target,
    gauss_newton_inner_product,
    joint_objective,
    product_inner_product,
    project_to_tangent,
    quotient_gauss_newton_direction,
    retract_state,
    safeguarded_backtracking,
    solve_quotient_gauss_newton,
    source_differential,
    stiefel_error,
    target_differential,
)
from bi_smart.types import FailureReason, NumericalFailure


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


def _regular_one_block_problem() -> tuple[BISMARTState, np.ndarray, float]:
    """Return a well-conditioned quotient problem with a nontrivial gauge.

    A two-dimensional source block has independent left and right rotations,
    while target rank one contributes no continuous target-factor rotation.
    The product tangent therefore has dimension 13, the vertical space has
    dimension 2, and the quotient has the appendix dimension 11.
    """

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
        A=np.array([[0.8], [0.6]]),
        B=np.array([[-0.6], [0.8]]),
        H=np.array([[1.3]]),
        G_blocks=(np.array([[2.0, 0.2], [-0.1, 1.0]]),),
    )
    X = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.25],
            [-1.0, 0.5, 1.0],
            [0.4, -0.7, 0.3],
        ]
    )
    return state, X, 0.4


def _direction_linear_combination(
    directions: tuple[BISMARTDirection, ...], weights: np.ndarray
) -> BISMARTDirection:
    """Form a small test-only linear combination of product directions."""

    if len(directions) != len(weights):
        raise ValueError("directions and weights must have equal lengths")
    reference = directions[0]

    def combine(name: str) -> np.ndarray:
        return sum(
            (
                float(weight) * getattr(direction, name)
                for direction, weight in zip(directions, weights, strict=True)
            ),
            start=np.zeros_like(getattr(reference, name)),
        )

    return BISMARTDirection(
        U0=combine("U0"),
        V0=combine("V0"),
        A=combine("A"),
        B=combine("B"),
        H=combine("H"),
        G_blocks=tuple(
            sum(
                (
                    float(weight) * direction.G_blocks[block_index]
                    for direction, weight in zip(
                        directions, weights, strict=True
                    )
                ),
                start=np.zeros_like(reference.G_blocks[block_index]),
            )
            for block_index in range(len(reference.G_blocks))
        ),
    )


def _flatten_direction(direction: object) -> np.ndarray:
    """Vectorize all product-metric blocks in their displayed order."""

    return np.concatenate(
        tuple(
            np.asarray(getattr(direction, name), dtype=float).ravel(order="C")
            for name in ("U0", "V0", "A", "B", "H")
        )
        + tuple(
            np.asarray(block, dtype=float).ravel(order="C")
            for block in getattr(direction, "G_blocks")
        )
    )


def _direction_from_flat(
    state: BISMARTState, vector: np.ndarray
) -> BISMARTDirection:
    """Undo :func:`_flatten_direction` using one state's block shapes."""

    flat = np.asarray(vector, dtype=float)
    offset = 0

    def take(shape: tuple[int, int]) -> np.ndarray:
        nonlocal offset
        size = int(np.prod(shape))
        result = flat[offset : offset + size].reshape(shape, order="C")
        offset += size
        return result

    result = BISMARTDirection(
        U0=take(state.U0.shape),
        V0=take(state.V0.shape),
        A=take(state.A.shape),
        B=take(state.B.shape),
        H=take(state.H.shape),
        G_blocks=tuple(take(core.shape) for core in state.G_blocks),
    )
    if offset != flat.size:
        raise ValueError("flat direction has the wrong product-space dimension")
    return result


def _one_block_vertical_generators(
    state: BISMARTState,
) -> tuple[BISMARTDirection, BISMARTDirection]:
    """Differentiate the two source rotations without using solver internals."""

    # Scaling a generator does not change the vertical span.  This unnormalised
    # basis makes the differentiated gauge formulas especially transparent.
    rotation_generator = np.array([[0.0, -1.0], [1.0, 0.0]])
    zero_U0 = np.zeros_like(state.U0)
    zero_V0 = np.zeros_like(state.V0)
    zero_A = np.zeros_like(state.A)
    zero_B = np.zeros_like(state.B)
    zero_H = np.zeros_like(state.H)

    left = BISMARTDirection(
        U0=state.U0 @ rotation_generator,
        V0=zero_V0,
        A=-rotation_generator @ state.A,
        B=zero_B,
        H=zero_H,
        G_blocks=(-rotation_generator @ state.G_blocks[0],),
    )
    right = BISMARTDirection(
        U0=zero_U0,
        V0=state.V0 @ rotation_generator,
        A=zero_A,
        B=-rotation_generator @ state.B,
        H=zero_H,
        G_blocks=(state.G_blocks[0] @ rotation_generator,),
    )
    return left, right


def _assert_directions_allclose(
    actual: BISMARTDirection,
    expected: BISMARTDirection,
    *,
    atol: float = 2e-10,
    rtol: float = 1e-8,
) -> None:
    """Compare all six blocks of two directions with one tolerance policy."""

    for name in ("U0", "V0", "A", "B", "H"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), atol=atol, rtol=rtol
        )
    for actual_block, expected_block in zip(
        actual.G_blocks, expected.G_blocks, strict=True
    ):
        np.testing.assert_allclose(
            actual_block, expected_block, atol=atol, rtol=rtol
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

    # A retraction curve stays inside the now-validated public state manifold.
    # Its derivative at zero is the projected product tangent, so central
    # differences still check all six displayed Euclidean gradient blocks,
    # including the diagonal-block projection of grad_G.
    tangent = project_to_tangent(state, direction)
    negative_tangent = _direction_linear_combination(
        (tangent,), np.array([-1.0])
    )
    epsilon = 1e-6
    plus = joint_objective(
        retract_state(state, tangent, epsilon), X, Y, observed_source, omega
    )
    minus = joint_objective(
        retract_state(state, negative_tangent, epsilon),
        X,
        Y,
        observed_source,
        omega,
    )
    numerical_derivative = (plus - minus) / (2.0 * epsilon)
    analytic_derivative = product_inner_product(gradients, tangent)
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


def test_quotient_solver_recovers_a_known_horizontal_direction() -> None:
    """The dense solve recovers an independently constructed quotient step.

    We first project an arbitrary ambient direction onto the product tangent,
    then remove its two vertical source-rotation components using a direct
    two-by-two product-metric Gram solve.  The artificial observations differ
    from the current fitted pair by exactly the differential of that known
    horizontal direction, so Gauss--Newton must recover it without any
    nonlinear or finite-difference approximation.
    """

    state, X, omega = _regular_one_block_problem()
    ambient = BISMARTDirection(
        U0=np.array([[0.1, -0.2], [0.05, 0.15], [-0.1, 0.3]]),
        V0=np.array([[-0.05, 0.2], [0.1, -0.1], [0.25, 0.05]]),
        A=np.array([[0.2], [-0.15]]),
        B=np.array([[-0.1], [0.25]]),
        H=np.array([[0.3]]),
        G_blocks=(np.array([[0.12, 0.07], [-0.04, -0.08]]),),
    )
    tangent = project_to_tangent(state, ambient)
    vertical = _one_block_vertical_generators(state)
    vertical_gram = np.array(
        [
            [product_inner_product(left, right) for right in vertical]
            for left in vertical
        ]
    )
    vertical_coefficients = np.linalg.solve(
        vertical_gram,
        np.array(
            [product_inner_product(generator, tangent) for generator in vertical]
        ),
    )
    horizontal = _direction_linear_combination(
        (tangent, *vertical),
        np.concatenate(([1.0], -vertical_coefficients)),
    )
    expected = _direction_linear_combination((horizontal,), np.array([0.1]))

    # Both vertical products are zero by a construction independent of the
    # solver's Parseval frame and gauge-generator implementation.
    for generator in vertical:
        assert product_inner_product(horizontal, generator) == pytest.approx(
            0.0, abs=2e-15
        )

    Y = X @ (fitted_target(state) + target_differential(state, expected))
    observed_source = fitted_source(state) + source_differential(state, expected)
    result = solve_quotient_gauss_newton(
        state, X, Y, observed_source, omega
    )
    _assert_directions_allclose(result.direction, expected)
    _assert_directions_allclose(
        quotient_gauss_newton_direction(state, X, Y, observed_source, omega),
        expected,
    )

    # The dimension certificate matches the appendix formula
    # d=r0(p+q-r0)+r(ku+kv-r)=11 and records no excess Jacobian nullity.
    diagnostics = result.diagnostics
    assert diagnostics.tangent_dimension == 13
    assert diagnostics.vertical_dimension == 2
    assert diagnostics.quotient_dimension == 11
    assert diagnostics.jacobian_rank == 11
    assert diagnostics.frame_dimension == 21
    assert diagnostics.residual_dimension == 24
    assert diagnostics.smallest_retained_singular_value > diagnostics.rank_tolerance
    assert diagnostics.tangency_error < 2e-12
    assert diagnostics.horizontality_error < 2e-12
    assert diagnostics.normal_residual_norm < 2e-12
    assert diagnostics.descent_identity_error < 2e-12

    # Check the public differential and gradient APIs against the same
    # certificate, including the sign and 1/n scaling in the target term.
    np.testing.assert_allclose(
        X @ target_differential(state, result.direction),
        Y - X @ fitted_target(state),
        atol=2e-12,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        source_differential(state, result.direction),
        observed_source - fitted_source(state),
        atol=2e-12,
        rtol=1e-10,
    )
    gradient = euclidean_gradients(state, X, Y, observed_source, omega)
    derivative = product_inner_product(gradient, result.direction)
    gn_norm_squared = gauss_newton_inner_product(
        state, result.direction, result.direction, X, omega
    )
    assert gn_norm_squared > 0.0
    assert derivative == pytest.approx(-gn_norm_squared, rel=2e-10, abs=2e-12)
    assert diagnostics.gauss_newton_norm_squared == pytest.approx(
        gn_norm_squared, rel=2e-10, abs=2e-12
    )
    assert diagnostics.as_metadata()["gn_norm_squared"] == pytest.approx(
        gn_norm_squared, rel=2e-10, abs=2e-12
    )

    for factor, factor_direction in (
        (state.U0, result.direction.U0),
        (state.V0, result.direction.V0),
        (state.A, result.direction.A),
        (state.B, result.direction.B),
    ):
        np.testing.assert_allclose(
            factor.T @ factor_direction + factor_direction.T @ factor,
            0.0,
            atol=2e-12,
        )


def test_horizontal_coordinate_and_full_tangent_moore_penrose_solves_agree(
) -> None:
    """An explicit horizontal basis gives the same quotient GN direction.

    This is deliberately a second formulation of the linear problem.  The
    production solver applies Moore--Penrose to a redundant full-product
    tangent frame.  Here the test first extracts an orthonormal product
    tangent basis, removes the independently generated vertical span, and
    solves least squares in the resulting nonredundant horizontal basis.
    """

    state, X, omega = _regular_one_block_problem()
    target_perturbation = 0.03 * np.array(
        [
            [1.0, -1.0, 0.5],
            [0.0, 2.0, -1.0],
            [1.5, 0.0, 1.0],
            [-1.0, 0.5, 2.0],
            [0.5, -1.5, 0.0],
        ]
    )
    source_perturbation = 0.02 * np.array(
        [
            [0.0, 1.0, -1.0],
            [-2.0, 0.0, 0.5],
            [1.0, -0.5, 0.0],
        ]
    )
    Y = X @ fitted_target(state) + target_perturbation
    observed_source = fitted_source(state) + source_perturbation

    # The projected canonical frame is Parseval but redundant.  Its nonzero
    # left singular vectors are therefore an orthonormal product-tangent
    # basis in the ordinary flattened Frobenius metric.
    frame = tangent_parseval_frame(state, atol=1e-12, rtol=1e-10)
    frame_matrix = np.column_stack(
        tuple(_flatten_direction(element) for element in frame.iter_elements())
    )
    tangent_left, tangent_values, _ = np.linalg.svd(
        frame_matrix, full_matrices=False
    )
    tangent_rank = int(np.count_nonzero(tangent_values > 1e-12))
    assert tangent_rank == frame.tangent_dimension
    tangent_basis = tangent_left[:, :tangent_rank]

    vertical_matrix = np.column_stack(
        tuple(
            _flatten_direction(generator)
            for generator in vertical_gauge_generators(state)
        )
    )
    vertical_coordinates = tangent_basis.T @ vertical_matrix
    vertical_left, vertical_values, _ = np.linalg.svd(
        vertical_coordinates, full_matrices=True
    )
    vertical_rank = int(np.count_nonzero(vertical_values > 1e-12))
    assert vertical_rank == 2
    horizontal_basis = tangent_basis @ vertical_left[:, vertical_rank:]
    assert horizontal_basis.shape[1] == 11
    np.testing.assert_allclose(
        horizontal_basis.T @ horizontal_basis,
        np.eye(11),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        horizontal_basis.T @ vertical_matrix,
        0.0,
        atol=2e-12,
    )

    sample_scale = np.sqrt(X.shape[0])
    residual = np.concatenate(
        (
            ((X @ fitted_target(state) - Y) / sample_scale).ravel(order="C"),
            (
                np.sqrt(omega)
                * (fitted_source(state) - observed_source)
            ).ravel(order="C"),
        )
    )
    horizontal_jacobian = np.empty((residual.size, 11), dtype=float)
    for column in range(horizontal_basis.shape[1]):
        basis_direction = _direction_from_flat(
            state, horizontal_basis[:, column]
        )
        horizontal_jacobian[:, column] = np.concatenate(
            (
                (
                    X @ target_differential(state, basis_direction)
                    / sample_scale
                ).ravel(order="C"),
                (
                    np.sqrt(omega)
                    * source_differential(state, basis_direction)
                ).ravel(order="C"),
            )
        )

    # Full column rank makes the horizontal Moore--Penrose solve identical to
    # the positive-definite normal-equation formulation permitted by the
    # appendix.  Both must match the production full-tangent MP direction.
    horizontal_coefficients = -np.linalg.pinv(
        horizontal_jacobian, rcond=1e-12
    ) @ residual
    normal_coefficients = np.linalg.solve(
        horizontal_jacobian.T @ horizontal_jacobian,
        -(horizontal_jacobian.T @ residual),
    )
    np.testing.assert_allclose(
        horizontal_coefficients, normal_coefficients, atol=2e-11, rtol=2e-9
    )
    expected_direction = horizontal_basis @ horizontal_coefficients

    full_tangent = solve_quotient_gauss_newton(
        state, X, Y, observed_source, omega
    )
    np.testing.assert_allclose(
        _flatten_direction(full_tangent.direction),
        expected_direction,
        atol=3e-10,
        rtol=2e-8,
    )


def test_quotient_solver_rejects_information_singular_design() -> None:
    """A regular state with deficient quotient information fails locally."""

    state, X, omega = _regular_one_block_problem()
    zero_design = np.zeros_like(X)
    with pytest.raises(NumericalFailure) as error:
        solve_quotient_gauss_newton(
            state,
            zero_design,
            np.zeros((zero_design.shape[0], state.V0.shape[0])),
            fitted_source(state),
            omega,
        )

    assert error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert "numerical rank" in str(error.value)
    assert "expected 11" in str(error.value)


def test_quotient_solver_checks_dense_workspace_before_allocation() -> None:
    """The correctness-first backend turns unsafe sizes into local failures."""

    state, X, omega = _regular_one_block_problem()
    with pytest.raises(NumericalFailure) as error:
        solve_quotient_gauss_newton(
            state,
            X,
            X @ fitted_target(state),
            fitted_source(state),
            omega,
            max_dense_work_bytes=1,
        )

    assert error.value.reason is FailureReason.DENSE_SOLVER_LIMIT
    assert "max_dense_work_bytes=1" in str(error.value)


def test_matrix_free_jacobian_and_adjoint_satisfy_duality() -> None:
    """The streamed VJP is the exact Frobenius adjoint of the streamed JVP."""

    state, X, omega = _regular_one_block_problem()
    frame = tangent_parseval_frame(state)
    operator = _JointJacobianOperator(state, X, omega, frame)
    generator = np.random.default_rng(20260901)
    coefficients = generator.standard_normal(operator.columns)
    residual_coordinates = generator.standard_normal(operator.rows)

    forward_product = float(
        np.vdot(operator.matvec(coefficients), residual_coordinates).real
    )
    adjoint_product = float(
        np.vdot(coefficients, operator.rmatvec(residual_coordinates)).real
    )
    assert forward_product == pytest.approx(
        adjoint_product, rel=2e-13, abs=2e-13
    )


def test_quotient_solver_rejects_unknown_backend_directly() -> None:
    """The public solver validates its backend independently of controls."""

    state, X, omega = _regular_one_block_problem()
    with pytest.raises(ValueError, match="solver_backend"):
        solve_quotient_gauss_newton(
            state,
            X,
            X @ fitted_target(state),
            fitted_source(state),
            omega,
            solver_backend="not-a-backend",
        )


def test_auto_solver_uses_dense_backend_when_workspace_fits() -> None:
    """Auto retains the rank-revealing reference below its memory cap."""

    state, X, omega = _regular_one_block_problem()
    result = solve_quotient_gauss_newton(
        state,
        X,
        X @ fitted_target(state),
        fitted_source(state),
        omega,
        solver_backend="auto",
    )

    assert result.diagnostics.solver_backend == "dense"
    assert result.diagnostics.jacobian_rank == (
        result.diagnostics.quotient_dimension
    )


def test_matrix_free_iteration_limit_is_a_local_singular_system_failure() -> None:
    """An unconverged LSQR call fails only its current refinement solve."""

    state, X, omega = _regular_one_block_problem()
    target_perturbation = np.arange(
        X.shape[0] * state.V0.shape[0], dtype=float
    ).reshape(X.shape[0], state.V0.shape[0])
    source_perturbation = np.arange(
        state.U0.shape[0] * state.V0.shape[0], dtype=float
    ).reshape(state.U0.shape[0], state.V0.shape[0])
    Y = X @ fitted_target(state) + 0.01 * target_perturbation
    observed_source = fitted_source(state) + 0.005 * source_perturbation

    with pytest.raises(NumericalFailure) as error:
        solve_quotient_gauss_newton(
            state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="matrix_free",
            matrix_free_max_iterations=1,
        )

    assert error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert "did not meet its residual certificate in 1 iterations" in str(
        error.value
    )


def test_matrix_free_solver_matches_dense_and_auto_bypasses_dense_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LSQR reproduces the dense direction without allocating dense J/gauge arrays."""

    state, X, omega = _regular_one_block_problem()
    target_perturbation = np.arange(
        X.shape[0] * state.V0.shape[0], dtype=float
    ).reshape(X.shape[0], state.V0.shape[0])
    source_perturbation = np.arange(
        state.U0.shape[0] * state.V0.shape[0], dtype=float
    ).reshape(state.U0.shape[0], state.V0.shape[0])
    Y = X @ fitted_target(state) + 0.01 * target_perturbation
    observed_source = fitted_source(state) + 0.005 * source_perturbation

    dense = solve_quotient_gauss_newton(
        state, X, Y, observed_source, omega, solver_backend="dense"
    )
    matrix_free = solve_quotient_gauss_newton(
        state,
        X,
        Y,
        observed_source,
        omega,
        solver_backend="matrix_free",
        matrix_free_max_iterations=200,
    )
    _assert_directions_allclose(
        matrix_free.direction, dense.direction, atol=2e-9, rtol=2e-7
    )
    dense_norm_squared = gauss_newton_inner_product(
        state, dense.direction, dense.direction, X, omega
    )
    matrix_free_norm_squared = gauss_newton_inner_product(
        state, matrix_free.direction, matrix_free.direction, X, omega
    )
    assert dense.diagnostics.gauss_newton_norm_squared == pytest.approx(
        dense_norm_squared, rel=2e-10, abs=2e-12
    )
    assert matrix_free.diagnostics.gauss_newton_norm_squared == pytest.approx(
        matrix_free_norm_squared, rel=2e-10, abs=2e-12
    )
    assert matrix_free.diagnostics.gauss_newton_norm_squared == pytest.approx(
        dense.diagnostics.gauss_newton_norm_squared,
        rel=5e-7,
        abs=2e-10,
    )
    assert matrix_free.diagnostics.solver_backend == "matrix_free"
    assert matrix_free.diagnostics.rank_certificate == (
        "algebraic_injectivity_plus_necessary_numerical_screens"
        "_and_rhs_postchecks"
    )
    assert matrix_free.diagnostics.jacobian_rank is None
    assert matrix_free.diagnostics.rank_tolerance is None
    assert matrix_free.diagnostics.largest_singular_value is None
    assert matrix_free.diagnostics.smallest_retained_singular_value is None
    assert matrix_free.diagnostics.structural_quotient_rank == 11
    assert matrix_free.diagnostics.reduced_design_rank_tolerance is not None
    assert (
        matrix_free.diagnostics.reduced_design_smallest_singular_value
        > matrix_free.diagnostics.reduced_design_rank_tolerance
    )
    assert matrix_free.diagnostics.compact_rank_tolerance is not None
    assert (
        matrix_free.diagnostics.compact_smallest_singular_value
        > matrix_free.diagnostics.compact_rank_tolerance
    )

    # The matrix-free path must not accidentally call the dense helper that
    # materializes all gauge coordinates.  A one-byte dense cap forces auto
    # selection before the reference backend begins state/gauge allocation.
    import bi_smart.refinement as refinement_module

    def reject_dense_gauge_materialization(*args: object, **kwargs: object) -> object:
        raise AssertionError("dense vertical gauge materialization was called")

    monkeypatch.setattr(
        refinement_module,
        "vertical_gauge_generators",
        reject_dense_gauge_materialization,
    )
    automatic = solve_quotient_gauss_newton(
        state,
        X,
        Y,
        observed_source,
        omega,
        solver_backend="auto",
        max_dense_work_bytes=1,
        matrix_free_max_iterations=200,
    )
    assert automatic.diagnostics.solver_backend == "matrix_free"
    _assert_directions_allclose(
        automatic.direction, dense.direction, atol=2e-9, rtol=2e-7
    )


def test_matrix_free_structural_rank_check_rejects_zero_residual_singular_design(
) -> None:
    """A zero LSQR right-hand side cannot hide a deficient fitting design."""

    state, X, omega = _regular_one_block_problem()
    zero_design = np.zeros_like(X)
    with pytest.raises(NumericalFailure) as error:
        solve_quotient_gauss_newton(
            state,
            zero_design,
            np.zeros((zero_design.shape[0], state.V0.shape[0])),
            fitted_source(state),
            omega,
            solver_backend="matrix_free",
        )

    assert error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert "reduced fitting design" in str(error.value)
    assert "expected 2" in str(error.value)


def test_matrix_free_scale_screen_rejects_tiny_source_weight_at_exact_fit(
) -> None:
    """A zero RHS cannot conceal source quotient directions below the rank cutoff."""

    state, X, _ = _regular_one_block_problem()
    omega = 1e-20
    Y = X @ fitted_target(state)
    observed_source = fitted_source(state)

    # The dense reference sees the lost source directions directly in its
    # singular spectrum.  The matrix-free backend must reach the same branch
    # decision without using that zero statistical right-hand side as a rank
    # probe.
    with pytest.raises(NumericalFailure) as dense_error:
        solve_quotient_gauss_newton(
            state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="dense",
        )
    with pytest.raises(NumericalFailure) as matrix_free_error:
        solve_quotient_gauss_newton(
            state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="matrix_free",
        )

    assert dense_error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert matrix_free_error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert "compact matrix-free Jacobian restriction" in str(
        matrix_free_error.value
    )


def test_matrix_free_compact_screen_rejects_weak_target_core_at_exact_fit(
) -> None:
    """The compact restriction catches target directions scaled away by H."""

    state, X, omega = _regular_one_block_problem()
    weak_state = BISMARTState(
        U0=state.U0,
        V0=state.V0,
        A=state.A,
        B=state.B,
        H=np.array([[1e-10]]),
        G_blocks=state.G_blocks,
        active_u=state.active_u,
        active_v=state.active_v,
    )
    Y = X @ fitted_target(weak_state)
    observed_source = fitted_source(weak_state)

    with pytest.raises(NumericalFailure) as dense_error:
        solve_quotient_gauss_newton(
            weak_state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="dense",
        )
    with pytest.raises(NumericalFailure) as matrix_free_error:
        solve_quotient_gauss_newton(
            weak_state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="matrix_free",
        )

    assert dense_error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert matrix_free_error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert "compact matrix-free Jacobian restriction" in str(
        matrix_free_error.value
    )


def test_matrix_free_compact_screen_does_not_reject_small_regular_core(
) -> None:
    """The necessary compact screen avoids the former ad-hoc false rejection."""

    state, X, omega = _regular_one_block_problem()
    scaled_state = BISMARTState(
        U0=state.U0,
        V0=state.V0,
        A=state.A,
        B=state.B,
        H=state.H,
        G_blocks=(5e-10 * state.G_blocks[0],),
        active_u=state.active_u,
        active_v=state.active_v,
    )
    Y = X @ fitted_target(scaled_state)
    observed_source = fitted_source(scaled_state)

    dense = solve_quotient_gauss_newton(
        scaled_state,
        X,
        Y,
        observed_source,
        omega,
        solver_backend="dense",
    )
    matrix_free = solve_quotient_gauss_newton(
        scaled_state,
        X,
        Y,
        observed_source,
        omega,
        solver_backend="matrix_free",
    )
    assert dense.diagnostics.jacobian_rank == dense.diagnostics.quotient_dimension
    assert matrix_free.diagnostics.compact_jacobian_rank == (
        matrix_free.diagnostics.compact_quotient_dimension
    )
    _assert_directions_allclose(matrix_free.direction, dense.direction)


def test_matrix_free_normal_direction_guard_rejects_numerically_weak_core(
) -> None:
    """Weak source-normal directions are checked outside the compact restriction."""

    state, X, omega = _regular_one_block_problem()
    weak_state = BISMARTState(
        U0=state.U0,
        V0=state.V0,
        A=state.A,
        B=state.B,
        H=state.H,
        G_blocks=(5e-11 * state.G_blocks[0],),
        active_u=state.active_u,
        active_v=state.active_v,
    )
    Y = X @ fitted_target(weak_state)
    observed_source = fitted_source(weak_state)

    with pytest.raises(NumericalFailure) as dense_error:
        solve_quotient_gauss_newton(
            weak_state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="dense",
        )
    with pytest.raises(NumericalFailure) as matrix_free_error:
        solve_quotient_gauss_newton(
            weak_state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="matrix_free",
        )

    assert dense_error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert matrix_free_error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert "full dense Jacobian cannot satisfy its numerical-rank policy" in str(
        matrix_free_error.value
    )


def test_matrix_free_checks_compact_rank_workspace_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The r0-only rank certificate has its own pre-allocation memory guard."""

    import bi_smart.refinement as refinement_module

    state, X, omega = _regular_one_block_problem()
    monkeypatch.setattr(refinement_module, "DEFAULT_MAX_COMPACT_WORK_BYTES", 1)
    with pytest.raises(NumericalFailure) as error:
        solve_quotient_gauss_newton(
            state,
            X,
            X @ fitted_target(state),
            fitted_source(state),
            omega,
            solver_backend="matrix_free",
        )

    assert error.value.reason is FailureReason.DENSE_SOLVER_LIMIT
    assert "compact quotient rank certificate" in str(error.value)
    assert "polynomial in r0" in str(error.value)


def test_matrix_free_external_guard_handles_equal_target_source_ranks(
) -> None:
    """U/V-normal modes remain screened when null(A.T) and null(B.T) are empty."""

    state = BISMARTState(
        U0=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        V0=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        A=np.eye(2),
        B=np.eye(2),
        H=1e-10 * np.array([[1.2, 0.2], [-0.1, 0.8]]),
        G_blocks=(1e-10 * np.array([[2.0, 0.3], [-0.2, 1.0]]),),
    )
    X = np.array(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, -0.3],
            [1.0, 1.0, 0.4],
            [-1.0, 0.5, 0.1],
            [0.3, -0.7, 1.0],
        ]
    )
    omega = 0.5
    Y = X @ fitted_target(state)
    observed_source = fitted_source(state)

    with pytest.raises(NumericalFailure) as dense_error:
        solve_quotient_gauss_newton(
            state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="dense",
        )
    with pytest.raises(NumericalFailure) as matrix_free_error:
        solve_quotient_gauss_newton(
            state,
            X,
            Y,
            observed_source,
            omega,
            solver_backend="matrix_free",
        )

    assert dense_error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert matrix_free_error.value.reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
    assert "full dense Jacobian cannot satisfy its numerical-rank policy" in str(
        matrix_free_error.value
    )


def test_quotient_solver_handles_rank_one_full_source_boundary() -> None:
    """The edge case r=r0=1 has no rotations but a four-dimensional quotient."""

    state = BISMARTState(
        U0=np.array([[1.0], [0.0]]),
        V0=np.array([[1.0], [0.0]]),
        A=np.ones((1, 1)),
        B=np.ones((1, 1)),
        H=np.array([[1.2]]),
        G_blocks=(np.array([[0.8]]),),
    )
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    result = solve_quotient_gauss_newton(
        state,
        X,
        X @ fitted_target(state),
        fitted_source(state),
        0.5,
    )

    assert result.diagnostics.tangent_dimension == 4
    assert result.diagnostics.vertical_dimension == 0
    assert result.diagnostics.quotient_dimension == 4
    assert result.diagnostics.jacobian_rank == 4
    assert product_inner_product(result.direction, result.direction) == 0.0


def test_quotient_solver_handles_nontrivial_equal_target_source_ranks() -> None:
    """The r=r0>1 boundary retains both source and target rotation gauges."""

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
        A=np.eye(2),
        B=np.eye(2),
        H=np.array([[1.2, 0.2], [-0.1, 0.8]]),
        G_blocks=(np.array([[2.0, 0.3], [-0.2, 1.0]]),),
    )
    X = np.array(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, -0.3],
            [1.0, 1.0, 0.4],
            [-1.0, 0.5, 0.1],
            [0.3, -0.7, 1.0],
        ]
    )
    result = solve_quotient_gauss_newton(
        state,
        X,
        X @ fitted_target(state),
        fitted_source(state),
        0.5,
    )

    # Product dimension 16 minus four independent left/right source/target
    # rotation gauges gives the 12-dimensional quotient formula.
    assert result.diagnostics.tangent_dimension == 16
    assert result.diagnostics.vertical_dimension == 4
    assert result.diagnostics.quotient_dimension == 12
    assert result.diagnostics.jacobian_rank == 12
    assert product_inner_product(result.direction, result.direction) == 0.0


def test_quotient_solver_handles_rectangular_fitted_matrices() -> None:
    """Different predictor/response dimensions use the same quotient formula."""

    state, X, omega = _regular_one_block_problem()
    rectangular = BISMARTState(
        U0=state.U0,
        V0=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        A=state.A,
        B=state.B,
        H=state.H,
        G_blocks=state.G_blocks,
    )
    result = solve_quotient_gauss_newton(
        rectangular,
        X,
        X @ fitted_target(rectangular),
        fitted_source(rectangular),
        omega,
    )

    # d=2*(3+4-2)+1*(2+2-1)=13.
    assert fitted_target(rectangular).shape == (3, 4)
    assert result.diagnostics.quotient_dimension == 13
    assert result.diagnostics.jacobian_rank == 13


def test_public_state_rejects_partial_source_block_support() -> None:
    """A numerically valid row mask is still invalid unless it selects blocks."""

    regular, X, omega = _regular_one_block_problem()
    del X, omega
    with pytest.raises(ValueError, match="partially selects source block"):
        BISMARTState(
            U0=regular.U0,
            V0=regular.V0,
            A=np.array([[1.0], [0.0]]),
            B=regular.B,
            H=regular.H,
            G_blocks=regular.G_blocks,
            active_u=np.array([True, False]),
            active_v=np.array([True, True]),
        )


def test_quotient_iterate_path_is_equivariant_under_all_block_gauges() -> None:
    """Directions and a multi-step retracted path commute with all gauges."""

    # The size-two source block exercises independent left/right block
    # rotations, while target rank two exercises both target-factor rotations.
    # The singleton source block also receives an orthogonal sign change.
    state = BISMARTState(
        U0=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        V0=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        A=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        B=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        H=np.array([[1.4, 0.2], [-0.3, 0.9]]),
        G_blocks=(
            np.array([[2.0, 0.3], [-0.1, 1.0]]),
            np.array([[0.4]]),
        ),
    )
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, -0.3],
            [0.0, 0.0, 1.0, 0.4],
            [1.0, 1.0, 0.0, 0.5],
            [0.0, 1.0, 1.0, -0.2],
            [1.0, 0.0, 1.0, 0.1],
        ]
    )
    target_perturbation = 0.02 * np.array(
        [
            [1.0, -1.0, 0.0, 2.0],
            [0.0, 1.0, -2.0, 1.0],
            [2.0, 0.0, 1.0, -1.0],
            [-1.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, -1.0, 0.0],
            [0.0, -2.0, 1.0, 1.0],
        ]
    )
    source_perturbation = 0.03 * np.array(
        [
            [0.0, 1.0, -1.0, 2.0],
            [-1.0, 0.0, 2.0, 1.0],
            [1.0, -2.0, 0.0, -1.0],
            [2.0, 1.0, -1.0, 0.0],
        ]
    )
    Y = X @ fitted_target(state) + target_perturbation
    observed_source = fitted_source(state) + source_perturbation
    omega = 0.7

    def rotation(angle: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

    left_source_block = rotation(0.37)
    right_source_block = rotation(-0.29)
    left_target = rotation(0.21)
    right_target = rotation(-0.43)
    left_source = np.zeros((3, 3))
    right_source = np.zeros((3, 3))
    left_source[:2, :2] = left_source_block
    right_source[:2, :2] = right_source_block
    left_source[2, 2] = -1.0
    right_source[2, 2] = 1.0

    def gauge_state(current: BISMARTState) -> BISMARTState:
        return BISMARTState(
            U0=current.U0 @ left_source,
            V0=current.V0 @ right_source,
            A=left_source.T @ current.A @ left_target,
            B=right_source.T @ current.B @ right_target,
            H=left_target.T @ current.H @ right_target,
            G_blocks=(
                left_source_block.T
                @ current.G_blocks[0]
                @ right_source_block,
                -current.G_blocks[1],
            ),
        )

    def gauge_direction(current: BISMARTDirection) -> BISMARTDirection:
        return BISMARTDirection(
            U0=current.U0 @ left_source,
            V0=current.V0 @ right_source,
            A=left_source.T @ current.A @ left_target,
            B=right_source.T @ current.B @ right_target,
            H=left_target.T @ current.H @ right_target,
            G_blocks=(
                left_source_block.T
                @ current.G_blocks[0]
                @ right_source_block,
                -current.G_blocks[1],
            ),
        )

    def assert_states_allclose(
        actual: BISMARTState, expected: BISMARTState
    ) -> None:
        for name in ("U0", "V0", "A", "B", "H"):
            np.testing.assert_allclose(
                getattr(actual, name),
                getattr(expected, name),
                atol=3e-10,
                rtol=2e-8,
            )
        for actual_block, expected_block in zip(
            actual.G_blocks, expected.G_blocks, strict=True
        ):
            np.testing.assert_allclose(
                actual_block,
                expected_block,
                atol=3e-10,
                rtol=2e-8,
            )

    canonical_state = state
    rotated_state = gauge_state(state)
    canonical_initial = canonical_state
    rotated_initial = rotated_state
    thresholds = TrustRegionThresholds(
        radius=1e6,
        h_min=0.1,
        g_min=0.1,
        separation_min=0.0,
    )
    first_canonical = None
    first_rotated = None
    for path_index in range(3):
        np.testing.assert_allclose(
            fitted_target(rotated_state),
            fitted_target(canonical_state),
            atol=3e-10,
            rtol=2e-8,
        )
        np.testing.assert_allclose(
            fitted_source(rotated_state),
            fitted_source(canonical_state),
            atol=3e-10,
            rtol=2e-8,
        )

        canonical = solve_quotient_gauss_newton(
            canonical_state, X, Y, observed_source, omega
        )
        rotated = solve_quotient_gauss_newton(
            rotated_state, X, Y, observed_source, omega
        )
        if path_index == 0:
            first_canonical = canonical
            first_rotated = rotated
        _assert_directions_allclose(
            rotated.direction,
            gauge_direction(canonical.direction),
            atol=3e-10,
            rtol=2e-8,
        )

        # Run the actual finite-path and Armijo rule on both representatives.
        # Re-solving after each accepted retraction checks a complete
        # safeguarded iterative path, not only a tangent/retraction primitive.
        canonical_step = safeguarded_backtracking(
            canonical_state,
            canonical.direction,
            canonical_initial,
            X,
            Y,
            observed_source,
            omega=omega,
            thresholds=thresholds,
            initial_step_size=0.05,
            contraction=0.5,
            armijo_constant=1e-4,
            max_trials=4,
        )
        rotated_step = safeguarded_backtracking(
            rotated_state,
            rotated.direction,
            rotated_initial,
            X,
            Y,
            observed_source,
            omega=omega,
            thresholds=thresholds,
            initial_step_size=0.05,
            contraction=0.5,
            armijo_constant=1e-4,
            max_trials=4,
        )
        assert canonical_step.accepted and canonical_step.state is not None
        assert rotated_step.accepted and rotated_step.state is not None
        assert rotated_step.trials == canonical_step.trials
        assert rotated_step.step_size == pytest.approx(canonical_step.step_size)
        next_canonical = canonical_step.state
        next_rotated = rotated_step.state
        assert_states_allclose(next_rotated, gauge_state(next_canonical))
        canonical_state = next_canonical
        rotated_state = next_rotated

    assert first_canonical is not None
    assert first_rotated is not None
    assert first_canonical.diagnostics.vertical_dimension == 4
    assert first_canonical.diagnostics.quotient_dimension == 23
    assert first_canonical.diagnostics.jacobian_rank == 23
    assert first_rotated.diagnostics.jacobian_rank == 23


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


def test_backtracking_accepts_the_first_safe_armijo_trial() -> None:
    """An already optimal state accepts its first valid zero-direction step."""

    state, X, _, _, omega = _tiny_problem()
    Y = X @ fitted_target(state)
    observed_source = fitted_source(state)
    zero = BISMARTDirection(
        U0=np.zeros_like(state.U0),
        V0=np.zeros_like(state.V0),
        A=np.zeros_like(state.A),
        B=np.zeros_like(state.B),
        H=np.zeros_like(state.H),
        G_blocks=tuple(np.zeros_like(core) for core in state.G_blocks),
    )
    thresholds = TrustRegionThresholds(
        radius=1.0,
        h_min=0.5,
        g_min=0.25,
        separation_min=0.1,
    )

    result = safeguarded_backtracking(
        state,
        zero,
        state,
        X,
        Y,
        observed_source,
        omega=omega,
        thresholds=thresholds,
        initial_step_size=0.8,
        contraction=0.5,
        armijo_constant=1e-4,
        max_trials=3,
    )

    assert result.accepted
    assert result.trials == 1
    assert result.step_size == pytest.approx(0.8)
    assert result.path_certificate is not None
    assert result.path_certificate.safe
    assert result.armijo_check is not None
    assert result.armijo_check.accepted
    assert result.state is not None
    # Retr(state, 0) is preserved exactly instead of round-tripping every
    # Stiefel factor through an SVD and risking a false zero-objective increase.
    assert result.state is state
    np.testing.assert_allclose(fitted_target(result.state), fitted_target(state))
    np.testing.assert_allclose(fitted_source(result.state), fitted_source(state))


def test_backtracking_rejects_unsafe_initial_path_then_accepts_next() -> None:
    """Path safety is checked before Armijo and skips an oversized first step."""

    state, X, omega = _regular_one_block_problem()
    target_perturbation = 0.02 * np.array(
        [
            [1.0, -1.0, 0.5],
            [0.0, 2.0, -1.0],
            [1.5, 0.0, 1.0],
            [-1.0, 0.5, 2.0],
            [0.5, -1.5, 0.0],
        ]
    )
    source_perturbation = 0.01 * np.array(
        [
            [0.0, 1.0, -1.0],
            [-2.0, 0.0, 0.5],
            [1.0, -0.5, 0.0],
        ]
    )
    Y = X @ fitted_target(state) + target_perturbation
    observed_source = fitted_source(state) + source_perturbation
    direction = solve_quotient_gauss_newton(
        state, X, Y, observed_source, omega
    ).direction
    direction_norm = np.sqrt(product_inner_product(direction, direction))
    assert 0.0 < direction_norm < 1e3

    # Normalize the first trial to eta*||xi||=1, which violates the appendix's
    # strict retraction-path bound.  Choose beta so the next trial is exactly
    # eta=1e-4: safely inside the path and small enough for GN Armijo decrease.
    initial_step_size = 1.0 / direction_norm
    contraction = 1e-4 * direction_norm
    thresholds = TrustRegionThresholds(
        radius=1e6,
        h_min=0.1,
        g_min=0.1,
        separation_min=0.0,
    )
    first_certificate = check_path_certificate(
        state,
        direction,
        state,
        omega=omega,
        step_size=initial_step_size,
        initial_step_size=initial_step_size,
        thresholds=thresholds,
    )
    second_certificate = check_path_certificate(
        state,
        direction,
        state,
        omega=omega,
        step_size=1e-4,
        initial_step_size=initial_step_size,
        thresholds=thresholds,
    )
    assert not first_certificate.safe
    assert "retraction_path_bound" in first_certificate.failed_checks
    assert second_certificate.safe

    result = safeguarded_backtracking(
        state,
        direction,
        state,
        X,
        Y,
        observed_source,
        omega=omega,
        thresholds=thresholds,
        initial_step_size=initial_step_size,
        contraction=contraction,
        armijo_constant=1e-4,
        max_trials=3,
    )

    assert result.accepted
    assert result.trials == 2
    assert result.step_size == pytest.approx(1e-4)
    assert result.path_certificate is not None
    assert result.path_certificate.safe
    assert result.armijo_check is not None
    assert result.armijo_check.accepted
