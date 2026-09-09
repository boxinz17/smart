"""Tests for the full-sample restricted-RRR plus GN pilot driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SIMULATION_DIR = Path(__file__).resolve().parents[1]
BASE_PATH = SIMULATION_DIR / "run_restricted_rrr.py"
RUNNER_PATH = SIMULATION_DIR / "run_restricted_rrr_gauss_newton.py"

# The GN runner deliberately reuses the checked grid/seed/write utilities from
# the restricted-RRR runner.  Register that script under its direct-execution
# module name before loading the new driver in isolation.
if "run_restricted_rrr" not in sys.modules:
    base_spec = importlib.util.spec_from_file_location("run_restricted_rrr", BASE_PATH)
    assert base_spec is not None and base_spec.loader is not None
    base = importlib.util.module_from_spec(base_spec)
    sys.modules[base_spec.name] = base
    base_spec.loader.exec_module(base)

SPEC = importlib.util.spec_from_file_location(
    "run_restricted_rrr_gauss_newton", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _setting(*, sigma0: float = 0.01) -> object:
    return runner.SimulationSetting(
        n=6,
        p=3,
        q=2,
        sigma0=sigma0,
        target_rank=1,
        source_rank=2,
        suffix=f"sigma0={sigma0}",
    )


def _data(**kwargs: object) -> dict[str, np.ndarray]:
    n, p, q = int(kwargs["n"]), int(kwargs["p"]), int(kwargs["q"])
    return {
        "X": np.arange(n * p, dtype=float).reshape(n, p),
        "Y": np.zeros((n, q)),
        "C_star": np.zeros((p, q)),
        "C0": np.eye(p, q),
    }


def _successful_initialization(coefficient: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        successful=True,
        coefficient=coefficient,
        state=object(),
        candidate=SimpleNamespace(
            metadata={"rrr_cutoff_gap": 1.0},
            failure_reason=None,
            message=None,
        ),
    )


def test_defaults_and_result_names_are_separate_from_restricted_rrr() -> None:
    assert runner.DEFAULT_SOURCE_WEIGHTS == (0.01, 0.1, 1.0, 10.0, 100.0)
    controls = runner.GaussNewtonPilotControls()
    assert controls.iteration_cap == 10
    assert controls.solver_backend == "matrix_free"
    assert controls.trust_radius_multiplier == 1.0
    assert controls.convergence_stopping is True
    assert controls.stopping_atol == 1e-12
    assert controls.stopping_rtol == 1e-10

    path = runner.result_path(
        Path("result"),
        model="model1",
        experiment="exp1",
        setting=_setting(),
        seed_id=2,
        omega=10.0,
    )
    assert path.name == (
        "RestrictedRRRGN_result_model1_exp1_sigma0=0.01_"
        "omega=10_rd_seed_id=2.pkl"
    )
    assert not path.name.startswith("RestrictedRRR_result_")

    oracle_path = runner.result_path(
        Path("result"),
        model="model1",
        experiment="exp4",
        setting=_setting(sigma0=0.0),
        seed_id=2,
        omega=None,
        omega_rule="gaussian-noise",
    )
    assert "omega=infinite_omega-rule=gaussian-noise" in oracle_path.name


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iteration_cap": 0}, "iteration_cap"),
        ({"trust_radius_multiplier": 0.0}, "trust_radius_multiplier"),
        ({"solver_backend": "mystery"}, "solver_backend"),
        ({"convergence_stopping": 1}, "convergence_stopping"),
        ({"stopping_atol": -1.0}, "stopping_atol"),
        ({"stopping_atol": True}, "stopping_atol"),
        ({"stopping_rtol": np.nan}, "stopping_rtol"),
    ],
)
def test_controls_reject_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        runner.GaussNewtonPilotControls(**kwargs)


def test_decrement_stopping_precedes_backtracking_and_uses_half_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negligible model reduction stops at the current accepted state."""

    monkeypatch.syspath_prepend(str(SIMULATION_DIR.parent / "bi-smart"))
    import bi_smart.refinement as refinement_module

    initial = np.ones((3, 2))
    fake_state = object()
    diagnostics = SimpleNamespace(
        gauss_newton_norm_squared=2e-9,
        as_metadata=lambda: {"gn_norm_squared": 2e-9},
    )
    monkeypatch.setattr(
        refinement_module, "fitted_source", lambda _state: np.ones((3, 2))
    )
    monkeypatch.setattr(
        refinement_module, "joint_objective", lambda *_args, **_kwargs: 10.0
    )
    monkeypatch.setattr(
        refinement_module, "default_core_thresholds", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        refinement_module,
        "solve_quotient_gauss_newton",
        lambda *_args, **_kwargs: SimpleNamespace(
            direction=object(), diagnostics=diagnostics
        ),
    )

    def must_not_backtrack(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a converged direction must not enter line search")

    monkeypatch.setattr(
        refinement_module, "safeguarded_backtracking", must_not_backtrack
    )

    controls = runner.GaussNewtonPilotControls()
    path = runner._run_gauss_newton_path(
        initial_state=fake_state,
        initial_coefficient=initial,
        X=np.ones((6, 3)),
        Y=np.zeros((6, 2)),
        observed_source=np.ones((3, 2)),
        omega=1.0,
        controls=controls,
    )

    # 0.5 * 2e-9 = 1e-9 <= 1e-12 + 1e-10 * 10 = 1.001e-9.
    # Comparing the raw squared norm would fail, so this locks in the
    # quadratic-model factor one half.
    assert path["success"] is True
    assert path["converged"] is True
    assert path["termination_reason"] == "gauss_newton_decrement"
    assert path["completed_iterations"] == 0
    assert len(path["iterates"]) == 1
    check = path["last_convergence_check"]
    assert check["state_iteration"] == 0
    assert check["predicted_reduction"] == pytest.approx(1e-9)
    assert check["threshold"] == pytest.approx(1.001e-9)
    assert check["criterion_met"] is True
    assert check["stopping_enabled"] is True


def test_decrement_stopping_can_be_disabled() -> None:
    controls = runner.GaussNewtonPilotControls(convergence_stopping=False)
    converged, reduction, threshold = runner._gauss_newton_stopping_test(
        objective=10.0,
        gauss_newton_norm_squared=0.0,
        controls=controls,
    )
    assert converged is False
    assert reduction == 0.0
    assert threshold == pytest.approx(1.001e-9)


def test_successful_path_uses_endpoint_without_oracle_selection(tmp_path: Path) -> None:
    setting = _setting()
    initial = np.ones((setting.p, setting.q))
    endpoint = np.full((setting.p, setting.q), 2.0)
    calls: dict[str, object] = {}

    def initialize(X: np.ndarray, Y: np.ndarray, C0: np.ndarray, **kwargs: object):
        calls["shapes"] = (X.shape, Y.shape, C0.shape)
        calls["initial_kwargs"] = kwargs
        return _successful_initialization(initial)

    def refine(**kwargs: object) -> dict[str, object]:
        calls["refine"] = kwargs
        return {
            "omega": kwargs["omega"],
            "status": "successful",
            "success": True,
            "failure_reason": None,
            "failure_message": None,
            "converged": False,
            "termination_reason": "iteration_cap",
            "last_convergence_check": None,
            "trust_radius": 1.0,
            "completed_iterations": 1,
            "requested_iterations": 1,
            "iterates": [
                {"iteration": 0, "C_hat": initial, "objective": 2.0},
                {"iteration": 1, "C_hat": endpoint, "objective": 1.0},
            ],
        }

    controls = runner.GaussNewtonPilotControls(iteration_cap=1)
    destination = tmp_path / "path.pkl"
    outcome, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp1",
        seed_id=0,
        random_seed=13,
        omega=10.0,
        controls=controls,
        destination=destination,
        generate_data_fn=_data,
        restricted_rrr_fn=initialize,
        refinement_path_fn=refine,
    )

    assert outcome == "written"
    assert result is not None and result["status"] == "successful"
    assert result["refinement_success"] is True
    assert result["refinement_converged"] is False
    assert result["refinement_termination_reason"] == "iteration_cap"
    assert result["used_t0_fallback"] is False
    assert calls["shapes"] == ((6, 3), (6, 2), (3, 2))
    assert calls["initial_kwargs"] == {
        "target_rank": 1,
        "source_rank": 2,
        "atol": 1e-12,
        "rtol": 1e-10,
    }
    # C_true is zero: t=0 has lower error than t=1.  Reporting t=1 proves the
    # driver follows the requested path endpoint and does not oracle-select.
    assert result["restricted_rrr_avg_err"] == 1.0
    assert result["avg_err"] == 2.0
    np.testing.assert_array_equal(result["C_hat"], endpoint)
    assert [item["avg_err"] for item in result["path"]["iterates"]] == [1.0, 2.0]
    assert 0 <= result["initialization_time_sec"] <= result["elapsed_time_sec"]
    assert 0 <= result["refinement_time_sec"] <= result["elapsed_time_sec"]

    with destination.open("rb") as handle:
        saved = pickle.load(handle)
    np.testing.assert_array_equal(saved["path"]["iterates"][0]["C_hat"], initial)


def test_successful_early_convergence_publishes_last_accepted_iterate(
    tmp_path: Path,
) -> None:
    setting = _setting()
    initial = np.ones((setting.p, setting.q))
    endpoint = np.full((setting.p, setting.q), 2.0)

    def converged_path(**kwargs: object) -> dict[str, object]:
        return {
            "omega": kwargs["omega"],
            "status": "successful",
            "success": True,
            "failure_reason": None,
            "failure_message": None,
            "converged": True,
            "termination_reason": "gauss_newton_decrement",
            "last_convergence_check": {
                "state_iteration": 1,
                "objective": 1.0,
                "gauss_newton_norm_squared": 1e-12,
                "predicted_reduction": 5e-13,
                "threshold": 1.1e-12,
                "stopping_enabled": True,
                "criterion_met": True,
            },
            "trust_radius": 1.0,
            "completed_iterations": 1,
            "requested_iterations": 10,
            "iterates": [
                {"iteration": 0, "C_hat": initial, "objective": 2.0},
                {"iteration": 1, "C_hat": endpoint, "objective": 1.0},
            ],
        }

    _, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp1",
        seed_id=0,
        random_seed=13,
        omega=10.0,
        controls=runner.GaussNewtonPilotControls(iteration_cap=10),
        destination=tmp_path / "converged.pkl",
        generate_data_fn=_data,
        restricted_rrr_fn=lambda *_args, **_kwargs: _successful_initialization(
            initial
        ),
        refinement_path_fn=converged_path,
    )

    assert result is not None
    assert result["status"] == "successful"
    assert result["refinement_success"] is True
    assert result["refinement_converged"] is True
    assert result["refinement_termination_reason"] == "gauss_newton_decrement"
    assert result["used_t0_fallback"] is False
    np.testing.assert_array_equal(result["C_hat"], endpoint)
    assert len(result["path"]["iterates"]) == 2


def test_failed_omega_path_publishes_t0_fallback_and_retains_diagnostics(
    tmp_path: Path,
) -> None:
    setting = _setting()
    initial = np.ones((setting.p, setting.q))

    def failed_path(**kwargs: object) -> dict[str, object]:
        return {
            "omega": kwargs["omega"],
            "status": "failed",
            "success": False,
            "failure_reason": "line_search_failed",
            "failure_message": "no safe step",
            "trust_radius": 1.0,
            "completed_iterations": 0,
            "requested_iterations": 2,
            "iterates": [{"iteration": 0, "C_hat": initial, "objective": 2.0}],
        }

    _, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp1",
        seed_id=0,
        random_seed=13,
        omega=0.1,
        controls=runner.GaussNewtonPilotControls(iteration_cap=2),
        destination=tmp_path / "failed.pkl",
        generate_data_fn=_data,
        restricted_rrr_fn=lambda *_args, **_kwargs: _successful_initialization(initial),
        refinement_path_fn=failed_path,
    )
    assert result is not None
    assert result["status"] == "successful_t0_fallback"
    assert result["success"] is True
    assert result["refinement_success"] is False
    assert result["refinement_converged"] is False
    assert result["refinement_termination_reason"] == "failure"
    assert result["used_t0_fallback"] is True
    np.testing.assert_array_equal(result["C_hat"], initial)
    assert result["avg_err"] == 1.0
    assert result["failure_reason"] == "line_search_failed"
    assert result["restricted_rrr_avg_err"] == 1.0
    assert len(result["path"]["iterates"]) == 1


def test_resume_skips_only_identical_metadata_and_rejects_stale_controls(
    tmp_path: Path,
) -> None:
    setting = _setting()
    controls = runner.GaussNewtonPilotControls(iteration_cap=2)
    destination = tmp_path / "resume.pkl"
    stored = {
        "result_schema_version": 3,
        "method": "full_sample_restricted_rrr_gauss_newton",
        "model": "model1",
        "experiment": "exp1",
        "setting": runner.asdict(setting),
        "rd_seed_id": 0,
        "random_seed": 13,
        "omega": 10.0,
        "omega_rule": "explicit",
        "controls": runner.asdict(controls),
    }
    with destination.open("wb") as handle:
        pickle.dump(stored, handle)

    def must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a compatible resume must not regenerate or refit")

    outcome, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp1",
        seed_id=0,
        random_seed=13,
        omega=10.0,
        controls=controls,
        destination=destination,
        generate_data_fn=must_not_run,
        restricted_rrr_fn=must_not_run,
        refinement_path_fn=must_not_run,
    )
    assert outcome == "skipped" and result is None

    changed_controls = runner.GaussNewtonPilotControls(
        iteration_cap=2, stopping_rtol=1e-8
    )
    with pytest.raises(RuntimeError, match=r"controls.*--force"):
        runner.run_setting(
            setting=setting,
            model="model1",
            experiment="exp1",
            seed_id=0,
            random_seed=13,
            omega=10.0,
            controls=changed_controls,
            destination=destination,
            generate_data_fn=must_not_run,
            restricted_rrr_fn=must_not_run,
            refinement_path_fn=must_not_run,
        )

    with pytest.raises(RuntimeError, match=r"random_seed.*--force"):
        runner.run_setting(
            setting=setting,
            model="model1",
            experiment="exp1",
            seed_id=0,
            random_seed=99,
            omega=10.0,
            controls=controls,
            destination=destination,
            generate_data_fn=must_not_run,
            restricted_rrr_fn=must_not_run,
            refinement_path_fn=must_not_run,
        )


def test_zero_source_noise_oracle_rule_returns_t0_without_calling_gn(
    tmp_path: Path,
) -> None:
    setting = _setting(sigma0=0.0)
    initial = np.ones((setting.p, setting.q))

    def must_not_refine(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("infinite source weight must skip finite GN")

    _, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp4",
        seed_id=0,
        random_seed=13,
        omega=None,
        omega_rule="gaussian-noise",
        controls=runner.GaussNewtonPilotControls(),
        destination=tmp_path / "exact.pkl",
        generate_data_fn=_data,
        restricted_rrr_fn=lambda *_args, **_kwargs: _successful_initialization(initial),
        refinement_path_fn=must_not_refine,
    )
    assert result is not None and result["success"] is True
    assert result["status"] == "successful_t0_only"
    assert result["refinement_success"] is None
    assert result["refinement_converged"] is None
    assert result["refinement_termination_reason"] == "source_noise_is_zero"
    assert result["used_t0_fallback"] is False
    assert result["omega"] is None
    assert result["omega_is_simulation_oracle"] is True
    assert result["refinement_skipped_reason"] == "source_noise_is_zero"
    assert result["path"]["converged"] is False
    assert result["path"]["termination_reason"] == "source_noise_is_zero"
    assert len(result["path"]["iterates"]) == 1
    np.testing.assert_array_equal(result["C_hat"], initial)


def test_inapplicable_rank_is_local_and_never_calls_estimators(tmp_path: Path) -> None:
    setting = runner.SimulationSetting(
        n=6,
        p=3,
        q=2,
        sigma0=0.01,
        target_rank=2,
        source_rank=1,
        suffix="invalid",
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("inapplicable rank must not fit")

    _, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp3",
        seed_id=0,
        random_seed=13,
        omega=1.0,
        controls=runner.GaussNewtonPilotControls(),
        destination=tmp_path / "invalid.pkl",
        generate_data_fn=_data,
        restricted_rrr_fn=fail,
        refinement_path_fn=fail,
    )
    assert result is not None
    assert result["status"] == "inapplicable"
    assert result["path"] is None


def test_dry_run_supports_explicit_and_noise_calibrated_omega(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_initial_implementations",
        lambda: (_ for _ in ()).throw(AssertionError("dry run imported packages")),
    )
    assert runner.main(
        [
            "0",
            "0",
            "0",
            "--setting-index",
            "0",
            "--omega",
            "1",
            "--omega",
            "10",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    ) == 0
    explicit_output = capsys.readouterr().out
    assert "omega=1" in explicit_output and "omega=10" in explicit_output

    assert runner.main(
        [
            "0",
            "3",
            "0",
            "--setting-index",
            "0",
            "--omega-rule",
            "gaussian-noise",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    ) == 0
    oracle_output = capsys.readouterr().out
    assert "omega=infinite" in oracle_output
    assert "omega-rule=gaussian-noise" in oracle_output
    assert list(tmp_path.iterdir()) == []


def test_cli_exposes_convergence_controls() -> None:
    args = runner._parser().parse_args(
        [
            "0",
            "0",
            "0",
            "--stopping-atol",
            "1e-9",
            "--stopping-rtol",
            "1e-7",
            "--no-convergence-stopping",
        ]
    )
    assert args.stopping_atol == 1e-9
    assert args.stopping_rtol == 1e-7
    assert args.convergence_stopping is False


def test_unexpected_refinement_exception_propagates_without_file(
    tmp_path: Path,
) -> None:
    setting = _setting()
    destination = tmp_path / "must-not-exist.pkl"
    initial = np.ones((setting.p, setting.q))

    with pytest.raises(RuntimeError, match="implementation bug"):
        runner.run_setting(
            setting=setting,
            model="model1",
            experiment="exp1",
            seed_id=0,
            random_seed=13,
            omega=1.0,
            controls=runner.GaussNewtonPilotControls(),
            destination=destination,
            generate_data_fn=_data,
            restricted_rrr_fn=lambda *_args, **_kwargs: (
                _successful_initialization(initial)
            ),
            refinement_path_fn=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("implementation bug")
            ),
        )
    assert not destination.exists()
