"""Sparse-only simulation IO, data provenance, and failure accounting tests."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_sparse_smart.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_sparse_smart", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def tiny_setting(sigma0=.01):
    return runner.SimulationSetting(6, 3, 2, sigma0, 1, 2, "tiny")


def tiny_data(**kwargs):
    n, p, q = kwargs["n"], kwargs["p"], kwargs["q"]
    return dict(X=np.arange(n*p, dtype=float).reshape(n, p), Y=np.zeros((n, q)),
                C_star=np.zeros((p, q)), C0=np.eye(p, q) * np.arange(1., q + 1),
                U0_star=np.full((p, 2), 987.), V0_star=np.full((q, 2), 987.))


def fake_api(calls, *, successful=True, partial=True):
    def calibration(**kwargs):
        calls["calibration"] = kwargs
        return SimpleNamespace(**kwargs)

    def model(**kwargs):
        calls["configuration"] = kwargs
        estimator = SimpleNamespace(status_="completed_iterations" if successful else "line_search_failed",
            success_=successful, message_="fit message", n_iter_=3,
            history_=[dict(objective=np.float64(1.), loss=.5)],
            diagnostics_={"source_check_passed": np.bool_(False), "unknown": float("inf")})

        def fit(X, Y, *, source):
            calls["fit"] = (X.copy(), Y.copy(), source)
            if partial:
                p, q = X.shape[1], Y.shape[1]
                estimator.coefficient_ = np.full((p, q), .25)
                estimator.initial_state_ = np.zeros(1)
                estimator.source_ = SimpleNamespace(left=np.eye(p), right=np.eye(q))
                estimator.chart_ = SimpleNamespace(reconstruct=lambda state:
                    (np.ones((p, 1)), np.ones(1), np.ones((q, 1))))
            return estimator
        estimator.fit = fit
        return estimator

    return SimpleNamespace(PracticalCalibration=calibration, SparseSMART=model,
        Margins=lambda **kw: SimpleNamespace(**kw),
        ExactSource=lambda U, V: SimpleNamespace(U=U, V=V, mode="exact"),
        NoisySource=lambda coefficient, noise_std, gap_lower:
            SimpleNamespace(coefficient=coefficient, noise_std=noise_std, gap_lower=gap_lower, mode="noisy"))


def fit(tmp_path, *, setting=None, config=None, api=None, generator=tiny_data, force=False):
    return runner.run_setting(setting=setting or tiny_setting(), model="model1", experiment="exp1",
        seed_id=0, random_seed=123, destination=tmp_path / "result.json",
        config=config or runner.RunnerConfig(), generate_data_fn=generator,
        sparse_api=api or fake_api({}), force=force)


def test_success_records_fixed_truth_ranks_metrics_and_strict_json(tmp_path):
    calls = {}
    def generator(**kwargs):
        calls["generator"] = kwargs
        return tiny_data(**kwargs)
    outcome, result = fit(tmp_path, api=fake_api(calls), generator=generator)
    assert outcome == "written"
    assert result["status"] == "complete" and result["success"]
    assert result["avg_err"] == .25
    assert result["initial_avg_err"] == 1.
    assert result["last_accepted_avg_err"] == .25
    assert calls["generator"]["r_star"] == 5
    assert calls["generator"]["r0_star"] == 10
    assert calls["generator"]["random_seed"] == 123
    assert result["diagnostics"]["unknown"] is None
    assert len(result["input_fingerprint"]) == 64
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_noisy_input_is_observed_with_actual_noise_and_empirical_gate(tmp_path):
    calls = {}
    fit(tmp_path, api=fake_api(calls))
    source = calls["fit"][2]
    assert source.mode == "noisy" and source.noise_std == .01 and source.gap_lower == 1.
    np.testing.assert_equal(source.coefficient, tiny_data(n=6, p=3, q=2)["C0"])
    assert calls["calibration"]["enforce_source_accuracy"] is False
    assert calls["configuration"]["sparsity"] == (2, 2)


def test_exact_frames_use_sorted_observed_svd_not_latent_generator_arrays(tmp_path):
    calls = {}
    fit(tmp_path, setting=tiny_setting(0), api=fake_api(calls))
    source = calls["fit"][2]
    assert source.mode == "exact"
    np.testing.assert_allclose(np.abs(source.U[:, 0]), [0., 1., 0.])
    np.testing.assert_allclose(np.abs(source.V[:, 0]), [0., 1.])
    assert calls["calibration"]["support_limits"] == (1, 1)


def test_strict_source_gate_forwarded_and_labeled(tmp_path):
    calls = {}
    _, result = fit(tmp_path, api=fake_api(calls), config=runner.RunnerConfig(strict_source_check=True))
    assert calls["calibration"]["enforce_source_accuracy"] is True
    assert result["source_check_mode"] == "strict"


@pytest.mark.parametrize("partial", [True, False])
def test_failures_remain_records_and_partial_errors_never_count_as_success(tmp_path, partial):
    _, result = fit(tmp_path, api=fake_api({}, successful=False, partial=partial))
    assert result["status"] == "failed" and not result["success"]
    assert result["failure_reason"] == "line_search_failed"
    assert result["avg_err"] is None
    assert result["last_accepted_avg_err"] == (.25 if partial else None)


@pytest.mark.parametrize("r,r0", [(5, 0), (5, 3), (11, 10)])
def test_invalid_fitted_dimensions_are_not_clamped_or_fitted(tmp_path, r, r0):
    setting = runner.SimulationSetting(6, 20, 20, .01, r, r0, "invalid")
    _, result = fit(tmp_path, setting=setting, api=SimpleNamespace())
    assert result["status"] == "inapplicable"
    assert not result["applicable"] and not result["success"]
    assert result["avg_err"] is None and result["C_hat"] is None
    assert result["configuration"]["rank"] == r
    assert result["configuration"]["source_rank"] == r0


def test_resume_checks_config_and_data_then_skips_fitting(tmp_path):
    calls = {}
    api = fake_api(calls)
    fit(tmp_path, api=api)
    calls.clear()
    outcome, existing = fit(tmp_path, api=api)
    assert outcome == "skipped" and "fit" not in calls
    assert existing["status"] == "complete"
    with pytest.raises(ValueError, match="configuration differs.*--force"):
        fit(tmp_path, config=runner.RunnerConfig(iterations=4), api=api)
    def changed_data(**kwargs):
        data = tiny_data(**kwargs)
        data["X"][0, 0] += 1
        return data
    with pytest.raises(ValueError, match="Generated inputs differ.*--force"):
        fit(tmp_path, generator=changed_data, api=api)
    outcome, _ = fit(tmp_path, generator=changed_data, api=api, force=True)
    assert outcome == "written" and "fit" in calls


def test_resume_rejects_changed_implementation(tmp_path, monkeypatch):
    fit(tmp_path)
    monkeypatch.setattr(runner, "_implementation_fingerprint", lambda api: "changed")
    with pytest.raises(ValueError, match="Implementation differs.*--force"):
        fit(tmp_path)


def test_invalid_calibration_rejected_before_generating_or_creating_output(tmp_path):
    with pytest.raises(ValueError, match="support_limit exceeds"):
        fit(tmp_path, config=runner.RunnerConfig(support_limit=100))
    with pytest.raises(ValueError, match="penalty"):
        fit(tmp_path, config=runner.RunnerConfig(penalty=float("nan")))
    assert not (tmp_path / "result.json").exists()


def test_dry_run_uses_existing_seeds_without_loading_estimators(tmp_path, monkeypatch, capsys):
    def forbidden():
        raise AssertionError("No generator or estimator should load during dry run")
    monkeypatch.setattr(runner, "_load_generator", forbidden)
    monkeypatch.setattr(runner, "_load_sparse_api", forbidden)
    assert runner.main(["0", "0", "0", "--setting-index", "0", "--dry-run",
                        "--output-root", str(tmp_path)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["random_seed"] == 2103139804
    assert record["setting"]["n"] == 200
    assert not list(tmp_path.iterdir())
