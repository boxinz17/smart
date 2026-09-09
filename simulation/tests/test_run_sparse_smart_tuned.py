"""Selection truth isolation, checkpoint provenance, and tuned-run failure records."""

from collections.abc import Mapping
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_sparse_smart_tuned.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_sparse_smart_tuned", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def tiny_setting(sigma0=.01):
    return runner.SimulationSetting(6, 3, 2, sigma0, 1, 2, "tiny")


def tiny_data(**kwargs):
    n, p, q = kwargs["n"], kwargs["p"], kwargs["q"]
    return dict(X=np.arange(n*p, dtype=float).reshape(n, p), Y=np.zeros((n, q)),
                C0=np.eye(p, q)*np.arange(1., q+1), C_star=np.zeros((p, q)))


def fake_api(calls, *, successful=True, error=None, overlapping_split=False):
    def tuner_factory(**kwargs):
        calls["config"] = kwargs
        tuner = SimpleNamespace()

        def fit(X, Y, *, source):
            calls["fit"] = (X.copy(), Y.copy(), source)
            if error is not None:
                calls["fit_complete"] = True
                raise error
            calls["fit_complete"] = True
            params = dict(init_penalty=kwargs["init_penalties"][0],
                          penalty_u=kwargs["penalties_u"][0], penalty_v=kwargs["penalties_v"][0],
                          support_limits=(2, 1), step_size_inverse=kwargs["step_size_inverse"])
            tuner.train_indices_ = np.array([0, 1, 2, 3])
            tuner.validation_indices_ = np.array([3, 4]) if overlapping_split else np.array([4, 5])
            failed = dict(candidate_id=0, params=params, success=False,
                          status="line_search_failed", message="rejected", has_partial_coefficient=True,
                          partial_validation_mse=.001, validation_mse=None)
            tuner.selection_history_ = [failed]
            tuner.success_ = successful
            tuner.status_ = "selected" if successful else "no_successful_candidate"
            tuner.message_ = "selected" if successful else "all candidates failed"
            tuner.best_params_ = params if successful else None
            tuner.best_score_ = .3 if successful else None
            tuner.selected_iteration_ = 2 if successful else None
            if successful:
                tuner.selection_history_.append(dict(candidate_id=1, params=params,
                    success=True, status="completed_iterations", validation_mse=.3))
                tuner.coefficient_ = np.full((X.shape[1], Y.shape[1]), .25)
                tuner.estimator_ = SimpleNamespace(n_iter_=4, history_=[dict(iteration=4, loss=.6)],
                    validation_history_=[dict(iteration=0, loss=.4), dict(iteration=2, loss=.3)],
                    diagnostics_={"theorem_certified": False, "nonfinite": float("inf")},
                    supports_={"u": np.array([0, 1]), "v": np.array([0])},
                    termination_reason_="iteration_budget", status_="completed_iterations")
            return tuner
        tuner.fit = fit
        return tuner

    return SimpleNamespace(SparseSMARTTuner=tuner_factory,
        Margins=lambda **kw: SimpleNamespace(**kw),
        ExactSource=lambda U, V: SimpleNamespace(U=U, V=V, mode="exact"),
        NoisySource=lambda coefficient, noise_std, gap_lower:
            SimpleNamespace(coefficient=coefficient, noise_std=noise_std, gap_lower=gap_lower, mode="noisy"))


def run(tmp_path, *, api=None, config=None, setting=None, generator=tiny_data, force=False):
    return runner.run_setting(setting=setting or tiny_setting(), model="model1", experiment="exp1",
        seed_id=0, random_seed=123, destination=tmp_path / "result.json",
        config=config or runner.RunnerConfig(), generate_data_fn=generator,
        sparse_api=api or fake_api({}), force=force)


class TruthGuard(Mapping):
    """Fail immediately if the runner touches evaluation truth before fit returns."""
    def __init__(self, data, calls):
        self.data, self.calls = data, calls

    def __getitem__(self, key):
        if key == "C_star":
            assert self.calls.get("fit_complete"), "Evaluation truth reached the tuning pipeline"
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def test_truth_is_not_read_until_selection_has_completed(tmp_path):
    calls = {}
    def generator(**kwargs):
        calls["generator"] = kwargs
        return TruthGuard(tiny_data(**kwargs), calls)
    _, result = run(tmp_path, api=fake_api(calls), generator=generator)
    assert result["success"] and result["avg_err"] == .25
    assert calls["generator"]["r_star"] == 5 and calls["generator"]["r0_star"] == 10
    assert "C_star" not in calls["config"] and "truth" not in calls["config"]
    assert result["configuration"]["tuning_uses_truth"] is False


def test_changing_truth_only_changes_evaluation_not_selected_model(tmp_path):
    calls1, calls2 = {}, {}
    _, first = run(tmp_path / "first", api=fake_api(calls1))
    def alternate_truth(**kwargs):
        data = tiny_data(**kwargs)
        data["C_star"][:] = 10
        return data
    _, second = run(tmp_path / "second", api=fake_api(calls2), generator=alternate_truth)
    assert first["best_params"] == second["best_params"]
    assert first["C_hat"] == second["C_hat"]
    assert first["validation_loss"] == second["validation_loss"]
    assert first["avg_err"] == pytest.approx(.25)
    assert second["avg_err"] == pytest.approx(9.75)
    assert first["observed_input_fingerprint"] == second["observed_input_fingerprint"]
    assert first["evaluation_truth_fingerprint"] != second["evaluation_truth_fingerprint"]


def test_success_records_splits_grid_selected_supports_iterations_errors_and_json(tmp_path):
    calls = {}
    outcome, result = run(tmp_path, api=fake_api(calls))
    assert outcome == "written" and result["status"] == "complete"
    assert result["split"]["n_train"] == 4 and result["split"]["n_validation"] == 2
    assert result["split"]["train_indices"] == [0, 1, 2, 3]
    assert result["split"]["validation_indices"] == [4, 5]
    assert result["split"]["refit_on_all_data"] is False
    assert result["selected_iteration"] == 2 and result["n_iter"] == 4
    assert result["selected_supports"] == {"u": [0, 1], "v": [0]}
    assert result["termination_reason"] == "iteration_budget"
    assert len(result["fit_errors"]) == 1
    assert result["fit_errors"][0]["has_partial_coefficient"] is True
    assert result["validation_loss"] == .3  # Never substitute the failed partial score .001.
    assert result["diagnostics"]["nonfinite"] is None
    for key in ("input_fingerprint", "observed_input_fingerprint", "evaluation_truth_fingerprint",
                "configuration_fingerprint", "implementation_fingerprint"):
        assert len(result[key]) == 64
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_default_support_is_full_actual_noisy_complement():
    setting = runner.experiment_settings(0, 0)[0]
    resolved = runner.resolved_configuration(setting, runner.RunnerConfig())
    assert resolved["support_limits"] == [[475, 225]]
    assert resolved["actual_complement_counts"] == [475, 225]
    assert resolved["candidate_count"] == 9
    exact = replace(setting, sigma0=0.)
    assert runner.resolved_configuration(exact, runner.RunnerConfig())["support_limits"] == [[25, 25]]


def test_sources_and_explicit_empirical_gate_are_forwarded(tmp_path):
    calls = {}
    run(tmp_path, api=fake_api(calls))
    source = calls["fit"][2]
    assert source.mode == "noisy" and source.noise_std == .01
    np.testing.assert_equal(source.coefficient, tiny_data(n=6, p=3, q=2)["C0"])
    assert calls["config"]["enforce_source_accuracy"] is False
    assert calls["config"]["support_limits"] is None
    assert calls["config"]["iterations"] == 500
    assert calls["config"]["stationarity_tol"] == 1e-6


def test_exact_source_uses_observed_svd_and_forwards_grid_and_strict_gate(tmp_path):
    calls = {}
    config = runner.RunnerConfig(init_penalties=(.02, .04), penalties_u=(0., .01),
                                 penalties_v=(.05,), support_limits=((0, 0), (1, 1)),
                                 strict_source_check=True, split_seed=5)
    _, result = run(tmp_path, setting=tiny_setting(0), config=config, api=fake_api(calls))
    source = calls["fit"][2]
    assert source.mode == "exact"
    np.testing.assert_allclose(np.abs(source.U[:, 0]), [0., 1., 0.])
    assert calls["config"]["support_limits"] == ((0, 0), (1, 1))
    assert calls["config"]["random_state"] == 5
    assert calls["config"]["enforce_source_accuracy"] is True
    assert result["configuration"]["candidate_count"] == 8


def test_all_failed_candidates_are_explicit_and_never_evaluated_as_winner(tmp_path):
    calls = {}
    _, result = run(tmp_path, api=fake_api(calls, successful=False))
    assert result["status"] == "all_candidates_failed"
    assert result["all_candidates_failed"] and not result["success"]
    assert result["failure_reason"] == "no_successful_candidate"
    assert result["C_hat"] is None and result["avg_err"] is None
    assert result["validation_loss"] is None and result["best_params"] is None
    assert len(result["selection_history"]) == len(result["fit_errors"]) == 1
    assert result["split"]["n_train"] == 4


def test_numerical_setup_error_is_written_but_programming_error_propagates(tmp_path):
    _, result = run(tmp_path, api=fake_api({}, error=ValueError("invalid setup")))
    assert result["status"] == "failed" and result["failure_reason"] == "ValueError"
    assert not result["all_candidates_failed"]  # No candidate grid was run.
    with pytest.raises(RuntimeError, match="programming error"):
        run(tmp_path / "unexpected", api=fake_api({}, error=RuntimeError("programming error")))
    assert not (tmp_path / "unexpected" / "result.json").exists()


def test_overlapping_validation_split_is_not_silently_published(tmp_path):
    with pytest.raises(RuntimeError, match="overlapping"):
        run(tmp_path, api=fake_api({}, overlapping_split=True))
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("rank,source_rank", [(5, 0), (5, 3), (11, 10)])
def test_inapplicable_dimensions_are_preserved_without_fitting(tmp_path, rank, source_rank):
    setting = runner.SimulationSetting(6, 20, 20, .01, rank, source_rank, "invalid")
    _, result = run(tmp_path, setting=setting, api=SimpleNamespace())
    assert result["status"] == "inapplicable" and not result["success"]
    assert result["configuration"]["rank"] == rank
    assert result["configuration"]["source_rank"] == source_rank
    assert result["selection_history"] == [] and result["avg_err"] is None


def test_resume_verifies_configuration_observations_truth_and_code(tmp_path, monkeypatch):
    calls = {}
    api = fake_api(calls)
    run(tmp_path, api=api)
    calls.clear()
    outcome, _ = run(tmp_path, api=api)
    assert outcome == "skipped" and "fit" not in calls
    with pytest.raises(ValueError, match="configuration differs.*--force"):
        run(tmp_path, api=api, config=runner.RunnerConfig(penalties_u=(.03,)))
    def changed_observations(**kwargs):
        data = tiny_data(**kwargs)
        data["X"][0, 0] += 1
        return data
    with pytest.raises(ValueError, match="Generated inputs differ.*--force"):
        run(tmp_path, api=api, generator=changed_observations)
    def changed_truth(**kwargs):
        data = tiny_data(**kwargs)
        data["C_star"][0, 0] += 1
        return data
    with pytest.raises(ValueError, match="Evaluation truth differs.*--force"):
        run(tmp_path, api=api, generator=changed_truth)
    monkeypatch.setattr(runner, "_implementation_fingerprint", lambda api, generator: "changed")
    with pytest.raises(ValueError, match="Implementation differs.*--force"):
        run(tmp_path, api=api)
    outcome, result = run(tmp_path, api=api, force=True)
    assert outcome == "written" and result["implementation_fingerprint"] == "changed"


@pytest.mark.parametrize("config", [
    runner.RunnerConfig(init_penalties=()), runner.RunnerConfig(penalties_u=(float("nan"),)),
    runner.RunnerConfig(penalties_v=(.01, .01)), runner.RunnerConfig(validation_fraction=1.),
    runner.RunnerConfig(support_limits=((1000, 1000),)),
])
def test_invalid_grid_fails_before_output(tmp_path, config):
    with pytest.raises(ValueError):
        run(tmp_path, config=config)
    assert not (tmp_path / "result.json").exists()


def test_cli_dry_run_uses_existing_seed_and_selected_grids_without_fits(tmp_path, monkeypatch, capsys):
    def forbidden():
        raise AssertionError("Dry run must not load generators or estimators")
    monkeypatch.setattr(runner, "_load_generator", forbidden)
    monkeypatch.setattr(runner, "_load_sparse_api", forbidden)
    assert runner.main(["0", "0", "0", "--setting-index", "0", "--dry-run",
                        "--init-penalties", ".01,.03", "--penalties-u", "0,.01",
                        "--penalties-v", ".02", "--support-grid", "475:225,100:50",
                        "--output-root", str(tmp_path)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["random_seed"] == 2103139804
    assert value["configuration"]["candidate_count"] == 8
    assert value["configuration"]["support_limits"] == [[475, 225], [100, 50]]
    assert "SparseSMARTTuned_result" in value["destination"]
    assert not list(tmp_path.iterdir())
