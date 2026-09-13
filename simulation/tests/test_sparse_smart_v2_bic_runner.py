"""Operational BIC fixtures: no scientific grids, estimator solves or scheduler."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


SIMULATION = Path(__file__).resolve().parents[1]
if str(SIMULATION) not in sys.path:
    sys.path.insert(0, str(SIMULATION))
SPEC = importlib.util.spec_from_file_location("_tested_v2_bic_runner", SIMULATION / "run_sparse_smart_v2_bic.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class EvaluationGuard(dict):
    """Evaluation-only fields must not be consulted before BIC is computed."""
    allowed = False

    def __getitem__(self, name):
        if name in {"C_star", "X_validation", "Y_validation"} and not self.allowed:
            raise AssertionError(f"Evaluation leakage through {name}")
        return super().__getitem__(name)


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    import sparse_smart_v2.selection as selection
    x = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [1., 1., 1.]])
    observed = np.array([[1.2, .1], [0, .1], [0, -.1], [1.2, -.1]])
    truth = np.array([[.7, 0], [0, 0], [0, 0]])
    data = EvaluationGuard(X=x, Y=observed, C_star=truth,
                           X_validation=x.copy(), Y_validation=x @ truth)
    group = dict(group_id="tiny", p=3, q=2, margins={})
    task = dict(task_id=0, group_id="tiny", rank=1, initializer_source_rank=2,
                init_penalty=.03, free_counts=[1, 2], include_rrr=False)
    config = dict(inverse_step=20., iterations=2, max_backtracks=3, stationarity_tol=1e-5,
                  adaptive_anchors=True, max_anchor_switches=16, refinement_solver="masked_anchor_projected",
                  support_tolerance=0., penalties_u=[0., .01], penalties_v=[0., .01])
    calls, score_calls, events = [], [], []
    control = dict(mode="success", bad_selection=False, interrupt_free=None, fail_free=None)
    source = object()

    class Chart:
        anchors_u = np.array([0])
        anchors_v = np.array([0])
        center_u = np.eye(1)
        center_v = np.eye(1)

        def reconstruct(self, state):
            return np.array([[1.], [0], [0]]), np.array([state[0]]), np.array([[1.], [0]])

        def unpack(self, state):
            return (None, np.array([state[0]]), None, np.zeros((2, 1)), np.zeros((1, 1)))

    class Estimator:
        def __init__(self, **options):
            assert options["validation_patience"] is None
            assert options["checkpoint_iterations"] == ()
            self.options = options

        def fit(self, X, Y, *, source, validation_data, _initialization_cache):
            assert X is x and Y is observed
            assert validation_data is None
            events.append("fit")
            free = self.options["free_directions"]
            direct = free == (3, 2)
            calls.append(dict(free=free, calibration=self.options["calibration"],
                              cache=_initialization_cache, source=source))
            assert source is None if direct else source is fixture_source
            if free[0] == control["interrupt_free"]:
                raise KeyboardInterrupt("fixture interrupts an unfinished block")
            if free[0] == control["fail_free"]:
                raise RuntimeError("transient fixture launch failure")
            if control["mode"] == "exception":
                raise RuntimeError("fixture process error")
            penalty_u = self.options["calibration"].penalties[0]
            # One candidate matches training; the other matches evaluation.
            amplitude = 1.2 if penalty_u else .7
            coefficient = np.array([[amplitude, 0], [0, 0], [0, 0]])
            self.method_ = "target_rrr" if direct else "sparse_smart_v2"
            self.validation_history_ = []
            self.success_ = control["mode"] == "success"
            self.status_ = "completed" if self.success_ else control["mode"]
            self.termination_reason_ = "max_iterations" if self.success_ else self.status_
            self.n_iter_ = 0 if direct or not self.success_ else 2
            self.selected_iteration_ = self.n_iter_ if self.success_ else None
            if control["bad_selection"]:
                self.selected_iteration_ = 0
            self.optimization_converged_ = False
            self.message_ = "fixture"
            self.metadata_ = {}
            self.coefficient_ = coefficient
            self.last_coefficient_ = coefficient
            self.rrr_certificate_ = {"certified": True}
            self.chart_ = Chart()
            self.last_state_ = np.array([amplitude])
            self.free_rows_ = SimpleNamespace(
                rows_u=np.arange(free[0]), rows_v=np.arange(free[1]),
                penalized_u=(np.arange(1, 3) >= free[0])[:, None],
                penalized_v=(np.arange(1, 2) >= free[1])[:, None])
            self.history_ = [{"iteration": self.n_iter_, "objective": 1.}]
            return self

    fixture_source = source
    api = SimpleNamespace(SparseSMARTv2=Estimator, Margins=lambda **kw: kw,
                          PracticalCalibration=lambda initial, penalties, inverse, caps:
                              SimpleNamespace(init_penalty=initial, penalties=penalties, caps=caps))
    real_score = selection.bic_score

    def score(X, Y, C, **options):
        assert X is x and Y is observed
        assert not (set(options) & {"validation_data", "truth", "C_star", "prediction_mse"})
        events.append("score")
        result = real_score(X, Y, C, **options)
        score_calls.append(result)
        data.allowed = True
        return result

    monkeypatch.setattr(selection, "bic_score", score)
    real_metrics = runner.legacy._metrics

    def metrics(C, data):
        assert events[-1] == "score"
        events.append("evaluate")
        result = real_metrics(C, data)
        data.allowed = False
        return result

    monkeypatch.setattr(runner.legacy, "_metrics", metrics)
    root = tmp_path / "bic"
    root.mkdir()
    plan = dict(method=runner.METHOD, schema_version=1, n_tasks=1, tasks=[task], groups=[group],
                plan_fingerprint="p" * 64, source_manifest_sha256="s" * 64, configuration=config)
    monkeypatch.setattr(runner, "load_plan", lambda root: (plan, {}))
    monkeypatch.setattr(runner, "load_group", lambda root, plan, group:
                        (data, {}, dict(source_metadata={}, fingerprints={"train": "unchanged"})))
    monkeypatch.setattr(runner, "_source", lambda *args: source)
    monkeypatch.setattr(runner.legacy, "_require_slurm", lambda root: None)
    monkeypatch.setattr(runner.legacy, "_load_api", lambda: api)
    monkeypatch.setattr(runner.legacy, "_verify_imports", lambda manifest: None)
    return SimpleNamespace(root=root, data=data, group=group, task=task, config=config,
                           api=api, source=source, calls=calls, score_calls=score_calls,
                           events=events, control=control, plan=plan)


def candidate(fixture, *, free=1, left=.01, right=.01):
    return runner.fit_candidate(fixture.api, fixture.group, fixture.task, fixture.config,
                                fixture.data, fixture.source, {}, free, left, right, "candidate")


def test_terminal_training_score_precedes_truth_and_validation_evaluation(fixture):
    row, arrays, _ = candidate(fixture)
    assert row["success"] and row["execution_success"]
    assert fixture.events == ["fit", "score", "evaluate"]
    assert row["selected_iteration"] == row["n_iter"] == 2
    expected = np.sum((fixture.data["Y"] - fixture.data["X"] @ arrays["coefficient"]) ** 2)
    assert row["selection"]["rss"] == pytest.approx(expected)
    # Truth-based error is deliberately very different from training RSS.
    assert row["metrics"]["training_prediction_mse"] != pytest.approx(expected / 8)
    assert row["selection"]["rss"] == pytest.approx(.04)
    assert arrays["state"][0] == arrays["coefficient"][0, 0]


@pytest.mark.parametrize("status", ["initialization_spectrum_failed", "anchor_selection_failed", "numerical_stagnation"])
def test_algorithmic_exclusions_are_distinct_from_execution_errors(fixture, status):
    fixture.control["mode"] = status
    row, arrays, history = candidate(fixture)
    assert not row["success"] and row["execution_success"]
    assert row["fit_status"] == row["termination_reason"] == status
    assert row["selection"] is None and row["metrics"] is None
    assert arrays == {} and history is None
    assert fixture.events == ["fit"]


def test_exceptions_are_execution_failures_and_cannot_be_scored(fixture):
    fixture.control["mode"] = "exception"
    row, arrays, history = candidate(fixture)
    assert not row["success"] and not row["execution_success"]
    assert row["fit_status"] == "execution_failed"
    assert row["error_type"] == "RuntimeError"
    assert row["selection"] is None and row["metrics"] is None
    assert not arrays and history is None


def test_rejects_a_validation_selected_prefix_as_a_terminal_candidate(fixture):
    fixture.control["bad_selection"] = True
    row, arrays, history = candidate(fixture)
    assert not row["success"] and not row["execution_success"]
    assert "not the terminal" in row["message"]
    assert fixture.events == ["fit"]
    assert not arrays and history is None


def test_rrr_endpoint_is_charged_full_model_dimension(fixture):
    row, arrays, _ = candidate(fixture, free=None, left=0., right=0.)
    assert row["success"]
    assert row["fit_method"] == "target_rrr"
    assert row["selection"]["model_dimension"] == 1 * (3 + 2 - 1)
    assert row["selection"]["support_u"] is None
    assert row["rrr_certificate"]["certified"]
    assert set(arrays) == {"coefficient"}
    assert fixture.calls[0]["source"] is None


def test_grouped_blocks_choose_bic_not_evaluation_and_save_only_block_winners(fixture):
    result = runner.run_task(fixture.root, 0)
    assert result["success"] and result["execution_success"]
    assert len(result["outcomes"]) == 6
    assert len({id(call["cache"]) for call in fixture.calls}) == 1
    assert len(fixture.calls) == 6
    retained = [row for row in result["outcomes"] if row["state_key"]]
    assert len(retained) == 2
    for row in retained:
        assert row["penalty_u"] == .01
        assert row["metrics"]["validation_mse"] > 0
        competitors = [c for c in result["outcomes"] if c["free_directions"] == row["free_directions"]]
        assert any(c["metrics"]["validation_mse"] == 0 for c in competitors)
        assert row["selection"]["score"] == min(c["selection"]["score"] for c in competitors)
    saved = runner._load_npz(fixture.root / "tasks/000000/states.npz")
    for row in retained:
        coefficient = saved[row["state_key"] + "coefficient"]
        expected = np.sum((fixture.data["Y"] - fixture.data["X"] @ coefficient) ** 2)
        assert row["selection"]["rss"] == pytest.approx(expected)
    for name, expected in result["files"].items():
        assert runner.sha(fixture.root / "tasks/000000" / name) == expected
    calls_before = len(fixture.calls)
    assert runner.run_task(fixture.root, 0) == result
    assert len(fixture.calls) == calls_before


def test_resume_reuses_completed_free_block_and_finishes_interrupted_one(fixture):
    fixture.control["interrupt_free"] = 2
    with pytest.raises(KeyboardInterrupt):
        runner.run_task(fixture.root, 0)
    folder = fixture.root / "tasks/000000"
    assert (folder / "block-1.json").exists()
    assert not (folder / "block-2.json").exists()
    original = (folder / "block-1.npz").read_bytes()
    before = len(fixture.calls)
    fixture.control["interrupt_free"] = None
    result = runner.run_task(fixture.root, 0)
    assert result["execution_success"] and len(result["outcomes"]) == 6
    assert all(call["free"] == (2, 2) for call in fixture.calls[before:])
    assert (folder / "block-1.npz").read_bytes() == original
    assert len([row for row in result["outcomes"] if row["state_key"]]) == 2


def test_resume_retries_execution_failed_blocks(fixture):
    fixture.control["fail_free"] = 1
    first = runner.run_task(fixture.root, 0)
    assert not first["execution_success"]
    before = len(fixture.calls)
    fixture.control["fail_free"] = None
    second = runner.run_task(fixture.root, 0)
    assert second["execution_success"], "Transient execution errors must not become permanently cached outcomes"
    assert len(fixture.calls) > before
    assert all(call["free"] == (1, 1) for call in fixture.calls[before:])


def test_completed_artifact_tampering_is_rejected_before_reuse(fixture):
    runner.run_task(fixture.root, 0)
    path = fixture.root / "tasks/000000/states.npz"
    path.write_bytes(path.read_bytes() + b"modified")
    before = len(fixture.calls)
    with pytest.raises(ValueError, match="artifact changed"):
        runner.run_task(fixture.root, 0)
    assert len(fixture.calls) == before
