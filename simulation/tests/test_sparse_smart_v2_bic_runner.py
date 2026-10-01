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
    control = dict(mode="success", bad_selection=False, interrupt_free=None, fail_free=None,
                   initializer_builds=0)
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
            assert options["checkpoint_iterations"] == (tuple(config["checkpoint_iterations"])
                if config.get("selection_rule") == "bic_checkpoint" else ())
            self.options = options

        def fit(self, X, Y, *, source, validation_data, _initialization_cache):
            assert X is x and Y is observed
            assert validation_data is None
            events.append("fit")
            free = self.options["free_directions"]
            direct = free == (3, 2)
            calls.append(dict(free=free, calibration=self.options["calibration"],
                              cache=_initialization_cache, source=source, options=self.options))
            assert source is None if direct else source is fixture_source
            if free[0] == control["interrupt_free"]:
                raise KeyboardInterrupt("fixture interrupts an unfinished block")
            if free[0] == control["fail_free"]:
                raise RuntimeError("transient fixture launch failure")
            if control["mode"] == "exception":
                raise RuntimeError("fixture process error")
            if not direct and "fixture_initializer" not in _initialization_cache:
                _initialization_cache["fixture_initializer"] = object()
                control["initializer_builds"] += 1
            penalty_u = self.options["calibration"].penalties[0]
            # One candidate matches training; the other matches evaluation.
            amplitude = 1.2 if penalty_u else .7
            self.fixture_checkpoints = control.get("checkpoint_amplitudes", {}).get(penalty_u)
            if self.fixture_checkpoints and not direct:
                amplitude = self.fixture_checkpoints[2]
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

        def checkpoint_model(self, iteration):
            assert iteration in self.options["checkpoint_iterations"]
            amplitude = self.fixture_checkpoints[iteration]
            view = SimpleNamespace(**vars(self))
            view.n_iter_ = view.selected_iteration_ = iteration
            view.coefficient_ = np.array([[amplitude, 0], [0, 0], [0, 0]])
            view.last_state_ = np.array([amplitude])
            return view

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
        return result

    monkeypatch.setattr(selection, "bic_score", score)
    real_metrics = runner.legacy._metrics

    def metrics(C, data):
        assert events[-1] == "score" or (config.get("selection_rule") == "bic_checkpoint" and events[-1] == "evaluate")
        data.allowed = True
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
    assert row["process_cpu_seconds"] >= 0
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
    assert [row["candidate_id"] for row in result["outcomes"]] == [
        f"t0_{free}_u{u}_v{v}" for free in (1, 2) for u, v in ((0, 1), (1, 0), (1, 1))]
    assert len({id(call["cache"]) for call in fixture.calls}) == 1
    assert fixture.control["initializer_builds"] == 1
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


def test_explicit_pairs_reuse_initializer_preserve_ids_and_deduplicate_rrr(fixture, monkeypatch):
    fixture.config.update(penalties_u=[0., .001, .0025, .01], penalties_v=[0., .001, .0025, .01],
                          penalty_pairs=[[.01, .01], [.001, .001], [.0025, .0025]], iterations=200)
    fixture.task["include_rrr"] = True
    cpu_times = iter(np.linspace(101., 108.25, 16))
    monkeypatch.setattr(runner.time, "process_time", lambda: next(cpu_times))
    result = runner.run_task(fixture.root, 0)
    assert result["execution_success"]
    assert len(result["outcomes"]) == len(fixture.calls) == 1 + 2 * 3
    assert [row["candidate_id"] for row in result["outcomes"]] == ["t0_rrr_u0_v0"] + [
        f"t0_{free}_u{i}_v{i}" for free in (1, 2) for i in (3, 1, 2)]
    assert [(row["penalty_u"], row["penalty_v"]) for row in result["outcomes"]] == [
        (0., 0.), (.01, .01), (.001, .001), (.0025, .0025), (.01, .01), (.001, .001), (.0025, .0025)]
    assert sum(row["fit_method"] == "target_rrr" for row in result["outcomes"]) == 1
    assert fixture.control["initializer_builds"] == 1
    assert len({id(call["cache"]) for call in fixture.calls}) == 1
    assert all(call["options"]["iterations"] == 200 for call in fixture.calls)
    assert all(call["options"]["stationarity_tol"] == 1e-5 for call in fixture.calls)
    assert result["validation_used_for_fit"] is False
    assert result["selection_rule"] == "bic_terminal"
    assert result["process_cpu_seconds"] == 7.25
    assert result["elapsed_seconds"] >= 0
    assert fixture.events == ["fit", "score", "evaluate"] * 7


def test_explicit_one_sided_pair_is_allowed_and_uses_its_axis_indices(fixture):
    fixture.config["penalty_pairs"] = [[0., .01]]
    result = runner.run_task(fixture.root, 0)
    assert [row["candidate_id"] for row in result["outcomes"]] == ["t0_1_u0_v1", "t0_2_u0_v1"]
    assert all(row["fit_method"] == "sparse_smart_v2" for row in result["outcomes"])


def test_registered_rrr_reuse_bypasses_only_rrr_fit(fixture, monkeypatch):
    fixture.task["include_rrr"] = True
    baseline = runner.run_task(fixture.root, 0)
    rrr_row = baseline["outcomes"][0]
    saved = runner._load_npz(fixture.root / "tasks/000000/states.npz")
    coefficient = saved[rrr_row["state_key"] + "coefficient"]
    descriptor = {"fixture": "registered endpoint"}
    fixture.plan["rrr_reuse"] = {"tiny": descriptor}
    fixture.plan["plan_fingerprint"] = "r" * 64
    fixture.calls.clear()
    fixture.events.clear()
    fixture.control["initializer_builds"] = 0
    reuse_calls = []

    def reuse_rrr(actual, group, task, config, data, candidate_id):
        assert actual is descriptor and group is fixture.group and task is fixture.task
        assert config is fixture.config and data is fixture.data
        assert candidate_id == "t0_rrr_u0_v0"
        reuse_calls.append(candidate_id)
        return dict(rrr_row, state_key=None), {"coefficient": coefficient}, {"history": []}

    monkeypatch.setitem(sys.modules, "sparse_smart_v2_cheap_bic_reuse", SimpleNamespace(reuse_rrr=reuse_rrr))
    root = fixture.root / "registered"
    result = runner.run_task(root, 0)
    assert result["execution_success"] and len(result["outcomes"]) == 7
    assert reuse_calls == ["t0_rrr_u0_v0"]
    assert len(fixture.calls) == 6
    assert all(call["free"] != (3, 2) for call in fixture.calls)
    assert fixture.control["initializer_builds"] == 1
    assert len({id(call["cache"]) for call in fixture.calls}) == 1
    assert fixture.events == ["fit", "score", "evaluate"] * 6
    for before, after in zip(baseline["outcomes"], result["outcomes"]):
        assert {k: v for k, v in before.items() if k not in ("elapsed_seconds", "process_cpu_seconds")} == {
            k: v for k, v in after.items() if k not in ("elapsed_seconds", "process_cpu_seconds")}
    reused = runner._load_npz(root / "tasks/000000/states.npz")
    np.testing.assert_array_equal(reused[result["outcomes"][0]["state_key"] + "coefficient"], coefficient)


def test_rrr_registry_for_another_group_keeps_legacy_rrr_fit(fixture, monkeypatch):
    fixture.task["include_rrr"] = True
    fixture.plan["rrr_reuse"] = {"other-group": {"fixture": "not this group"}}

    def unexpected_reuse(*args):
        raise AssertionError("Unregistered group must compute its own RRR endpoint")

    monkeypatch.setitem(sys.modules, "sparse_smart_v2_cheap_bic_reuse", SimpleNamespace(reuse_rrr=unexpected_reuse))
    result = runner.run_task(fixture.root, 0)
    assert result["execution_success"] and len(fixture.calls) == 7
    assert sum(call["free"] == (3, 2) for call in fixture.calls) == 1


def test_rrr_reuse_integrity_failure_propagates_without_refitting(fixture, monkeypatch):
    fixture.task["include_rrr"] = True
    fixture.plan["rrr_reuse"] = {"tiny": {"fixture": "changed endpoint"}}

    def rejected_reuse(*args):
        raise ValueError("RRR state hash mismatch")

    monkeypatch.setitem(sys.modules, "sparse_smart_v2_cheap_bic_reuse", SimpleNamespace(reuse_rrr=rejected_reuse))
    with pytest.raises(ValueError, match="RRR state hash mismatch"):
        runner.run_task(fixture.root, 0)
    assert not fixture.calls
    assert not (fixture.root / "tasks/000000/block-rrr.json").exists()
    assert not (fixture.root / "tasks/000000/result.json").exists()


@pytest.mark.parametrize("registry", [None, []])
def test_invalid_rrr_registry_fails_before_creating_task(fixture, registry):
    fixture.plan["rrr_reuse"] = registry
    with pytest.raises(ValueError, match="group-to-descriptor mapping"):
        runner.run_task(fixture.root, 0)
    assert not fixture.calls
    assert not (fixture.root / "tasks").exists()


def test_assessment_gate_precedes_fitting(fixture, monkeypatch):
    fixture.config["cheap_bic_phase"] = "assessment"

    def verify_assessment_gate(root, plan):
        assert root == fixture.root and plan is fixture.plan
        assert not (root / "tasks").exists()
        assert not fixture.calls
        fixture.events.append("gate")

    monkeypatch.setitem(sys.modules, "sparse_smart_v2_cheap_bic_reuse",
                        SimpleNamespace(verify_assessment_gate=verify_assessment_gate))
    result = runner.run_task(fixture.root, 0)
    assert result["execution_success"]
    assert fixture.events == ["gate"] + ["fit", "score", "evaluate"] * 6


def test_missing_assessment_gate_prevents_task_artifacts_and_fits(fixture, monkeypatch):
    fixture.config["cheap_bic_phase"] = "assessment"

    def verify_assessment_gate(root, plan):
        raise ValueError("Assessment pair is not frozen")

    monkeypatch.setitem(sys.modules, "sparse_smart_v2_cheap_bic_reuse",
                        SimpleNamespace(verify_assessment_gate=verify_assessment_gate))
    with pytest.raises(ValueError, match="Assessment pair is not frozen"):
        runner.run_task(fixture.root, 0)
    assert not fixture.calls
    assert not (fixture.root / "tasks").exists()


@pytest.mark.parametrize("pairs, message", [
    (None, "nonempty list"),
    ({"u": .01, "v": .01}, "nonempty list"),
    ([], "nonempty list"),
    ([.01, .01], "contain two values"),
    ([[.01]], "contain two values"),
    ([[.01, .01, .01]], "contain two values"),
    ([[True, .01]], "finite nonnegative"),
    ([["0.01", .01]], "finite nonnegative"),
    ([[float("nan"), .01]], "finite nonnegative"),
    ([[.01, float("inf")]], "finite nonnegative"),
    ([[-.01, .01]], "finite nonnegative"),
    ([[0., 0.]], "RRR is handled separately"),
    ([[.01, .01], [.01, .01]], "distinct pairs"),
    ([[.0025, .01]], "existing axes"),
    ([[.01, .0025]], "existing axes"),
])
def test_invalid_explicit_pairs_fail_before_creating_task_or_fitting(fixture, pairs, message):
    fixture.config["penalty_pairs"] = pairs
    with pytest.raises(ValueError, match=message):
        runner.run_task(fixture.root, 0)
    assert not fixture.calls
    assert not (fixture.root / "tasks").exists()


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


def checkpoint_policy(fixture, amplitudes):
    fixture.config.update(selection_rule="bic_checkpoint", checkpoint_iterations=[0, 1, 2])
    fixture.control["checkpoint_amplitudes"] = amplitudes


@pytest.mark.parametrize("amplitudes, expected_iteration", [
    ({0: 1.2, 1: .8, 2: .9}, 0),
    ({0: .8, 1: 1.2, 2: .9}, 1),
    ({0: 1.2, 1: 1.2, 2: .9}, 0),
])
def test_checkpoint_bic_uses_initializer_and_early_states_with_earliest_tie(fixture, amplitudes, expected_iteration):
    checkpoint_policy(fixture, {.01: amplitudes})
    row, arrays, history = candidate(fixture)
    assert row["success"] and row["selected_iteration"] == expected_iteration
    assert row["n_iter"] == 2 and not row["optimization_converged"]
    assert [c["iteration"] for c in row["checkpoint_scores"]] == [0, 1, 2]
    assert fixture.events == ["fit", "score", "score", "score", "evaluate", "evaluate"]
    assert row["selection"]["rss"] == pytest.approx(.04)
    assert row["terminal_selection"]["rss"] > row["selection"]["rss"]
    assert arrays["coefficient"][0, 0] == 1.2
    assert arrays["terminal_coefficient"][0, 0] == .9
    assert row["terminal_metrics"]["coefficient_rmse"] < row["metrics"]["coefficient_rmse"]
    assert history["history"] == [{"iteration": 2, "objective": 1.}]


def test_checkpoint_rule_adds_successful_terminal_when_not_requested(fixture):
    checkpoint_policy(fixture, {.01: {0: .8, 1: .9, 2: 1.2}})
    fixture.config["checkpoint_iterations"] = [0, 1]
    row, arrays, _ = candidate(fixture)
    assert row["success"] and row["selected_iteration"] == 2
    assert [c["iteration"] for c in row["checkpoint_scores"]] == [0, 1, 2]
    assert row["selection"] == row["terminal_selection"]
    np.testing.assert_array_equal(arrays["coefficient"], arrays["terminal_coefficient"])


def test_numerically_failed_checkpoint_trajectory_does_not_salvage_initializer(fixture):
    checkpoint_policy(fixture, {.01: {0: 1.2, 1: .9, 2: .9}})
    fixture.control["mode"] = "numerical_stagnation"
    row, arrays, history = candidate(fixture)
    assert not row["success"] and row["execution_success"]
    assert row["selection"] is row["terminal_selection"] is None
    assert row["metrics"] is row["terminal_metrics"] is None
    assert row["checkpoint_scores"] == [] and arrays == {} and history is None
    assert fixture.events == ["fit"]


def test_checkpoint_and_terminal_block_winners_are_saved_separately_and_resume(fixture):
    checkpoint_policy(fixture, {0.: {0: 1.2, 1: .8, 2: .8}, .01: {0: .7, 1: .9, 2: 1.1}})
    fixture.task["include_rrr"] = True
    result = runner.run_task(fixture.root, 0)
    assert result["selection_rule"] == "bic_checkpoint"
    assert result["execution_success"]
    assert len(fixture.calls) == 7 and fixture.control["initializer_builds"] == 1
    assert sum(row["fit_method"] == "target_rrr" for row in result["outcomes"]) == 1
    folder = fixture.root / "tasks/000000"
    saved = runner._load_npz(folder / "states.npz")
    for free in (1, 2):
        rows = [row for row in result["outcomes"] if row["free_directions"] == [free, free]]
        selected = [row for row in rows if row["state_key"]]
        terminal = [row for row in rows if row["terminal_state_key"]]
        assert len(selected) == len(terminal) == 1
        assert selected[0]["candidate_id"] != terminal[0]["candidate_id"]
        assert selected[0]["selected_iteration"] == 0
        assert saved[selected[0]["state_key"] + "coefficient"][0, 0] == 1.2
        assert saved[terminal[0]["terminal_state_key"] + "coefficient"][0, 0] == 1.1
        assert selected[0]["state_key"].startswith("c")
        assert terminal[0]["terminal_state_key"].startswith("tc")
        assert set(name.removeprefix(terminal[0]["terminal_state_key"]) for name in saved
                   if name.startswith(terminal[0]["terminal_state_key"])) == {
            "coefficient", "P", "d", "Q", "state", "weighted_u", "weighted_v", "penalized_u",
            "penalized_v", "free_rows_u", "free_rows_v", "anchors_u", "anchors_v", "center_u", "center_v"}
        block = runner.read(folder / f"block-{free}.json")
        assert block["retained"] == selected[0]["candidate_id"]
        assert block["terminal_retained"] == terminal[0]["candidate_id"]
    for name, expected in result["files"].items():
        assert runner.sha(folder / name) == expected
    count = len(fixture.calls)
    # Exercise block-level reconstruction, including the terminal controls.
    (folder / "result.json").unlink()
    resumed = runner.run_task(fixture.root, 0)
    assert len(fixture.calls) == count
    assert resumed["outcomes"] == result["outcomes"]
    for key, value in saved.items():
        np.testing.assert_array_equal(runner._load_npz(folder / "states.npz")[key], value)


@pytest.mark.parametrize("checkpoints", [None, [], [1, 2], [0, 1, 1], [0, 2, 1], [0, -1], [0, True], [0, 3], [0, 1.0]])
def test_invalid_checkpoints_rejected_before_fit_or_task_creation(fixture, checkpoints):
    fixture.config.update(selection_rule="bic_checkpoint", checkpoint_iterations=checkpoints)
    with pytest.raises(ValueError, match="checkpoint"):
        runner.run_task(fixture.root, 0)
    assert not fixture.calls and not (fixture.root / "tasks").exists()


def test_real_checkpoint_charts_and_masks_survive_continuation_and_serialization(monkeypatch, tmp_path):
    """One tiny integration fit forces a chart switch; no scientific grid is run."""
    from dataclasses import replace
    import gzip
    import json
    import sparse_smart_v2 as api
    import sparse_smart_v2.anchor_solver as solver
    import sparse_smart_v2.estimator as estimator
    from sparse_smart.chart import AnchorChart
    from sparse_smart_v2.selection import bic_score
    from sparse_smart_v2.support import FreeRows

    original = solver.refine
    segments, fitted = [], []

    def refine(*args, **options):
        segments.append(options["iterations"])
        if len(segments) == 1:
            options["iterations"] = 2
        result = original(*args, **options)
        if len(segments) == 1:
            return replace(result, status="numerical_stagnation", termination_reason="numerical_stagnation",
                           last_rejection="left anchor singular value is below anchor_min")
        return result

    def switch(chart, state, free_rows, **options):
        P, d, Q = chart.reconstruct(state)
        u = 2 if chart.anchors_u[0] == 1 else 1
        v = 1 if chart.anchors_v[0] == 2 else 2
        fresh = AnchorChart(chart.n_u, chart.n_v, [u], [v], np.sign(P[[u]]), np.sign(Q[[v]]))
        masks = [np.repeat((~np.isin(getattr(fresh, f"complement_{side}"),
                                   getattr(free_rows, f"rows_{side}")))[:, None], fresh.rank, axis=1)
                 for side in ("u", "v")]
        return SimpleNamespace(chart=fresh, state=fresh.initial_state(P, d, Q),
            free_rows=FreeRows(free_rows.rows_u, free_rows.rows_v, masks[0], masks[1]),
            event=dict(old_anchors=dict(u=chart.anchors_u.tolist(), v=chart.anchors_v.tolist()),
                       new_anchors=dict(u=[u], v=[v])))

    def make_model(**options):
        model = api.SparseSMARTv2(**options)
        fitted.append(model)
        return model

    monkeypatch.setattr(solver, "refine", refine)
    monkeypatch.setattr(estimator, "reanchor", switch)
    local_api = SimpleNamespace(SparseSMARTv2=make_model, Margins=api.Margins,
                                PracticalCalibration=api.PracticalCalibration)
    x = 2 * np.eye(4)
    p, q = np.array([0., .8, .6, 0.]), np.array([0., .6, .8, 0.])
    truth = 2 * np.outer(p, q)
    data = dict(X=x, Y=x @ truth + .01 * np.eye(4), C_star=truth,
                X_validation=x, Y_validation=x @ truth)
    group = dict(p=4, q=4, margins=dict(d_lower=.1, d_upper=8., gap=.1, anchor_min=.04))
    task = dict(rank=1, initializer_source_rank=3, init_penalty=.08)
    config = dict(selection_rule="bic_checkpoint", checkpoint_iterations=[0, 1, 2, 3, 5],
        iterations=5, stationarity_tol=None, max_backtracks=20, adaptive_anchors=True,
        max_anchor_switches=16, refinement_solver="masked_anchor_projected", inverse_step=20., support_tolerance=0.)
    row, arrays, history = runner.fit_candidate(local_api, group, task, config, data,
        np.diag([5., 4., 3., 2.]), {}, 3, .001, .001, "tiny_real")
    assert row["success"], row["message"]
    assert segments == [5, 3] and row["chart_transitions"] == 1
    model = fitted[0]
    assert model.validation_history_ == []
    assert model.checkpoint_model(2).chart_.anchors_u.tolist() != model.checkpoint_model(3).chart_.anchors_u.tolist()
    for saved in row["checkpoint_scores"]:
        view = model.checkpoint_model(saved["iteration"])
        expected = bic_score(x, data["Y"], view.coefficient_, rank=1, free_directions=[3, 3],
            weighted_u=view.chart_.unpack(view.last_state_)[-2],
            weighted_v=view.chart_.unpack(view.last_state_)[-1],
            penalized_u=view.free_rows_.penalized_u, penalized_v=view.free_rows_.penalized_v)
        assert saved["selection"] == expected.as_dict()
    for prefix, iteration in (("", row["selected_iteration"]), ("terminal_", row["n_iter"])):
        view = model.checkpoint_model(iteration)
        for name in ("anchors_u", "anchors_v", "center_u", "center_v"):
            np.testing.assert_array_equal(arrays[prefix + name], getattr(view.chart_, name))
        np.testing.assert_allclose(arrays[prefix + "coefficient"], view.coefficient_)
        np.testing.assert_array_equal(arrays[prefix + "penalized_u"], view.free_rows_.penalized_u)
    runner.legacy._history(tmp_path / "history.json.gz", history)
    decoded = json.loads(gzip.decompress((tmp_path / "history.json.gz").read_bytes()))
    assert len(decoded["chart_transitions"]) == 1
    assert decoded["terminal_record"]["iteration"] == 5
