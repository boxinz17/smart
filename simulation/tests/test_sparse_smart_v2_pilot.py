"""Small operational fixtures; these tests never launch the pilot grid."""
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SIMULATION = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_tested_v2_pilot", SIMULATION / "run_sparse_smart_v2_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)
REAL_CASE_SPECS = pilot._case_specs


def test_authorized_plan_grid_and_seed_semantics():
    config = pilot._configuration()
    cases, seed_hash = pilot._case_specs()
    assert len(cases) == 40 and len(seed_hash) == 64
    assert len(config["init_penalties"]) * len(config["penalties_u"]) * len(config["penalties_v"]) == 120
    assert sorted({case["seed_id"] for case in cases}) == [3, 4, 5, 6, 7]
    assert {case["seed_id"]: case["random_seed"] for case in cases} == {
        3: 816464123, 4: 367992409, 5: 1982656189, 6: 225255238, 7: 561980200}
    for case in cases:
        assert case["n_train"] == (200 if case["model_id"] == 0 else 300)
        assert case["free_directions"] == [case["source_rank"]] * 2
        assert case["support_limits"] == [(case["p"]-case["source_rank"])*5,
                                          (case["q"]-case["source_rank"])*5]
    assert config["generator_source_rank"] == 10
    assert config["n_validation"] == 200 and config["iterations"] == 2000
    assert config["adaptive_anchors"] is True and config["max_anchor_switches"] == 16
    assert config["rrr_shortcut"] is True


def test_paper_models_experiments_match_canonical_settings_and_fitted_rank_semantics():
    cases, _ = pilot._case_specs(models=[0, 1, 2], experiments=[0, 1], seed_ids=range(30))
    from run_restricted_rrr import experiment_settings, load_experiment_seeds
    seeds = load_experiment_seeds()
    assert len(cases) == 990
    assert len({case["case_id"] for case in cases}) == 990
    for case in cases:
        setting = experiment_settings(case["model_id"], case["experiment_id"])[case["setting_index"]]
        assert (case["n_train"], case["p"], case["q"], case["rank"], case["source_rank"], case["sigma0"]) == (
            setting.n, setting.p, setting.q, setting.target_rank, setting.source_rank, setting.sigma0)
        assert case["random_seed"] == seeds[case["seed_id"]]
    largest = [case for case in cases if case["model_id"] == 2 and case["experiment_id"] == 0]
    assert sorted({case["n_train"] for case in largest}) == [500, 700, 1000, 1200, 1500]
    assert all((case["p"], case["q"], case["rank"]) == (300, 200, 5) for case in largest)
    rank_cases = [case for case in cases if case["experiment_id"] == 1]
    assert sorted({case["rank"] for case in rank_cases}) == [1, 3, 5, 7, 9, 11]
    assert pilot._configuration()["generator_target_rank"] == 5


def test_rank11_requires_explicit_policy_and_expansion_changes_only_that_setting():
    raw, _ = pilot._case_specs(models=[0, 1, 2], experiments=[0, 1], seed_ids=range(30))
    with pytest.raises(ValueError, match="rank11-policy"):
        pilot._configure_cases(raw, 10, "error")
    expanded, omitted = pilot._configure_cases(raw, 10, "expand")
    assert len(expanded) == 990 and not omitted
    for case in expanded:
        size = 11 if case["rank"] == 11 else 10
        assert case["initializer_source_rank"] == size
        assert case["source_rank"] == size
        assert case["free_directions"] == [size, size]
        assert case["support_limits"] == [(case["p"]-size)*case["rank"], (case["q"]-size)*case["rank"]]
    assert all(case["free_directions"] == [10, 10] for case in raw)  # Inputs are not rewritten.
    assert all(case["source_rank"] == 10 for case in raw)
    subset, omitted = pilot._configure_cases(raw, 10, "omit")
    assert len(subset) == 900 and len(omitted) == 90
    assert all(case["rank"] <= 9 for case in subset)
    assert {row["reason"] for row in omitted} == {"explicit_rank11_omission"}


@pytest.mark.parametrize("selectors", [
    {"models": []}, {"models": [True]}, {"models": [3]}, {"models": [0, 0]},
    {"experiments": [-1]}, {"seed_ids": [100]}, {"seed_ids": [1.5]},
    {"experiments": [0], "setting_indices": [5]},
    {"experiments": [0, 1], "setting_indices": [5]},
])
def test_invalid_case_selectors_fail_without_generation(selectors):
    with pytest.raises(ValueError):
        pilot._case_specs(**selectors)


def test_seed_only_selection_retains_original_pilot_settings():
    cases, _ = pilot._case_specs(seed_ids=[1, 0])
    assert len(cases) == 16
    assert {c["setting_index"] for c in cases if c["experiment_id"] == 2} == {2, 3}
    assert {c["setting_index"] for c in cases if c["experiment_id"] == 3} == {4, 5}
    assert [c["seed_id"] for c in cases[:2]] == [0, 1]


def test_model_iii_training_bytes_match_legacy_generator_and_validation_is_independent():
    pilot._paths()
    from external_validation_data import generate_external_validation
    from smart import generate_data
    case = pilot._case_specs(models=[2], experiments=[0], seed_ids=[0], setting_indices=[0])[0][0]
    arguments = dict(n=case["n_train"], p=case["p"], q=case["q"], sigma0=case["sigma0"],
                     sigma=.5, r_star=5, r0_star=10, random_seed=case["random_seed"])
    legacy = generate_data(**arguments)
    prepared = generate_external_validation(n_train=arguments.pop("n"), n_validation=200, **arguments)
    for name in ("X", "Y", "C0", "C_star"):
        np.testing.assert_array_equal(prepared[name], legacy[name])
    assert pilot._array_fingerprint(prepared, ("X", "Y", "C0")) == pilot._array_fingerprint(legacy, ("X", "Y", "C0"))
    assert prepared["X_validation"].shape == (200, 300)
    assert prepared["Y_validation"].shape == (200, 200)
    assert not np.array_equal(prepared["X_validation"], prepared["X"][:200])


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    import sparse_smart_v2 as api
    monkeypatch.syspath_prepend(str(SIMULATION))
    import external_validation_data
    root = tmp_path / "campaign"
    source_root = root / "source"
    frozen_runner = source_root / "simulation" / "run_sparse_smart_v2_pilot.py"
    frozen_runner.parent.mkdir(parents=True)
    frozen_runner.write_bytes((SIMULATION / "run_sparse_smart_v2_pilot.py").read_bytes())
    monkeypatch.setattr(pilot, "__file__", str(frozen_runner))
    manifest = dict(schema_version=1, source_root=str(source_root),
                    files={"simulation/run_sparse_smart_v2_pilot.py": pilot._sha(frozen_runner)})
    pilot._json(root / "source-manifest.json", manifest)
    case = dict(case_id="m0_e2_k2_s3", model_id=0, model="model1", experiment_id=2,
        experiment="exp3", setting_index=2, seed_id=3, random_seed=816464123,
        n_train=8, p=6, q=6, sigma0=.01, rank=5, source_rank=5,
        free_directions=[5, 5], support_limits=[5, 5])
    config = pilot._configuration()
    config.update(init_penalties=[.03, 3.], penalties_u=[0., .01], penalties_v=[0.],
        iterations=2, checkpoint_interval=1, validation_interval=1, n_validation=8,
        validation_patience=300, validation_min_iterations=500, rrr_shortcut=False)
    monkeypatch.setattr(pilot, "_case_specs", lambda: ([dict(case)], "fixture_seed_hash"))
    monkeypatch.setattr(pilot, "_configuration", lambda: config)
    monkeypatch.setattr(pilot, "_require_slurm", lambda root: None)
    monkeypatch.setattr(pilot, "_load_api", lambda: api)
    monkeypatch.setattr(pilot, "_verify_imports", lambda manifest: None)
    generator_calls = []

    def generator(**kwargs):
        generator_calls.append(kwargs)
        n, p, q = kwargs["n_train"], kwargs["p"], kwargs["q"]
        X = np.vstack((np.eye(p) * np.sqrt(n), np.zeros((n-p, p))))
        truth = np.diag([3., 2.5, 2., 1.5, 1., 0.])
        XV = X.copy()
        return dict(X=X, Y=X@truth, C0=np.diag([6., 5., 4., 3., 2., 1.]), C_star=truth,
                    X_validation=XV, Y_validation=XV@truth,
                    validation_seed_metadata={"seed_sequence_entropy": [kwargs["random_seed"], kwargs["seed_tag"]]})

    monkeypatch.setattr(external_validation_data, "generate_external_validation", generator)

    def initializers(api, case, config, source, data):
        records, arrays = [], {}
        for index, penalty in enumerate(config["init_penalties"]):
            valid = index == 0
            d = np.array([3., 2.5, 2., 1.5, 1.]) - .03 if valid else np.zeros(5)
            P = Q = np.eye(case["initializer_source_rank"])[:, :5]
            values = dict(P=P, Q=Q, d=d, coefficient=(P*d)@Q.T,
                          n_iter=np.zeros(case["initializer_source_rank"], dtype=int),
                          dual_gaps=np.zeros(case["initializer_source_rank"]))
            arrays.update({f"i{index}_{name}": value for name, value in values.items()})
            records.append(dict(index=index, init_penalty=penalty, eligible=valid,
                status="completed" if valid else "initialization_spectrum_failed",
                message="Fixture initializer", metadata={"theorem_certified": False},
                kkt_residual=0., converged=True))
        return records, arrays

    monkeypatch.setattr(pilot, "_preflight_initializers", initializers)
    return root, case, config, generator_calls, initializers


def test_plan_frozen_override_does_not_change_free_sets(campaign):
    root, _, _, _, _ = campaign
    value = pilot.plan(root, initializer_source_rank=6)
    assert value["cases"][0]["initializer_source_rank"] == 6
    assert value["cases"][0]["source_rank"] == 5
    assert value["cases"][0]["free_directions"] == [5, 5]
    assert (root / "work-items.tsv").read_text() == "0\n1\n2\n3\n"
    assert pilot.plan(root, initializer_source_rank=6) == value
    with pytest.raises(ValueError, match="existing plan differs"):
        pilot.plan(root)


def test_spectral_overrides_are_frozen_without_changing_anchor_or_other_settings(campaign):
    root, _, defaults, _, _ = campaign
    original_margins = dict(defaults["margins"])
    value = pilot.plan(root, d_lower=.001, spectral_gap=.0001)
    expected = dict(original_margins, d_lower=.001, gap=.0001)
    assert value["configuration"]["margins"] == expected
    assert pilot._read(root/"plan.json")["configuration"]["margins"] == expected
    assert defaults["margins"] == original_margins  # Do not mutate a shared default config.
    assert value["configuration"]["initialization_spectrum"] == "reject"
    assert value["configuration"]["margins"]["anchor_min"] == .04
    assert value["configuration"]["margins"]["d_upper"] == 12.
    assert pilot.plan(root, d_lower=.001, spectral_gap=.0001) == value
    with pytest.raises(ValueError, match="existing plan differs"):
        pilot.plan(root)


@pytest.mark.parametrize("name", ["d_lower", "spectral_gap", "anchor_min"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), -float("inf"), True, "0.001", 10**400])
def test_invalid_spectral_overrides_fail_before_writing_plan(campaign, name, value):
    root, _, _, _, _ = campaign
    with pytest.raises(ValueError, match="finite and strictly positive"):
        pilot.plan(root, **{name: value})
    assert not (root/"plan.json").exists()
    assert not (root/"work-items.tsv").exists()


@pytest.mark.parametrize("overrides,match", [
    ({"d_lower": 12.}, "less than"),
    ({"d_lower": 3.}, "exceed 4"),
    ({"spectral_gap": 3.}, "no rank-5 feasible spectrum"),
    ({"anchor_min": 1/16}, "less than 1/16"),
    ({"anchor_min": .1}, "less than 1/16"),
])
def test_spectral_box_constraints_are_checked_before_plan(campaign, overrides, match):
    root, _, _, _, _ = campaign
    with pytest.raises(ValueError, match=match):
        pilot.plan(root, **overrides)
    assert not (root/"plan.json").exists()


def test_spectral_feasibility_checks_each_fitted_rank_and_accepts_closed_box_boundary():
    config = pilot._configuration()
    config_before = json.loads(json.dumps(config))
    case = lambda rank: dict(case_id=f"rank{rank}", rank=rank)
    valid = pilot._spectral_configuration(config, [case(1), case(5)], d_lower=1., spectral_gap=2.75)
    assert valid["margins"]["gap"] == 2.75  # [12, 9.25, 6.5, 3.75, 1] fits exactly.
    with pytest.raises(ValueError, match="rank-9.*rank9"):
        pilot._spectral_configuration(config, [case(1), case(5), case(9)], d_lower=1., spectral_gap=2.75)
    # Rank one has no adjacent pair; its gap does not consume spectral width.
    assert pilot._spectral_configuration(config, [case(1)], spectral_gap=100.)["margins"]["gap"] == 100.
    assert config == config_before


def test_fixed_anchor_and_initializer_subset_freeze_the_requested_grid(campaign, monkeypatch):
    root, _, defaults, _, _ = campaign
    defaults.update(penalties_u=[0., .001, .0025, .01, .04], penalties_v=[0., .001, .0025, .01])
    original = json.loads(json.dumps(defaults))
    value = pilot.plan(root, anchor_min=.02, init_penalties=[.003])
    assert value["configuration"] == dict(original, init_penalties=[.003],
                                          margins=dict(original["margins"], anchor_min=.02))
    assert defaults == original
    assert value["n_tasks"] == 20
    assert {task["init_penalty"] for task in value["tasks"]} == {.003}
    assert {(task["penalty_u"], task["penalty_v"]) for task in value["tasks"]} == {
        (u, v) for u in original["penalties_u"] for v in original["penalties_v"]}
    assert (root/"work-items.tsv").read_text().splitlines() == [str(i) for i in range(20)]
    options = pilot._model_options(pilot._load_api(), value["cases"][0], value["configuration"], .003)
    assert options["margins"].anchor_min == .02
    assert options["calibration"].init_penalty == .003
    assert pilot.plan(root, anchor_min=.02, init_penalties=[.003]) == value
    for overrides in ({"anchor_min": .04, "init_penalties": [.003]},
                      {"anchor_min": .02, "init_penalties": [.003, .03]}):
        with pytest.raises(ValueError, match="existing plan differs"):
            pilot.plan(root, **overrides)
    assert pilot._read(root/"plan.json") == value


def test_omitted_anchor_and_initializer_options_retain_the_existing_plan(campaign):
    root, _, defaults, _, _ = campaign
    value = pilot.plan(root)
    assert value["configuration"] == defaults
    assert pilot.plan(root, anchor_min=None, init_penalties=None) == value
    assert pilot.plan(root, anchor_min=defaults["margins"]["anchor_min"],
                      init_penalties=defaults["init_penalties"]) == value


@pytest.mark.parametrize("penalties", [
    [], [0., 0.], [.003, .003], [-.001], [float("nan")], [float("inf")], [-float("inf")],
    [True], [np.bool_(False)], ["0.003"], [10**400], .003, "0.003", {.003, .03}, {"penalty": .003},
])
def test_invalid_initializer_grids_fail_before_writing_plan(campaign, penalties):
    root, _, _, _, _ = campaign
    with pytest.raises(ValueError, match="distinct finite nonnegative"):
        pilot.plan(root, init_penalties=penalties)
    assert not (root/"plan.json").exists()
    assert not (root/"work-items.tsv").exists()


def test_initializer_grid_accepts_zero_and_preserves_requested_order(campaign):
    root, _, _, _, _ = campaign
    value = pilot.plan(root, init_penalties=[.03, 0., .003])
    assert value["configuration"]["init_penalties"] == [.03, 0., .003]
    assert [task["init_penalty"] for task in value["tasks"]] == [.03, .03, 0., 0., .003, .003]
    options = pilot._model_options(pilot._load_api(), value["cases"][0], value["configuration"], 0.)
    assert options["calibration"].init_penalty == 0.


def test_explicit_full_paper_plan_counts_and_immutability(campaign, monkeypatch):
    root, _, _, _, _ = campaign
    monkeypatch.setattr(pilot, "_paths", lambda: SIMULATION.parent)
    monkeypatch.setattr(pilot, "_case_specs", REAL_CASE_SPECS)
    # Preserve real campaign grid while the fixture's Slurm/source guards stay local.
    config = dict(pilot._configuration(), init_penalties=[.003, .03, .1, .3, 1., 3.],
                  penalties_u=[0., .001, .0025, .01, .04], penalties_v=[0., .001, .0025, .01],
                  rrr_shortcut=True)
    monkeypatch.setattr(pilot, "_configuration", lambda: dict(config))
    selectors = dict(models=[0, 1, 2], experiments=[0, 1], seed_ids=range(30), rank11_policy="expand")
    value = pilot.plan(root, 10, **selectors)
    assert value["n_cases"] == 990 and value["n_tasks"] == 113850
    assert value["configuration"]["rank"] == "per_case"
    assert value["configuration"]["fitted_ranks"] == [1, 3, 5, 7, 9, 11]
    assert value["configuration"]["generator_target_rank"] == 5
    assert value["configuration"]["generator_source_rank"] == 10
    assert all(case["source_rank"] == (11 if case["rank"] == 11 else 10)
               for case in value["cases"])
    assert value["case_selection"]["rank11_policy"] == "expand"
    assert value["case_selection"]["setting_indices_by_experiment"] == {"0": [0, 1, 2, 3, 4], "1": [0, 1, 2, 3, 4, 5]}
    assert value["tasks"][-1]["task_id"] == 113849
    assert len((root / "work-items.tsv").read_text().splitlines()) == 113850
    assert value["task_deduplication"]["removed_duplicate_tasks"] == 4950
    assert len(value["task_deduplication"]["canonical_rrr_tasks"]) == 990
    assert pilot.plan(root, 10, **selectors) == value
    with pytest.raises(ValueError, match="existing plan differs"):
        pilot.plan(root, 10, **dict(selectors, rank11_policy="omit"))


def test_cli_passes_explicit_scientific_selectors(tmp_path, monkeypatch):
    observed = {}
    def planner(root, initializer, **selectors):
        observed.update(root=root, initializer=initializer, **selectors)
        return dict(n_cases=1, n_tasks=120)
    monkeypatch.setattr(pilot, "plan", planner)
    assert pilot.main(["plan", "--root", str(tmp_path), "--models", "2", "--experiments", "1",
                       "--seed-ids", "0", "1", "--setting-indices", "0", "4", "--initializer-source-rank", "10",
                       "--rank11-policy", "error"]) == 0
    assert observed == dict(root=tmp_path, initializer=10, models=[2], experiments=[1], seed_ids=[0, 1],
                            setting_indices=[0, 4], rank11_policy="error", d_lower=None, spectral_gap=None,
                            anchor_min=None, init_penalties=None, refinement_solver=None)


def test_constraint_aware_solver_is_explicit_and_frozen(campaign):
    root, case, _, _, _ = campaign
    value = pilot.plan(root, refinement_solver="masked_anchor_projected")
    assert value["configuration"]["refinement_solver"] == "masked_anchor_projected"
    api = pilot._load_api()
    options = pilot._model_options(api, value["cases"][0], value["configuration"], .003)
    assert options["refinement_solver"] == "masked_anchor_projected"
    with pytest.raises(ValueError, match="existing plan differs"):
        pilot.plan(root)
    with pytest.raises(ValueError, match="unsupported refinement_solver"):
        pilot.plan(root, refinement_solver="silent_fallback")


def test_cli_passes_spectral_bounds(tmp_path, monkeypatch):
    observed = {}
    def planner(root, initializer, **options):
        observed.update(options)
        return dict(n_cases=1, n_tasks=120)
    monkeypatch.setattr(pilot, "plan", planner)
    assert pilot.main(["plan", "--root", str(tmp_path), "--d-lower", "0.001", "--spectral-gap", "0.0001"]) == 0
    assert observed["d_lower"] == .001 and observed["spectral_gap"] == .0001


def test_cli_passes_fixed_anchor_and_initializer_subset(tmp_path, monkeypatch):
    observed = {}
    def planner(root, initializer, **options):
        observed.update(options)
        return dict(n_cases=1, n_tasks=20)
    monkeypatch.setattr(pilot, "plan", planner)
    assert pilot.main(["plan", "--root", str(tmp_path), "--anchor-min", "0.02",
                       "--init-penalties", "0.003"]) == 0
    assert observed["anchor_min"] == .02 and observed["init_penalties"] == [.003]


def test_cli_passes_constraint_aware_solver(tmp_path, monkeypatch):
    observed = {}
    def planner(root, initializer, **options):
        observed.update(options)
        return dict(n_cases=1, n_tasks=20)
    monkeypatch.setattr(pilot, "plan", planner)
    assert pilot.main(["plan", "--root", str(tmp_path), "--refinement-solver", "masked_anchor_projected"]) == 0
    assert observed["refinement_solver"] == "masked_anchor_projected"


def test_source_manifest_and_plan_tampering_fail(campaign):
    root, _, _, _, _ = campaign
    pilot.plan(root)
    value = pilot._read(root / "plan.json")
    value["configuration"]["iterations"] += 1
    pilot._json(root / "plan.json", value)
    with pytest.raises(ValueError, match="plan identity/hash"):
        pilot.prepare(root)


def test_preparation_generates_once_and_retains_legacy_fingerprints(campaign):
    root, _, _, calls, _ = campaign
    plan = pilot.plan(root)
    ready = pilot.prepare(root)
    assert ready["success"] and ready["status"] == "complete"
    assert len(calls) == 1
    assert calls[0]["n_train"] == 8 and calls[0]["n_validation"] == 8
    assert calls[0]["random_seed"] == 816464123 and calls[0]["r0_star"] == 10
    assert ready["cases"][0]["eligible_initializers"] == 1
    data, meta = pilot._read_case(root, plan["cases"][0], plan)
    assert meta["fingerprints"] == pilot._fingerprints(data)
    pilot.prepare(root)
    assert len(calls) == 1


def test_preparation_blocks_all_failed_initializers(campaign, monkeypatch):
    root, _, _, _, original = campaign
    pilot.plan(root)

    def all_bad(*args):
        records, arrays = original(*args)
        for row in records:
            row["eligible"] = False
        return records, arrays

    monkeypatch.setattr(pilot, "_preflight_initializers", all_bad)
    with pytest.raises(RuntimeError, match="No eligible initializer"):
        pilot.prepare(root)
    assert not (root / "preparation.json").exists()
    assert pilot._read(root / "preparation-preflight.json")["status"] == "blocked"


def test_reference_mismatch_withholds_preparation(campaign):
    root, _, _, _, _ = campaign
    pilot._json(root / "reference-cases.json", {"cases": {"m0_e2_s3_k2": {"fingerprints": {}}}})
    pilot.plan(root)
    with pytest.raises(ValueError, match="prior pilot reference"):
        pilot.prepare(root)
    assert not (root / "preparation.json").exists()


def test_invalid_initializer_becomes_scientific_outcome_without_fitting(campaign, monkeypatch):
    root, _, _, _, _ = campaign
    pilot.plan(root)
    pilot.prepare(root)
    monkeypatch.setattr(pilot, "_load_api", lambda: (_ for _ in ()).throw(AssertionError("must not refit")))
    result = pilot.fit(root, 2)
    assert result["execution_success"] and not result["success"]
    assert result["fit_status"] == "initialization_spectrum_failed"
    assert result["refinement_skipped"] == "initializer_failed_strict_preflight"
    assert result["avg_err"] is None and result["n_iter"] == 0
    assert pilot.fit(root, 2) == pilot._read(root / "tasks/00002/result.json")


def test_one_real_fit_reuses_initializer_and_publishes_live_states(campaign, monkeypatch):
    root, _, _, _, _ = campaign
    pilot.plan(root)
    pilot.prepare(root)
    import sparse_smart_v2.estimator as estimator
    monkeypatch.setattr(estimator, "reduced_lasso", lambda *a, **k: (_ for _ in ()).throw(AssertionError("repeated initializer")))
    result = pilot.fit(root, 0)
    assert result["execution_success"] and result["success"], result
    assert result["n_iter"] == 2
    destination = root / "tasks/00000"
    live = pilot._read(destination / "live-checkpoint.json")
    assert live["iteration"] == 2
    assert pilot._sha(destination / live["checkpoint_file"]) == live["checkpoint_sha256"]
    assert pilot._sha(destination / live["history_file"]) == live["history_sha256"]
    assert len(list(destination.glob("live-checkpoint-*.npz"))) == 2
    with np.load(destination / "states.npz", allow_pickle=False) as saved:
        assert "selected_P" in saved and "terminal_P" in saved and "checkpoint_1_state" in saved
    assert result["avg_err"] == result["selected_metrics"]["coefficient_rmse"]
    resumed = pilot.fit(root, 0)
    assert resumed == pilot._read(destination / "result.json")


def test_corrupted_data_is_execution_failure_not_scientific_failure(campaign):
    root, case, _, _, _ = campaign
    pilot.plan(root)
    pilot.prepare(root)
    (root / "cases" / case["case_id"] / "data.npz").write_bytes(b"corrupt")
    result = pilot.fit(root, 0)
    assert not result["execution_success"] and not result["success"]
    assert result["status"] == "execution_failed" and "hash mismatch" in result["message"]


def test_cli_rejects_relative_root_and_non_slurm_execution(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        pilot.main(["plan", "--root", "relative"])
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        pilot._require_slurm(tmp_path)


def test_saved_incumbent_and_checkpoints_keep_their_own_charts(tmp_path):
    from types import SimpleNamespace
    from sparse_smart.chart import AnchorChart
    old = AnchorChart(3, 3, [0], [0], np.eye(1), np.eye(1))
    new = AnchorChart(3, 3, [1], [0], np.eye(1), np.eye(1))
    P0 = np.array([[.8], [.6], [0.]])
    P1 = np.array([[.2], [.8], [np.sqrt(.32)]])
    Q = np.array([[1.], [0.], [0.]])
    x0 = old.initial_state(P0, np.array([2.]), Q)
    x1 = new.initial_state(P1, np.array([2.]), Q)
    c0 = SimpleNamespace(state=x0, selected_state=x0, chart=old, selected_chart=old,
                         iteration=0, selected_iteration=0, anchor_switches=())
    c1 = SimpleNamespace(state=x1, selected_state=x0, chart=new, selected_chart=old,
                         iteration=5, selected_iteration=0, anchor_switches=({"iteration": 3},))
    model = SimpleNamespace(chart_=new, selected_chart_=old, state_=x0, last_state_=x1,
                            checkpoints_={0: c0, 5: c1})
    arrays = pilot._state_arrays(model)
    np.testing.assert_allclose(arrays["selected_P"], P0)
    np.testing.assert_allclose(arrays["terminal_P"], P1)
    pilot._npz(tmp_path / "states.npz", arrays)
    with np.load(tmp_path / "states.npz", allow_pickle=False) as saved:
        for iteration, expected in ((0, P0), (5, P1)):
            prefix = f"checkpoint_{iteration}_terminal_"
            chart = AnchorChart(3, 3, *(saved[prefix + field] for field in
                                 ("anchors_u", "anchors_v", "center_u", "center_v")))
            np.testing.assert_allclose(chart.reconstruct(saved[f"checkpoint_{iteration}_state"])[0], expected)
        np.testing.assert_array_equal(saved["checkpoint_5_selected_anchors_u"], [0])
    # Publishing an earlier checkpoint after the model has moved charts must
    # also reconstruct it using the checkpoint geometry, not model.chart_.
    live = pilot._state_arrays(model, c0)
    np.testing.assert_allclose(live["terminal_P"], P0)
    np.testing.assert_array_equal(live["anchors_u"], [0])
    assert json.loads(json.dumps(pilot._checkpoint_metadata(c1))) == {
        "iteration": 5, "selected_iteration": 0, "anchor_switches": [{"iteration": 3}]}


def test_legacy_missing_rrr_policy_remains_opted_out(campaign):
    root, _, config, _, _ = campaign
    config.pop("rrr_shortcut")
    value = pilot.plan(root)
    options = pilot._model_options(pilot._load_api(), value["cases"][0], value["configuration"], .03)
    assert options["rrr_shortcut"] is False
    assert pilot._rrr_reason(value["cases"][0], value["configuration"], 0., 0.) is None


def test_preflight_explicitly_disables_shortcut(monkeypatch):
    from types import SimpleNamespace
    options = []
    class FakeEstimator:
        def __init__(self, **kwargs): options.append(kwargs)
        def fit(self, *args, **kwargs):
            self.success_, self.status_, self.message_, self.metadata_ = False, "initialization_spectrum_failed", "fixture", {}
    api = SimpleNamespace(SparseSMARTv2=FakeEstimator)
    monkeypatch.setattr(pilot, "_model_options", lambda *a, **k: {"rrr_shortcut": True})
    records, arrays = pilot._preflight_initializers(api, {}, {"init_penalties": [.03]}, None, {"X": None,"Y": None})
    assert options == [{"rrr_shortcut": False}]
    assert not records[0]["eligible"] and not arrays


@pytest.mark.parametrize("all_rows_free", [False, True])
def test_rrr_bypasses_failed_initializers_with_physical_artifact_and_audit(campaign, monkeypatch, all_rows_free):
    root, case, config, _, original = campaign
    config["rrr_shortcut"] = True
    if all_rows_free:
        case["free_directions"] = [6, 6]
        case["support_limits"] = [0, 0]
        config["penalties_u"], config["penalties_v"] = [.01], [.04]
    def all_bad(*args):
        records, arrays = original(*args)
        for row in records:
            row["eligible"] = False
            row["status"] = "initialization_spectrum_failed"
        return records, arrays
    monkeypatch.setattr(pilot, "_preflight_initializers", all_bad)
    import external_validation_data
    def below_floor_tied(**kwargs):
        n, p, q = kwargs["n_train"], kwargs["p"], kwargs["q"]
        X = np.vstack((np.diag([1., 2., 3., 0., 0., 0.]), np.zeros((n-p, p))))
        truth = np.diag([.02, .01, .001, 0., 0., 0.])
        return dict(X=X, Y=X@truth, C0=np.eye(p,q), C_star=truth,
                    X_validation=X.copy(), Y_validation=X@truth, validation_seed_metadata={})
    monkeypatch.setattr(external_validation_data, "generate_external_validation", below_floor_tied)
    value = pilot.plan(root, initializer_source_rank=6)
    prepared = pilot.prepare(root)
    assert prepared["success"] and prepared["cases"][0]["eligible_initializers"] == 0
    assert prepared["cases"][0]["rrr_candidates"] == 1
    import sparse_smart_v2.estimator as estimator
    monkeypatch.setattr(estimator, "reduced_lasso", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RRR does not initialize")))
    result = pilot.fit(root, 0)
    assert result["execution_success"] and result["success"], result
    assert result["fit_method"] == "target_rrr"
    assert result["initialization_preflight_bypassed"] is True and result["n_iter"] == 0
    destination = root / "tasks/00000"
    arrays = dict(np.load(destination / "states.npz", allow_pickle=False))
    assert arrays["states_schema_version"] == 3
    assert not any("anchor" in key or key.endswith("_state") for key in arrays)
    live = pilot._read(destination / "live-checkpoint.json")
    assert live["iteration"] == live["selected_iteration"] == 0
    np.testing.assert_allclose(arrays["selected_coefficient"], np.diag([.02,.01,.001,0.,0.,0.]), atol=1e-14)
    spec = importlib.util.spec_from_file_location("rrr_runner_integration_audit", SIMULATION/"audit_sparse_smart_v2_campaign.py")
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    data, meta = pilot._read_case(root, value["cases"][0], value)
    report = audit.audit_task(destination, value, {case["case_id"]: (data, {}, meta)})
    assert report["success"] and report["rrr_optimality_checked"] and report["minimum_norm_checked"]
    assert report["design_rank"] == 3
    assert report["accepted_states_checked"] == 0
    # Re-seal a null-space coefficient perturbation: fitted values are unchanged,
    # but the minimum-norm convention must still reject the artifact.
    changed = arrays["selected_coefficient"].copy()
    changed[5, 5] = .001
    left, d, right_t = np.linalg.svd(changed, full_matrices=False)
    for prefix in ("selected", "terminal"):
        arrays[f"{prefix}_coefficient"] = changed
        arrays[f"{prefix}_P"], arrays[f"{prefix}_d"], arrays[f"{prefix}_Q"] = left[:,:4], d[:4], right_t[:4].T
        result[f"{prefix}_metrics"] = pilot._metrics(changed, data)
    arrays["checkpoint_0_coefficient"] = changed
    result["avg_err"] = result["selected_metrics"]["coefficient_rmse"]
    pilot._npz(destination / "states.npz", arrays)
    result["files"]["states.npz"] = pilot._sha(destination / "states.npz")
    pilot._json(destination / "result.json", result)
    status = pilot._read(destination / "status.json")
    status["result_sha256"] = pilot._sha(destination / "result.json")
    pilot._json(destination / "status.json", status)
    with pytest.raises(ValueError, match="minimum-norm"):
        audit.audit_task(destination, value, {case["case_id"]: (data, {}, meta)})


def test_rrr_numerical_failure_retains_result_without_fake_chart_artifact(campaign, monkeypatch):
    root, _, config, _, _ = campaign
    config["rrr_shortcut"] = True
    pilot.plan(root)
    pilot.prepare(root)
    import sparse_smart_v2.estimator as estimator
    def failed_rrr(*args):
        raise np.linalg.LinAlgError("fixture RRR decomposition failure")
    monkeypatch.setattr(estimator, "target_rrr", failed_rrr)
    result = pilot.fit(root, 0)
    assert result["execution_success"] and not result["success"]
    assert result["fit_status"] == "numerical_failure"
    assert "states.npz" not in result["files"]
    assert result["avg_err"] is None
    assert pilot._read(root/"tasks/00000/status.json")["status"] == "finished"


def test_restrictive_caps_prevent_rrr_preparation_bypass(campaign, monkeypatch):
    root, case, config, _, original = campaign
    config["rrr_shortcut"] = True
    case["support_limits"] = [4, 5]
    def all_bad(*args):
        records, arrays = original(*args)
        for row in records: row["eligible"] = False
        return records, arrays
    monkeypatch.setattr(pilot, "_preflight_initializers", all_bad)
    pilot.plan(root)
    with pytest.raises(RuntimeError, match="No eligible initializer"):
        pilot.prepare(root)
    report = pilot._read(root/"preparation-preflight.json")
    assert report["cases"][0]["rrr_candidates"] == 0


@pytest.mark.parametrize("free,caps,shortcut,expected,rrr_indices", [
    ([5, 5], [5, 5], True, 115, list(range(0, 120, 20))),
    ([6, 6], [0, 0], True, 1, list(range(120))),
    ([6, 5], [0, 5], True, 91, list(range(0, 120, 4))),
    ([5, 5], [4, 5], True, 120, []),
    ([6, 5], [0, 4], True, 120, []),
    ([5, 5], [5, 5], False, 120, []),
])
def test_new_task_plan_deduplicates_only_equivalent_rrr_endpoints(
        campaign, free, caps, shortcut, expected, rrr_indices):
    root, case, config, _, _ = campaign
    case.update(free_directions=free, support_limits=caps)
    config.update(init_penalties=[.003, .03, .1, .3, 1., 3.],
                  penalties_u=[0., .001, .0025, .01, .04], penalties_v=[0., .001, .0025, .01],
                  rrr_shortcut=shortcut)
    value = pilot.plan(root)
    tasks = value["tasks"]
    assert len(tasks) == value["n_tasks"] == expected
    assert [task["task_id"] for task in tasks] == list(range(expected))
    grid = list(product(config["init_penalties"], config["penalties_u"], config["penalties_v"]))
    retained_indices = [i for i in range(120) if not rrr_indices or i not in rrr_indices[1:]]
    assert [task["grid_index"] for task in tasks] == retained_indices
    assert [(task["init_penalty"], task["penalty_u"], task["penalty_v"]) for task in tasks] == [
        grid[i] for i in retained_indices]
    assert (root/"work-items.tsv").read_text().splitlines() == [str(i) for i in range(expected)]
    aliases = value["task_deduplication"]["canonical_rrr_tasks"]
    assert len(aliases) == int(bool(rrr_indices))
    if aliases:
        assert aliases[0]["equivalent_grid_indices"] == rrr_indices
        assert aliases[0]["grid_index"] == aliases[0]["task_id"] == 0
    assert value["task_deduplication"]["expanded_task_count"] == 120
    assert value["task_deduplication"]["removed_duplicate_tasks"] == 120-expected


def test_deduplication_keeps_independent_case_rrr_fits_and_first_original_grid_index(campaign):
    _, case, config, _, _ = campaign
    config.update(rrr_shortcut=True, penalties_u=[.01, 0.])
    tasks, provenance = pilot._planned_tasks([case, dict(case, case_id="another_case")], config)
    assert [task["task_id"] for task in tasks] == list(range(6))
    assert [task["grid_index"] for task in tasks] == [0, 1, 2, 0, 1, 2]
    assert [row["task_id"] for row in provenance["canonical_rrr_tasks"]] == [1, 4]
    assert [row["equivalent_grid_indices"] for row in provenance["canonical_rrr_tasks"]] == [[1, 3], [1, 3]]


def test_frozen_legacy_full_grid_is_not_deduplicated_when_prepared_or_executed(campaign):
    root, _, config, _, _ = campaign
    config.update(init_penalties=[.003, .03, .1, .3, 1., 3.],
                  penalties_u=[0., .001, .0025, .01, .04], penalties_v=[0., .001, .0025, .01],
                  rrr_shortcut=True)
    value = pilot.plan(root)
    value.pop("task_deduplication")
    value.pop("plan_fingerprint")
    case_id = value["cases"][0]["case_id"]
    value["tasks"] = [dict(task_id=i, case_id=case_id, grid_index=i,
        init_penalty=initial, penalty_u=left, penalty_v=right)
        for i, (initial, left, right) in enumerate(product(
            config["init_penalties"], config["penalties_u"], config["penalties_v"]))]
    value["n_tasks"] = 120
    value["plan_fingerprint"] = pilot._digest(value)
    pilot._json(root/"plan.json", value)
    (root/"work-items.tsv").write_text("".join(f"{i}\n" for i in range(120)))
    frozen_bytes = (root/"plan.json").read_bytes()
    prepared = pilot.prepare(root)
    assert prepared["n_tasks"] == 120 and prepared["cases"][0]["rrr_candidates"] == 6
    result = pilot.fit(root, 100)  # Sixth initializer's previously duplicated RRR task.
    assert result["task"] == value["tasks"][100]
    assert result["success"] and result["fit_method"] == "target_rrr"
    assert result["initialization_preflight_bypassed"] is True
    assert (root/"plan.json").read_bytes() == frozen_bytes
    assert len((root/"work-items.tsv").read_text().splitlines()) == 120
