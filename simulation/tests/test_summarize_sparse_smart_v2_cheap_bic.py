"""Cheap-pilot selection, provenance, split and uncertainty tests; no fits."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cheap_bic_summary_tested", HERE.parent / "summarize_sparse_smart_v2_cheap_bic.py")
summary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary
spec.loader.exec_module(summary)


class NoMetrics(dict):
    def __getitem__(self, key):
        if key == "metrics":
            raise AssertionError("development inspected evaluation metrics")
        return super().__getitem__(key)


def candidate(initial, score, index, *, rrr=False, eligible=True):
    return NoMetrics(task_id=index, candidate_id=f"candidate-{index}", init_penalty=initial,
        fit_method="target_rrr" if rrr else "sparse_smart_v2", eligible=eligible,
        selection={"score": score} if eligible else None, classification="eligible" if eligible else "initializer_exclusion",
        termination_reason="target_rrr_closed_form" if rrr else "max_iterations", metrics="forbidden")


def record(index=0, seed=0, *, scores=None, n=10, q=2):
    scores = scores or [0., 1., 2., 3., 4., 5.]
    return dict(group=dict(group_id=f"group-{index}", seed_id=seed, n_train=n, q=q), meta={},
        rows=[candidate(initial, scores[i], i) for i, initial in enumerate(summary.INITIALIZERS)] +
             [candidate(.003, 20., 6, rrr=True)])


def full_plan(phase="develop"):
    seeds = range(5) if phase == "develop" else range(5, 10)
    groups, tasks, rank_records = [], [], {}
    phase_name = "development" if phase == "develop" else "assessment"
    for model in range(3):
        for seed in seeds:
            for protocol, count in (("low_floor", 5), ("standard_floor", 6)):
                for setting in range(count):
                    group = dict(group_id=f"m{model}-{protocol}-{setting}-s{seed}", model_id=model,
                        seed_id=seed, protocol=protocol, n_train=30+setting, sigma0=.01 if protocol == "low_floor" else setting,
                        p=40, q=20)
                    group["dataset_id"] = group["group_id"]
                    rank_records[group["dataset_id"]] = {"selected_rank": 5}
                    group["selected_target_rank"] = 5
                    group["rank_selection_sha256"] = summary.digest(rank_records[group["dataset_id"]])
                    groups.append(group)
                    for i, initial in enumerate(summary.INITIALIZERS):
                        tasks.append(dict(task_id=len(tasks), group_id=group["group_id"], rank=5, init_penalty=initial,
                            initializer_source_rank=10, free_counts=[5, 7, 10, 15, 20], include_rrr=i == 0))
    return dict(groups=groups, tasks=tasks, plan_fingerprint="frozen-plan", source_manifest_sha256="manifest",
        cheap_bic=dict(phase=phase_name, rank_records=rank_records, split_fingerprint="frozen-split"),
        configuration=dict(iterations=200, cheap_bic_phase=phase_name, init_penalties=list(summary.INITIALIZERS), penalty_pairs=[list(p) for p in summary.PENALTIES],
            selection="bic_terminal", selection_rule="bic_terminal", validation_patience=None))


def test_all_pairs_use_same_groups_and_group_equal_normalization():
    # Pair (.003,.03) wins group one, while (.1,.3) wins the second. The
    # first group's huge raw BIC scale must not dominate its normalized scale.
    records = [record(0, scores=[0, 0, 100, 100, 200, 200], n=100, q=10),
               record(1, scores=[10, 10, 0, 0, 20, 20], n=1, q=1)]
    for r in records:
        r["rows"][-1]["selection"]["score"] = 1000.
    table, selected = summary.pair_scores(records)
    assert len(table) == 15 and {row["groups"] for row in table} == {2}
    assert selected["init_penalties"] == [.003, .1]
    lookup = {tuple(row["init_penalties"]): row for row in table}
    assert lookup[(.003, .03)]["mean_normalized_bic_regret"] == 5.
    assert lookup[(.1, .3)]["mean_normalized_bic_regret"] == pytest.approx(.05)


def test_pair_ties_are_lexicographic_and_metrics_are_never_accessed():
    rows = [record(scores=[0.] * 6)]
    table, selected = summary.pair_scores(rows)
    assert selected["init_penalties"] == [.003, .03]
    assert all(row["mean_normalized_bic_regret"] == 0 for row in table)


def test_scientific_initializer_exclusions_keep_rrr_in_every_pair():
    r = record()
    for row in r["rows"][:-1]:
        row.update(eligible=False, selection=None, classification="initializer_exclusion")
    table, selected = summary.pair_scores([r])
    assert all(row["rrr_selected"] == 1 and row["mean_normalized_bic_regret"] == 0 for row in table)
    assert summary.inventory([r])["rrr_only_groups"] == 1
    assert selected["init_penalties"] == [.003, .03]


def test_no_rrr_or_eligible_candidate_is_an_error():
    r = record()
    for row in r["rows"]:
        row["eligible"] = False
    with pytest.raises(ValueError, match="no eligible"):
        summary.pair_scores([r])


@pytest.mark.parametrize("damage", ["missing_group", "wrong_seed", "duplicate_group", "missing_init", "duplicate_rrr", "budget", "independent_penalties"])
def test_full_phase_validation_rejects_unbalanced_or_changed_protocol(damage):
    plan = full_plan()
    if damage == "missing_group":
        plan["groups"].pop()
    elif damage == "wrong_seed":
        plan["groups"][0]["seed_id"] = 5
    elif damage == "duplicate_group":
        plan["groups"][1] = plan["groups"][0]
    elif damage == "missing_init":
        plan["tasks"].pop()
    elif damage == "duplicate_rrr":
        plan["tasks"][1]["include_rrr"] = True
    elif damage == "budget":
        plan["configuration"]["iterations"] = 2000
    else:
        plan["configuration"]["penalty_pairs"].append([0., .001])
    with pytest.raises(ValueError):
        summary.validate_phase(plan, "develop")


def test_phase_validation_accepts_exact_both_banks():
    assert summary.validate_phase(full_plan(), "develop") == list(range(5))
    assert summary.validate_phase(full_plan("assess"), "assess") == list(range(5, 10))


@pytest.fixture
def development(tmp_path, monkeypatch):
    root = tmp_path / "development-root"; root.mkdir()
    (root / "source-manifest.json").write_text(json.dumps({"files": {"frozen.py": "hash"}}))
    plan = full_plan()
    records = []
    for i, group in enumerate(plan["groups"]):
        r = record(i, group["seed_id"])
        r["group"] = group
        records.append(r)
    monkeypatch.setattr(summary, "read_bank", lambda *args: (plan, records, []))
    monkeypatch.setattr(summary, "audit_training_winner", lambda *args: {"passed": True})
    return root, plan, records


def test_develop_freezes_verified_pair_with_all_165_groups(development):
    root, _, _ = development
    report = summary.develop(root)
    assert report["success"], report["errors"]
    pair = summary.load_pair(report["pair_file"])
    assert pair["development_groups"] == 165
    assert pair["selected_init_penalties"] == [.003, .03]
    assert all(row["groups"] == 165 for row in pair["pair_scores"])
    assert not pair["evaluation_metrics_used"]
    before = Path(report["pair_file"]).read_bytes()
    assert summary.develop(root)["success"]
    assert Path(report["pair_file"]).read_bytes() == before


def test_missing_development_group_blocks_pair_and_emits_coverage(development, monkeypatch):
    root, plan, records = development
    monkeypatch.setattr(summary, "read_bank", lambda *args: (plan, records[:-1], [{"group_id": "missing", "error": "missing task"}]))
    report = summary.develop(root)
    assert not report["success"] and report["complete_groups"] == 164
    assert not (root / "development/selected-pair.json").exists()


def test_changed_selection_cannot_overwrite_frozen_pair(development):
    root, _, records = development
    first = summary.develop(root)
    contents = Path(first["pair_file"]).read_bytes()
    for record_ in records:
        record_["rows"][5]["selection"]["score"] = -10.
    second = summary.develop(root)
    assert not second["success"]
    assert Path(first["pair_file"]).read_bytes() == contents


def test_assessment_validates_pair_before_opening_assessment_bank(development, monkeypatch):
    root, _, _ = development
    report = summary.develop(root)
    path = Path(report["pair_file"])
    pair = json.loads(path.read_text()); pair["selected_init_penalties"] = [.1, .3]
    path.write_text(json.dumps(pair))
    monkeypatch.setattr(summary, "read_bank", lambda *args: pytest.fail("assessment bank opened before pair validation"))
    with pytest.raises(ValueError, match="fingerprint"):
        summary.assess(root / "assessment", path, make_plots=False)


def test_pair_file_rejects_unequal_pair_cohorts_even_with_consistent_hash(development):
    root, _, _ = development
    report = summary.develop(root)
    path = Path(report["pair_file"])
    pair = json.loads(path.read_text()); pair["pair_scores"][0]["groups"] = 164
    pair["artifact_fingerprint"] = summary.digest({k: v for k, v in pair.items() if k != "artifact_fingerprint"})
    path.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match="cohort"):
        summary.load_pair(path)


def test_seed_block_ci_does_not_treat_alias_groups_as_independent():
    rows = [dict(seed_id=seed, delta=float(seed)) for seed in range(5, 10) for _ in range(33)]
    stats = summary.seed_block_statistics(rows, "delta", expected_seeds=range(5, 10))
    assert stats["seed_blocks"] == 5 and stats["groups"] == 165
    assert stats["mean"] == 7.
    assert stats["se"] == pytest.approx(2.5 ** .5 / 5 ** .5)
    assert stats["ci95_high"] - 7 == pytest.approx(2.7764451051977987 * stats["se"])


def test_seed_blocks_reject_missing_or_unequal_group_coverage():
    rows = [dict(seed_id=seed, delta=1.) for seed in range(5, 10)]
    with pytest.raises(ValueError, match="incomplete"):
        summary.seed_block_statistics(rows[:-1], "delta", expected_seeds=range(5, 10))
    with pytest.raises(ValueError, match="unequal"):
        summary.seed_block_statistics(rows + rows[:1], "delta", expected_seeds=range(5, 10))


def test_exhaustive_2000_results_are_never_treated_as_200(tmp_path):
    old = dict(configuration={"iterations": 2000})
    old["plan_fingerprint"] = summary.digest(old)
    (tmp_path / "plan.json").write_text(json.dumps(old))
    result = summary.exhaustive_inventory(tmp_path, {})
    assert result["A"] == result["B"] == "unavailable_exact_200_budget"
    assert not result["performance_used"]


@pytest.fixture
def saved_task(tmp_path):
    root = tmp_path / "bank"; folder = root / "tasks/000000"; folder.mkdir(parents=True)
    task = dict(task_id=0, rank=1, group_id="g", init_penalty=.003, free_counts=[1], include_rrr=True)
    group = dict(group_id="g", n_train=20, p=4, q=3)
    plan = dict(plan_fingerprint="plan", source_manifest_sha256="source",
                configuration=dict(penalty_pairs=[list(p) for p in summary.PENALTIES]))
    meta = dict(fingerprints={"training": "input"})
    rows = []
    for i, pair in enumerate((*summary.PENALTIES, (0., 0.))):
        direct = i == 3
        score = dict(criterion="bic", method="target_rrr" if direct else "sparse_smart_v2", n=20, p=4, q=3, rank=1,
                     rss=100., model_dimension=6, support_tolerance=0., score=summary.base.bic_from_rss(100., n=20, q=3, model_dimension=6))
        rows.append(dict(candidate_id=f"c{i}", rank=1, free_directions=[4, 3] if direct else [1, 1], init_penalty=.003,
            penalty_u=pair[0], penalty_v=pair[1], execution_success=True, success=True, fit_method=score["method"],
            fit_status="completed", n_iter=0 if direct else 200, selected_iteration=0 if direct else 200,
            termination_reason="target_rrr_closed_form" if direct else "max_iterations", optimization_converged=direct,
            rrr_certificate=dict(certified=True), selection=score, metrics={"poison": "must never be read"}))
    (folder / "states.npz").write_bytes(b"hash-checked fixture; state audit tested separately")
    result = dict(schema_version=1, method="SparseSMARTv2BIC", plan_fingerprint="plan", source_manifest_sha256="source",
        task=task, group_id="g", fingerprints=meta["fingerprints"], status="complete", execution_success=True, success=True,
        validation_used_for_fit=False, selection_rule="bic_terminal", slurm=dict(job_id="12", step_id="1"), outcomes=rows,
        files={"states.npz": summary.base.sha(folder / "states.npz")})
    def seal():
        (folder / "result.json").write_text(json.dumps(result))
        status = {key: result[key] for key in ("schema_version", "method", "plan_fingerprint", "source_manifest_sha256", "task",
                                            "group_id", "execution_success", "success")}
        status.update(status="finished", result_sha256=summary.base.sha(folder / "result.json"))
        (folder / "status.json").write_text(json.dumps(status))
    seal()
    for prefix in ("process", "launcher"):
        (folder / f"{prefix}-exit-code.txt").write_text("0\n")
        (folder / f"{prefix}-status.tsv").write_text("job_id\tstep_id\texit_code\tfinished_utc\n12\t1\t0\t2026-09-14\n")
    return root, plan, group, task, meta, result, seal


def test_training_task_checks_do_not_consult_metrics(saved_task):
    root, plan, group, task, meta, _, _ = saved_task
    rows = summary.check_training_task(root, plan, group, task, meta)
    assert len(rows) == 4 and all("metrics" not in row for row in rows)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "coverage", "execution", "validation", "wrong_budget", "bad_bic"])
def test_training_task_rejects_incomplete_or_invalid_outputs(saved_task, damage):
    root, plan, group, task, meta, result, seal = saved_task
    folder = root / "tasks/000000"
    if damage == "missing":
        (folder / "result.json").unlink()
    elif damage == "corrupt":
        (folder / "states.npz").write_bytes(b"altered")
    else:
        if damage == "coverage":
            result["outcomes"].pop(0)
        elif damage == "execution":
            result["outcomes"][0]["execution_success"] = False
        elif damage == "validation":
            result["outcomes"][0]["termination_reason"] = "validation_stop"
        elif damage == "wrong_budget":
            result["outcomes"][0].update(n_iter=2000, selected_iteration=2000)
        elif damage == "bad_bic":
            result["outcomes"][0]["selection"]["score"] += 1
        seal()
    with pytest.raises((OSError, ValueError)):
        summary.check_training_task(root, plan, group, task, meta)


def test_training_task_allows_declared_numerical_exclusion(saved_task):
    root, plan, group, task, meta, result, seal = saved_task
    result["outcomes"][0].update(success=False, selection=None, fit_status="line_search_failed")
    seal()
    rows = summary.check_training_task(root, plan, group, task, meta)
    assert rows[0]["classification"] == "numerical_stagnation" and not rows[0]["eligible"]


def test_training_state_audit_does_not_load_truth_or_validation(tmp_path):
    folder = tmp_path / "tasks/000000"; folder.mkdir(parents=True)
    x = np.arange(80., dtype=float).reshape(20, 4) / 100.
    y = np.ones((20, 3))
    coefficient = np.zeros((4, 3)); coefficient[0, 0] = 1.
    # Object arrays would raise with allow_pickle=False if the audit accessed them.
    data_path = tmp_path / "data.npz"
    np.savez(data_path, X=x, Y=y, C_star=np.array([object()], dtype=object),
             X_validation=np.array([object()], dtype=object), Y_validation=np.array([object()], dtype=object))
    state_path = folder / "states.npz"
    np.savez(state_path, winner_coefficient=coefficient)
    score = summary.base.bic_score(x, y, coefficient, rank=1, direct_rrr=True).as_dict()
    row = dict(task_id=0, candidate_id="c", state_key="winner_", rank=1, fit_method="target_rrr", selection=score)
    meta = dict(files={"data.npz": {"path": str(data_path)}})
    group = dict(n_train=20, p=4, q=3)
    assert summary.audit_training_winner(tmp_path, row, meta, group)["evaluation_data_read"] is False
    row["selection"]["score"] -= 1
    with pytest.raises(ValueError, match="BIC"):
        summary.audit_training_winner(tmp_path, row, meta, group)


def test_chart_training_state_audit_checks_reconstruction_and_support(tmp_path):
    from sparse_smart_v2.support import choose_free_rows
    folder = tmp_path / "tasks/000000"; folder.mkdir(parents=True)
    x = np.arange(80., dtype=float).reshape(20, 4) / 100.
    y = np.ones((20, 3))
    chart = summary.base.AnchorChart(4, 3, [0], [0], np.eye(1), np.eye(1))
    packed = chart.pack(np.zeros(0), np.zeros(0), np.array([.8]), np.zeros((3, 1)), np.zeros((2, 1)))
    p, d, q = chart.reconstruct(packed)
    coefficient = (p*d) @ q.T
    mask = choose_free_rows(chart, (2, 2))
    state = dict(coefficient=coefficient, P=p, d=d, Q=q, state=packed,
        weighted_u=chart.unpack(packed)[-2], weighted_v=chart.unpack(packed)[-1],
        penalized_u=mask.penalized_u, penalized_v=mask.penalized_v,
        free_rows_u=mask.rows_u, free_rows_v=mask.rows_v, anchors_u=chart.anchors_u,
        anchors_v=chart.anchors_v, center_u=chart.center_u, center_v=chart.center_v)
    np.savez(folder / "states.npz", **{"winner_"+k: v for k, v in state.items()})
    np.savez(tmp_path / "data.npz", X=x, Y=y, C_star=np.array([object()], dtype=object))
    np.savez(tmp_path / "source.npz", left=np.eye(4), right=np.eye(3))
    score = summary.base.bic_score(x, y, coefficient, rank=1, free_directions=[2, 2],
        **{k: state[k] for k in ("weighted_u", "weighted_v", "penalized_u", "penalized_v")}).as_dict()
    row = dict(task_id=0, candidate_id="chart", state_key="winner_", rank=1, fit_method="sparse_smart_v2",
               free_directions=[2, 2], selection=score)
    meta = dict(files={name: {"path": str(tmp_path / name)} for name in ("data.npz", "source.npz")})
    group = dict(n_train=20, p=4, q=3, margins=dict(d_lower=.05, d_upper=12., gap=.01, anchor_min=.04))
    assert summary.audit_training_winner(tmp_path, row, meta, group)["passed"]
    state["coefficient"] *= 2
    np.savez(folder / "states.npz", **{"winner_"+k: v for k, v in state.items()})
    with pytest.raises(ValueError, match="reconstruction"):
        summary.audit_training_winner(tmp_path, row, meta, group)


def test_assessment_flow_reports_paired_metrics_and_seed_blocks(development, tmp_path, monkeypatch):
    devroot, _, _ = development
    pair_report = summary.develop(devroot)
    root = tmp_path / "assessment-root"; root.mkdir()
    (root / "source-manifest.json").write_text((devroot / "source-manifest.json").read_text())
    plan = full_plan("assess")
    plan["display_cases"] = []
    records = []
    prep = dict(groups=[])
    for i, group in enumerate(plan["groups"]):
        r = record(i, group["seed_id"]); r["group"] = group
        records.append(r)
        prep["groups"].append({"group_id": group["group_id"]})
        for experiment in range(4):
            plan["display_cases"].append(dict(case_id=f"{group['group_id']}-e{experiment}", group_id=group["group_id"],
                model_id=group["model_id"], seed_id=group["seed_id"], experiment_id=experiment, setting_index=0,
                n_train=30, rank=5, source_rank=10, sigma0=.01))
    (root / "preparation.json").write_text(json.dumps(prep))
    monkeypatch.setattr(summary, "read_bank", lambda *args: (plan, records, []))
    monkeypatch.setattr(summary.base, "load_group", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(summary.base, "audit_winner", lambda *args: {"passed": True})
    monkeypatch.setattr(summary, "timing_summary", lambda *args: {"group_timings": []})
    original_read = summary.read
    def read(path):
        if Path(path).name == "result.json":
            row = dict(candidate_id="candidate-0", metrics={k: .1 for k in summary.METRICS})
            return dict(outcomes=[row])
        return original_read(path)
    monkeypatch.setattr(summary, "read", read)
    for r in records:
        for row in r["rows"]:
            row.update(rank=5, free_directions=[5, 5], penalty_u=.001, penalty_v=.001,
                       selected_iteration=200, optimization_converged=False)
    report = summary.assess(root, pair_report["pair_file"], make_plots=False)
    assert report["success"], report["errors"]
    assert report["pair_frozen_before_assessment"]
    assert len(report["paired_seed_block_summary"]) == 3
    assert all(row["seed_blocks"] == 5 and row["groups"] == 165 and row["mean"] == 0
               for row in report["paired_seed_block_summary"])
    assert (root / "assessment/performance-curves.csv").exists()


@pytest.mark.parametrize("pair", [[.003, .03], [.1, .3]])
def test_runtime_counts_selected_task_initialization_and_rrr_once(tmp_path, monkeypatch, pair):
    plan = dict(groups=[dict(group_id="g", dataset_id="dataset", seed_id=5)], tasks=[
        dict(task_id=i, group_id="g", init_penalty=initial, include_rrr=i == 0)
        for i, initial in enumerate(summary.INITIALIZERS)])
    def read(path):
        return dict(process_cpu_seconds=10., elapsed_seconds=12., outcomes=[dict(fit_method="target_rrr",
            process_cpu_seconds=2., elapsed_seconds=3., reuse_provenance={"origin": "old"}, original_elapsed_seconds=4.)])
    monkeypatch.setattr(summary, "read", read)
    out = summary.timing_summary(tmp_path, plan, pair)
    row = out["group_timings"][0]
    assert row["C_actual_task_process_cpu_seconds"] == 60.
    assert row["D_reconstructed_process_cpu_seconds"] == (20. if pair[0] == .003 else 22.)
    assert row["D_reconstructed_elapsed_work_seconds"] == (24. if pair[0] == .003 else 27.)
    assert not out["standalone_D_runtime_measured"]
    assert out["totals"]["D_reconstructed_cpu_plus_once_per_dataset_rank_seconds"] is None


def test_missing_rrr_cpu_is_not_replaced_with_wall_time(tmp_path, monkeypatch):
    plan = dict(groups=[dict(group_id="g", dataset_id="dataset", seed_id=5)], tasks=[
        dict(task_id=i, group_id="g", init_penalty=initial, include_rrr=i == 0)
        for i, initial in enumerate(summary.INITIALIZERS)])
    monkeypatch.setattr(summary, "read", lambda path: dict(process_cpu_seconds=10., elapsed_seconds=12.,
        outcomes=[dict(fit_method="target_rrr", elapsed_seconds=3.)]))
    row = summary.timing_summary(tmp_path, plan, [.1, .3])["group_timings"][0]
    assert row["D_reconstructed_process_cpu_seconds"] is None
    assert row["D_reconstructed_elapsed_work_seconds"] == 27.
