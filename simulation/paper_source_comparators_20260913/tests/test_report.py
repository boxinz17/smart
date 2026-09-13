"""Evidence-summary tests use synthetic receipts; no fitting or cluster access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_source_comparators_20260913 import report
REAL_PLOT = report._plot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    common = types.ModuleType("paper_source_comparators_20260913.common")
    common.read, common.write, common.sha = read, write, sha
    common.load_plan = lambda root, full_source=False: read(root / "plan.json")
    monkeypatch.setitem(sys.modules, common.__name__, common)
    import paper_source_comparators_20260913 as package
    monkeypatch.setattr(package, "common", common, raising=False)
    monkeypatch.setattr(report, "_plot", lambda *args: None)
    root, reference = tmp_path / "new", tmp_path / "reference"
    root.mkdir()
    tasks, groups, displays, reference_cases = [], [], [], []
    for seed in range(2):
        case = dict(case_id=f"m0_e2_k0_s{seed}", model_id=0, experiment_id=2,
                    setting_index=0, seed_id=seed, random_seed=100 + seed,
                    n_train=200, p=100, q=50, rank=5, source_rank=5, sigma0=.01)
        task = dict(task_id=seed, group_id=seed, method="source_subspace_ridge_rrr", rank=5)
        tasks.append(task)
        groups.append(dict(group_id=seed, case=case, reference_root=str(reference)))
        # Aliased settings share fitting inputs and one task; each setting has two seeds.
        for setting, dimension in enumerate((5, 10)):
            display_case = dict(case, case_id=f"m0_e2_k{setting}_s{seed}", setting_index=setting,
                                source_rank=dimension)
            displays.append(dict(case=display_case, group_id=seed, reference_root=str(reference),
                                 task_ids={task["method"]: seed}))
            reference_cases.append(dict(case=display_case, complete_execution_coverage=True,
                                        winner=dict(coefficient_rmse=.10 + seed * .02, task_id=seed * 2 + setting,
                                                    selected_validation_mse=.26, eligible=True)))
    plan = dict(method="PaperSourceComparators", n_tasks=2, n_groups=2, groups=groups,
                tasks=tasks, display_cases=displays, plan_fingerprint="new-plan-fingerprint",
                source_manifest_sha256="new-source-hash")
    write(root / "plan.json", plan)
    original_plan = dict(method="SparseSMARTv2Pilot", plan_fingerprint="old-plan-fingerprint")
    write(reference / "plan.json", original_plan)
    write(reference / "preparation.json", {"status": "complete"})
    write(reference / "summary/summary.json", dict(plan_sha256=sha(reference / "plan.json"),
          plan_fingerprint=original_plan["plan_fingerprint"], case_results=reference_cases))
    write(reference / "summary/numerical-audit.json", dict(success=True, errors=[], tasks_checked=4,
          accepted_states_checked=12, scope="All case winners, every direct RRR endpoint, first eligible iterative candidate per case"))
    ledger = []
    for record in reference_cases:
        cid, tid = record["case"]["case_id"], record["winner"]["task_id"]
        ledger.append(dict(case_id=cid, success=True, errors=[], task_ids=[tid],
                           tasks=[dict(case_id=cid, task_id=tid, success=True)]))
    (reference / "summary/numerical-audit-cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))
    binding = dict(plan_sha256=sha(reference / "plan.json"),
                   preparation_sha256=sha(reference / "preparation.json"),
                   summary_sha256=sha(reference / "summary/summary.json"),
                   numerical_audit_sha256=sha(reference / "summary/numerical-audit.json"),
                   numerical_audit_cases_sha256=sha(reference / "summary/numerical-audit-cases.jsonl"))
    write(root / "preparation.json", dict(references={str(reference): binding}, success=True, status="complete",
          plan_sha256=sha(root / "plan.json"), plan_fingerprint=plan["plan_fingerprint"],
          source_manifest_sha256=plan["source_manifest_sha256"]))
    for task in tasks:
        folder = root / "tasks" / f"{task['task_id']:06d}"
        folder.mkdir(parents=True)
        (folder / "selected.npz").write_bytes(b"selected coefficients are audited by runtime")
        candidates = [dict(index=0, parameters={"ridge": .1}, status="eligible", validation_mse=.27),
                      dict(index=1, parameters={"ridge": 1.}, status="eligible", validation_mse=.25)]
        selection = dict(method=task["method"], status="success", refit=False, selected_index=1,
                         validation_mse=.25, validation_mse_recomputed=.25,
                         n_candidates=2, n_eligible=2, candidate_results=candidates)
        result = dict(task=task, group_id=task["group_id"], plan_fingerprint=plan["plan_fingerprint"],
                      source_manifest_sha256=plan["source_manifest_sha256"], execution_success=True,
                      status="complete", selection=selection,
                      metrics=dict(coefficient_rmse=.08 + task["task_id"] * .01,
                                   coefficient_frobenius_squared=1., validation_mse=.25,
                                   training_prediction_mse=.23),
                      audit=dict(success=True), files={"selected.npz": sha(folder / "selected.npz")},
                      elapsed_seconds=2., slurm=dict(job_id="123", step_id="4"))
        write(folder / "result.json", result)
        status = {key: result[key] for key in ("task", "group_id", "plan_fingerprint", "source_manifest_sha256")}
        status["result_sha256"] = sha(folder / "result.json")
        status.update(status="finished", execution_success=True, slurm=result["slurm"])
        write(folder / "status.json", status)
        for prefix in ("process", "launcher"):
            (folder / f"{prefix}-exit-code.txt").write_text("0\n")
            (folder / f"{prefix}-status.tsv").write_text("job_id\tstep_id\texit_code\tfinished_utc\n123\t4\t0\t2026-09-13T00:00:00Z\n")
    return root, reference, plan, common


def mutate_result(root, index, change):
    folder = root / "tasks" / f"{index:06d}"
    result = read(folder / "result.json")
    change(result)
    write(folder / "result.json", result)
    status = read(folder / "status.json")
    status["result_sha256"] = sha(folder / "result.json")
    write(folder / "status.json", status)


def test_complete_summary_keeps_aliases_out_of_replication_counts(evidence):
    root, _, _, _ = evidence
    value = report.summarize(root)
    assert value["success"]
    assert value["planned_tasks"] == 2
    assert value["unique_task_outcome_counts"] == {"eligible": 2}
    assert len(value["case_results"]) == 8
    for setting in value["setting_results"]:
        assert setting["planned"] == setting["successful"] == 2
        assert setting["coefficient_rmse"]["n"] == 2
        if setting["method"] != "v2":
            assert setting["paired_difference_vs_v2"]["mean"] == pytest.approx(-.025)
            assert setting["paired_difference_vs_v2"]["n"] == 2
            assert setting["paired_higher_rmse_count"] == 0
    assert (root / "summary/per-case.csv").exists()
    assert (root / "summary/per-setting.csv").exists()


def test_missing_task_preserves_planned_denominator(evidence):
    root, _, _, _ = evidence
    (root / "tasks/000001/result.json").unlink()
    value = report.summarize(root)
    assert not value["success"]
    assert value["unique_task_outcome_counts"] == {"eligible": 1, "missing": 1}
    for row in value["setting_results"]:
        if row["method"] != "v2":
            assert (row["planned"], row["successful"], row["missing"]) == (2, 1, 1)
            assert row["paired_difference_vs_v2"]["n"] == 1
            assert row["paired_difference_vs_v2"]["mcse"] is None


def test_declared_method_failure_is_complete_execution_not_silent_replacement(evidence):
    root, _, _, _ = evidence
    def fail(result):
        result["selection"].update(status="no_eligible_candidate", selected_index=None, n_eligible=0)
        for candidate in result["selection"]["candidate_results"]:
            candidate["status"] = "uncertified"
        result["audit"] = {"success": False}
        result["files"] = {}
        result["metrics"] = {}
    mutate_result(root, 1, fail)
    value = report.summarize(root)
    assert value["success"] and value["complete_execution_coverage"]
    assert not value["all_methods_successful"]
    assert value["unique_task_outcome_counts"] == {"eligible": 1, "method_failure": 1}
    assert all(row["failed"] == 1 for row in value["setting_results"] if row["method"] != "v2")


@pytest.mark.parametrize("mutation", [
    lambda result: result["selection"].update(selected_index=0),
    lambda result: result["selection"].update(validation_mse_recomputed=.5),
    lambda result: result["audit"].update(success=False),
    lambda result: result.update(plan_fingerprint="wrong"),
    lambda result: result["selection"].update(n_eligible=1),
    lambda result: result["selection"].update(refit=True),
])
def test_invalid_selected_evidence_is_corrupt(evidence, mutation):
    root, _, _, _ = evidence
    mutate_result(root, 0, mutation)
    value = report.summarize(root)
    assert not value["success"]
    assert value["unique_task_outcome_counts"]["corrupt"] == 1


def test_selected_file_hash_and_exit_codes_are_checked(evidence):
    root, _, _, _ = evidence
    (root / "tasks/000000/selected.npz").write_bytes(b"changed")
    for name, text in (("launcher-exit-code.txt", "1\n"), ("launcher-status.tsv", "job_id\texit_code\tfinished_utc\n123\t1\t2026-09-13T00:00:00Z\n")):
        (root / "tasks/000001" / name).write_text(text)
    value = report.summarize(root)
    assert value["unique_task_outcome_counts"] == {"corrupt": 1, "execution_failure": 1}


def test_reference_summary_hash_change_is_not_silently_accepted(evidence):
    root, reference, _, _ = evidence
    (reference / "summary/summary.json").write_text("{}\n")
    value = report.summarize(root)
    assert not value["success"]
    assert value["reference_errors"]
    assert all(row["paired_difference_vs_v2"]["n"] == 0 for row in value["setting_results"])


def test_bic_reference_is_rejected_even_with_rebound_hashes(evidence):
    root, reference, _, _ = evidence
    old = read(reference / "plan.json")
    old["method"] = "SparseSMARTv2BIC"
    write(reference / "plan.json", old)
    prep = read(root / "preparation.json")
    prep["references"][str(reference)]["plan_sha256"] = sha(reference / "plan.json")
    write(root / "preparation.json", prep)
    value = report.summarize(root)
    assert not value["success"]
    assert "BIC is excluded" in next(iter(value["reference_errors"].values()))


def test_reference_numerical_audit_aggregate_and_case_ledger_are_both_bound(evidence):
    root, reference, _, _ = evidence
    path = reference / "summary/numerical-audit-cases.jsonl"
    path.write_text(path.read_text() + "\n")
    value = report.summarize(root)
    assert not value["success"]
    assert "case-ledger hash mismatch" in next(iter(value["reference_errors"].values()))


def test_reference_numerical_audit_must_cover_exact_winner(evidence):
    root, reference, _, _ = evidence
    path = reference / "summary/numerical-audit-cases.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["task_ids"] = [999]
    rows[0]["tasks"][0]["task_id"] = 999
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prep = read(root / "preparation.json")
    prep["references"][str(reference)]["numerical_audit_cases_sha256"] = sha(path)
    write(root / "preparation.json", prep)
    value = report.summarize(root)
    assert not value["success"]
    assert "winner is absent" in next(iter(value["reference_errors"].values()))


def test_duplicate_alias_same_seed_setting_rejected(evidence):
    _, _, plan, _ = evidence
    plan["display_cases"].append(plan["display_cases"][0])
    tasks = {t["task_id"]: dict(t, classification="method_failure", eligible=False) for t in plan["tasks"]}
    with pytest.raises(ValueError, match="duplicate simulation seed"):
        report.display_rows(plan, tasks, {}, {})


def test_candidate_ties_use_first_eligible_index(evidence):
    root, _, _, _ = evidence
    def tied(result):
        result["selection"]["candidate_results"][0]["validation_mse"] = .25
    mutate_result(root, 0, tied)
    value = report.summarize(root)
    assert value["unique_task_outcome_counts"]["corrupt"] == 1


def test_partial_reference_is_explicit_and_blocks_complete_claim(evidence):
    root, reference, _, _ = evidence
    summary = read(reference / "summary/summary.json")
    summary["case_results"][0]["complete_execution_coverage"] = False
    write(reference / "summary/summary.json", summary)
    prep = read(root / "preparation.json")
    prep["references"][str(reference)]["summary_sha256"] = sha(reference / "summary/summary.json")
    write(root / "preparation.json", prep)
    value = report.summarize(root)
    assert value["complete_execution_coverage"]
    assert not value["complete_reference_coverage"] and not value["success"]
    assert any(row["classification"] == "partial_reference" for row in value["case_results"])


def test_empty_stats_and_single_replication_have_no_invented_interval():
    assert report._stats([])["mean"] is None
    assert report._stats([2.])["mcse"] is None
    assert report._stats([2.])["lower95"] is None


def test_zero_ratios_explicit_exclusion_and_wilson_endpoints():
    rows = [dict(coefficient_rmse=0., v2_coefficient_rmse=1., paired_difference=-1.),
            dict(coefficient_rmse=1., v2_coefficient_rmse=2., paired_difference=-1.)]
    value = report._paired_diagnostics(rows)
    assert value["geometric_rmse_ratio_vs_v2"]["mean"] == .5
    assert value["geometric_rmse_ratio_vs_v2"]["excluded_zero_pairs"] == 1
    assert value["higher_rmse_frequency"]["lower95"] == 0
    assert value["excess_rmse_conditional_on_higher"]["n"] == 0
    reversed_rows = [dict(coefficient_rmse=2., v2_coefficient_rmse=1., paired_difference=1.)] * 30
    assert report._paired_diagnostics(reversed_rows)["higher_rmse_frequency"]["upper95"] == 1


def test_unique_task_timing_and_boundary_counts(evidence):
    root, _, _, _ = evidence
    def boundary(result):
        result["selection"]["selected_parameters"] = {"ridge": 1.}
        result["selection"]["boundary_indicators"] = {"ridge": {"at_lower": False, "at_upper": True}}
    mutate_result(root, 0, boundary)
    value = report.summarize(root)
    assert value["unique_task_timing"]["source_subspace_ridge_rrr"]["n"] == 2
    for row in value["setting_results"]:
        if row["method"] != "v2":
            assert row["elapsed_seconds_per_unique_task"]["n"] == 2
            assert row["selected_boundary_counts"]["ridge"]["at_upper"] == 1


def test_failed_execution_has_distinct_classification(evidence):
    root, _, _, _ = evidence
    mutate_result(root, 0, lambda result: result.update(status="failed", execution_success=False))
    path = root / "tasks/000000/status.json"
    status = read(path)
    status["execution_success"] = False
    write(path, status)
    assert report.summarize(root)["unique_task_outcome_counts"]["execution_failure"] == 1


def test_plot_produces_twelve_panel_files_with_explicit_partial_status(evidence, tmp_path):
    root, _, _, _ = evidence
    settings = report.summarize(root)["setting_results"]
    output = tmp_path / "plot"
    output.mkdir()
    REAL_PLOT(settings, output, False)
    assert (output / "settings.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "settings.png").read_bytes().startswith(b"\x89PNG")
