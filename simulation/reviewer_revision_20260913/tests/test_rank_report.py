"""Synthetic rank CSV/JSON summaries only; no model fits or array archives."""
from copy import deepcopy
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rank_report as report


METHODS = ("v2", "v2_transfer_only", "initializer_only", "target_ridge_rrr",
           "source_subspace_rrr", "source_subspace_ridge_rrr", "target_rrr", "oracle_subspace_rrr")


def _fixture_rows(risks=(2., 8.), references=(1., 16.)):
    case = dict(case_id="reference_tiny", family="reference", level=0, n_train=6,
                p=10, q=8, target_rank=3, source_rank=5)
    tasks = [dict(case_index=0, seed=index+1, expected_methods=list(METHODS)) for index in range(len(risks))]
    plan = dict(cases=[case], tasks=tasks, n_cases=1, n_tasks=len(tasks))
    rows = []
    for index, task in enumerate(tasks):
        for method in METHODS:
            risk = risks[index] if method in ("v2", "v2_transfer_only") else references[index]
            source = [5, 5] if method in report.SOURCE_METHODS else 5 if method in report.V2 else None
            rows.append(dict(task_index=index, case_id=case["case_id"], family="reference", level=0,
                seed=task["seed"], n_train=6, p=10, q=8, method=method,
                status="failed" if risk is None else "success", population_prediction_excess=risk,
                coefficient_rmse=None if risk is None else math.sqrt(risk), numerical_rank=3,
                selected_fitted_rank=3, selected_source_dimension=source,
                selected_method="sparse_smart_v2" if method in report.V2 else method,
                total_seconds=10., tuning_seconds=2., tuning_scope="fitting_calls_plus_required_source_preparation"))
    return plan, rows


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def _saved_study(root, *, timing=False):
    plan, rows = _fixture_rows()
    _write_json(root / "rank-plan.json", plan)
    digest = hashlib.sha256((root / "rank-plan.json").read_bytes()).hexdigest()
    _write_json(root / "rank-audit.json", dict(audit_passed=True, all_planned_tasks_complete=True,
        plan_sha256=digest, issues=[], tasks=[dict(task_index=i, case_id="reference_tiny", state="complete")
                                            for i in range(len(plan["tasks"]))]))
    _write_json(root / "rank-summary.json", dict(plan_sha256=digest, n_planned_tasks=len(plan["tasks"]),
        n_planned_method_outcomes=len(rows), status_counts=dict(Counter(row["status"] for row in rows))))
    _write_csv(root / "rank-per-replication.csv", rows)
    if timing:
        for index, task in enumerate(plan["tasks"]):
            methods = {}
            for row in (row for row in rows if row["task_index"] == index):
                methods[row["method"]] = dict(success=row["status"] == "success",
                    metrics=dict(population_prediction_excess=row["population_prediction_excess"]),
                    runtime=dict(fitting_call_seconds=2., initializer_pass_wall_seconds=.25), fitting_call_seconds=1.5)
            _write_json(root / "rank_tasks" / f"task-{index:06d}" / "result.json", dict(complete=True,
                plan_sha256=digest, task=task, case=plan["cases"][0], methods=methods,
                shared_source_decomposition_seconds=.5))
    return plan, rows


def _pair(summary, method="v2", reference="target_ridge_rrr"):
    return next(row for row in summary["paired"] if row["method"] == method and row["reference"] == reference)


def test_paired_geometric_ratio_is_not_arithmetic_ratio():
    summary = report.summarize_rows(*_fixture_rows())
    pair = _pair(summary)
    assert pair["geometric_ratio"] == pytest.approx(1.)
    assert pair["geometric_ratio"] != pytest.approx((2.+.5)/2)
    assert pair["log_ratio_mcse"] == pytest.approx(math.log(2))
    assert pair["ratio_low"] == pytest.approx(math.exp(-report.Z*math.log(2)))
    assert pair["ratio_high"] == pytest.approx(math.exp(report.Z*math.log(2)))
    assert pair["mean_difference"] == -3.5
    assert pair["difference_mcse"] == pytest.approx(4.5)
    assert pair["harm_count"] == 1 and pair["n_planned"] == pair["n_paired"] == pair["n_ratio_available"] == 2


def test_single_seed_does_not_invent_monte_carlo_interval():
    pair = _pair(report.summarize_rows(*_fixture_rows((2.,), (1.,))))
    assert pair["geometric_ratio"] == 2 and pair["mean_difference"] == 1
    assert pair["log_ratio_mcse"] is pair["ratio_low"] is pair["ratio_high"] is None
    assert pair["difference_mcse"] is None


def test_failures_and_zero_risks_keep_pair_and_ratio_denominators_distinct():
    plan, rows = _fixture_rows((2., 0., None, 0.), (1., 0., 2., 4.))
    summary = report.summarize_rows(plan, rows)
    pair = _pair(summary)
    assert pair["n_planned"] == 4 and pair["n_paired"] == 3 and pair["n_unavailable"] == 1
    assert pair["n_ratio_available"] == 1 and pair["geometric_ratio"] == 2
    assert pair["ratio_low"] is None and pair["mean_difference"] == -1
    method = next(row for row in summary["methods"] if row["method"] == "v2")
    assert method["n_failed"] == 1 and method["risk"]["n"] == 3
    assert method["tuning_seconds"]["n"] == 4  # failed attempted computation retained


def test_source_dimension_counts_exclude_endpoint_target_and_oracle():
    plan, rows = _fixture_rows()
    first = next(row for row in rows if row["method"] == "v2" and row["task_index"] == 0)
    first.update(selected_method="target_rrr", selected_source_dimension=None)
    summary = report.summarize_rows(plan, rows)
    source = [row for row in summary["selections"] if row["dimension"] == "source_dimension"]
    assert not {"target_rrr", "target_ridge_rrr", "oracle_subspace_rrr"} & {row["method"] for row in source}
    selected = next(row for row in source if row["method"] == "v2")
    assert selected["count"] == selected["n_applicable"] == 1
    assert selected["n_planned"] == selected["n_success"] == 2 and selected["n_irrelevant"] == 1
    subspace = next(row for row in source if row["method"] == "source_subspace_rrr")
    assert subspace["value"] == 5 and subspace["count"] == 2


def test_missing_task_json_cannot_promote_csv_wall_time_to_fitting_time(tmp_path):
    _saved_study(tmp_path)
    plan, rows, provenance = report.load_study(tmp_path)
    assert all(row["tuning_seconds"] is None for row in rows)
    assert all(row["total_seconds"] == 10 for row in rows)
    summary = report.summarize_rows(plan, rows)
    assert all(row["n_timing_unavailable"] == 2 and row["tuning_seconds"]["n"] == 0 for row in summary["methods"])
    assert "rank-per-replication.csv" in provenance


def test_shared_svd_is_charged_only_to_observed_subspace_methods(tmp_path):
    _saved_study(tmp_path, timing=True)
    _, rows, _ = report.load_study(tmp_path)
    for row in rows:
        method = row["method"]
        assert row["source_preparation_added_seconds"] == (.5 if method in report.SOURCE_METHODS else 0.)
        if method == "initializer_only":
            assert row["fitting_call_seconds"] is None
            assert row["tuning_seconds"] == .25 and row["tuning_scope"] == "initializer_pass_wall_time"
        elif method in ("v2", "v2_transfer_only"):
            assert row["tuning_seconds"] == 2.
        else:
            assert row["tuning_seconds"] == (2. if method in report.SOURCE_METHODS else 1.5)


@pytest.mark.parametrize("corruption", ("failed_audit", "plan_hash", "duplicate_row", "duplicate_audit_task", "timing_identity"))
def test_invalid_audit_or_row_identity_is_rejected(tmp_path, corruption):
    _saved_study(tmp_path, timing=corruption == "timing_identity")
    if corruption == "duplicate_row":
        path = tmp_path / "rank-per-replication.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows.append(deepcopy(rows[0]))
        _write_csv(path, rows)
    elif corruption == "timing_identity":
        path = tmp_path / "rank_tasks/task-000000/result.json"
        value = json.loads(path.read_text()); value["task"]["seed"] = 999
        _write_json(path, value)
    else:
        path = tmp_path / "rank-audit.json"
        value = json.loads(path.read_text())
        if corruption == "failed_audit":
            value["audit_passed"] = False
        elif corruption == "plan_hash":
            value["plan_sha256"] = "0"*64
        else:
            value["tasks"].append(deepcopy(value["tasks"][0]))
        _write_json(path, value)
    with pytest.raises(ValueError):
        report.load_study(tmp_path)
