"""Performance export checks use compact synthetic audits; no estimators run."""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_sparse_smart_campaign_curves as curves
import summarize_sparse_smart_campaign as campaign
from paper_reference import read_reference
from test_sparse_smart_campaign_summary import entry, config, write_index, audit_for, publish


@pytest.fixture(scope="module")
def references():
    return read_reference()[0]


def report_for(root, specs):
    entries = []
    for i, spec in enumerate(specs):
        planned = entry(i, options=spec.get("configuration"), inapplicable=spec.get("inapplicable", False))
        setting = planned["cell"]["simulation_setting"]
        setting.update(p=100, q=50, target_rank=5)
        if "source_rank" in spec:
            setting.update(source_rank=spec["source_rank"], suffix=f"rs={spec['source_rank']}")
        entries.append(planned)
    path, index = write_index(root, entries, version=2)
    for planned, spec in zip(entries, specs):
        if spec.get("missing"):
            continue
        audit = audit_for(index, planned, value=spec.get("value", 1.), incomplete=spec.get("partial", False),
                          launch=spec.get("launch", False), failed=spec.get("failed", False))
        audit["summary"] and audit["summary"].update(setting=planned["cell"]["simulation_setting"]["suffix"])
        publish(root, audit)
    return campaign.summarize_campaign(path)


def available(tables, cap=None):
    return [r for r in tables["budget_statistics"] if r["population"] == "available_results"
            and (cap is None or r["iteration_budget"] == cap)]


def test_pool_complete_and_partial_individual_seeds_with_sample_sd(tmp_path, references):
    report = report_for(tmp_path, [dict(value=1), dict(value=3, partial=True), dict(value=11, partial=True)])
    tables = curves.build_tables(report, references)
    row = available(tables, 2000)[0]
    assert row["n"] == 3 and row["partial_count"] == 2
    assert row["seed_ids"] == [0, 1, 2]
    assert row["coefficient_error_mean"] == 10
    assert row["coefficient_error_sample_sd"] == pytest.approx(math.sqrt(112))
    assert row["coefficient_error_mcse"] == pytest.approx(math.sqrt(112 / 3))
    strict = [r for r in tables["performance_curves"] if r["population"] == "complete_grid"][0]
    assert strict["n"] == 1 and strict["coefficient_error_mean"] == 2
    assert strict["coefficient_error_sample_sd"] is strict["coefficient_error_mcse"] is None
    assert available(tables, 500)[0]["partial_count"] == 0


def test_uses_each_settings_configured_maximum_not_best_error_cap(tmp_path, references):
    report = report_for(tmp_path, [dict(), dict(source_rank=7, configuration=config((500, 8000)))])
    report["cases"][0]["summary"]["caps"][0]["coefficient_error"] = .000001
    tables = curves.build_tables(report, references)
    rows = [r for r in tables["performance_curves"] if r["population"] == "available_results"]
    assert [(r["x"], r["iteration_budget"]) for r in rows] == [(5, 2000), (7, 8000)]
    assert rows[0]["coefficient_error_mean"] == 2
    assert len(available(tables)) == 4


def test_failed_missing_and_unusable_cases_do_not_enter_statistics(tmp_path, references):
    report = report_for(tmp_path, [dict(value=2), dict(missing=True), dict(failed=True), dict(launch=True)])
    tables = curves.build_tables(report, references)
    row = available(tables, 2000)[0]
    assert row["n"] == 1 and row["coefficient_error_mean"] == 4 and row["unavailable"] == 3
    assert len(tables["per_seed"]) == 8
    missing = [r for r in tables["per_seed"] if r["record_state"] == "missing"]
    assert all(r["coefficient_error"] is None for r in missing)


def test_inapplicable_is_gap_never_zero(tmp_path, references):
    report = report_for(tmp_path, [dict(inapplicable=True, source_rank=3)])
    tables = curves.build_tables(report, references)
    row = available(tables, 2000)[0]
    assert row["n"] == 0 and row["inapplicable"] == 1 and row["coefficient_error_mean"] is None
    points = [r for r in tables["paper_comparison"] if r["experiment"] == "exp3" and r["x"] == 3]
    assert len(points) == 6 and all(r["comparison_status"] == "inapplicable" for r in points)
    assert all(r["sparse_smart_mean"] is None for r in points)


def as_policy_report(report):
    report = deepcopy(report)
    report["schema_version"] = 3
    for row in report["cases"]:
        row.update(policy_execution_usable=row["execution_usable"], has_permitted_missing_tasks=False,
                   missing_tuning=dict(allowed_tasks=[], unavailable_tasks=[], unavailable_grid_candidate_ids=[],
                                       planned_candidate_count=2, available_candidate_count=2))
        row["execution"]["available_tasks_success"] = row["execution_usable"]
    return report


def test_policy_missing_case_enters_available_but_not_complete_or_missing_sensitivity(tmp_path, references):
    report = as_policy_report(report_for(tmp_path, [dict(value=1), dict(value=3, partial=True)]))
    row = report["cases"][1]
    task = dict(task_id="cancelled", grid_candidate_ids=[1])
    row.update(execution_usable=False, policy_execution_usable=True, has_permitted_missing_tasks=True,
               missing_tuning=dict(allowed_tasks=[task], unavailable_tasks=[task], unavailable_grid_candidate_ids=[1],
                                   planned_candidate_count=2, available_candidate_count=1))
    row["execution"].update(fit_success=False, launch_success=False, issues=[dict(kind="fit_execution")],
                            available_tasks_success=True)
    for cap in row["summary"]["caps"]:
        cap.update(coverage_complete=False, reached_candidates=1, unresolved_candidate_count=1)
    tables = curves.build_tables(report, references)
    selected = {r["population"]: r for r in tables["performance_curves"]}
    assert selected["available_results"]["n"] == 2
    assert selected["available_results"]["coefficient_error_mean"] == 4
    assert selected["available_results"]["permitted_missing_seed_count"] == 1
    assert selected["complete_grid"]["n"] == selected["excluding_permitted_missing"]["n"] == 1
    caption = curves._missing_task_caption(tables["per_seed"])
    assert caption == "Permitted missing fits: x=5, seed 1 retains 1/2 candidates."
    assert "additional validation observations" in curves._plot_comparison_caveat(0)
    assert "n=200" not in curves._plot_comparison_caveat(0)
    assert "n=200" in curves._plot_comparison_caveat(1)


def test_policy_cannot_excuse_unrelated_failure(tmp_path, references):
    report = as_policy_report(report_for(tmp_path, [dict(launch=True)]))
    report["cases"][0]["policy_execution_usable"] = True
    report["cases"][0]["execution"]["available_tasks_success"] = True
    with pytest.raises(ValueError, match="without permitted missing tasks"):
        curves.build_tables(report, references)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["cases"].append(deepcopy(r["cases"][0])), "Case count"),
    (lambda r: r["cases"][1].update(seed_id=0, random_seed=r["cases"][0]["random_seed"]), "seed_id mismatch"),
    (lambda r: r["cases"][0]["scientific"].update(status="apparently_good"), "Unknown scientific status"),
    (lambda r: r["cases"][0]["summary"]["caps"][-1].update(coefficient_error=float("nan")), "coefficient_error"),
    (lambda r: r["cases"][0]["summary"]["caps"][-1].update(selected_iteration=9000), "exceeds budget"),
    (lambda r: r["cases"][0]["summary"]["caps"][-1].update(success=False), "Complete cap has no selected estimate"),
    (lambda r: r["cases"][0].update(execution_usable=False), "Contradictory execution"),
    (lambda r: r["cases"][0]["scientific"].update(audit_passed=False), "Failed scientific audit"),
])
def test_rejects_malformed_compact_records(tmp_path, references, mutation, match):
    report = report_for(tmp_path, [dict(), dict()])
    mutation(report)
    with pytest.raises(ValueError, match=match):
        curves.build_tables(report, references)


def test_duplicate_seeds_and_mixed_configurations_rejected(tmp_path, references):
    report = report_for(tmp_path, [dict(), dict()])
    duplicate = deepcopy(report["cases"][0])
    duplicate["case_key"] = "different-key"
    report["cases"][1] = duplicate
    with pytest.raises(ValueError, match="Duplicate seed"):
        curves.build_tables(report, references)
    row = report["cases"][1]
    row.update(seed_id=1, random_seed=1001)
    row["summary"]["seed_id"] = 1
    row["configuration"]["checkpoint_interval"] = 100
    row["summary"]["configuration"]["checkpoint_interval"] = 100
    row["configuration_fingerprint"] = campaign._digest(row["configuration"])
    with pytest.raises(ValueError, match="Inconsistent setting/configuration"):
        curves.build_tables(report, references)


def test_reference_matches_exact_x_and_model_dimensions(tmp_path, references):
    report = report_for(tmp_path, [dict(source_rank=7), dict(source_rank=5)])
    tables = curves.build_tables(report, references)
    assert [r["x"] for r in available(tables, 2000)] == [5, 7]
    report["cases"][0]["simulation_setting"]["source_rank"] = 6
    with pytest.raises(ValueError, match="No exact paper reference"):
        curves.build_tables(report, references)
    report["cases"][0]["simulation_setting"]["source_rank"] = 7
    report["cases"][0]["simulation_setting"]["q"] = 100
    with pytest.raises(ValueError, match="dimensions differ"):
        curves.build_tables(report, references)


def test_export_records_provenance_and_csv_blanks(tmp_path):
    report = report_for(tmp_path / "case-root", [dict(value=1), dict(value=3, partial=True)])
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(report))
    out = tmp_path / "exports"
    manifest = curves.export_campaign(path, out, plots=False)
    assert manifest["input_summary"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["eligible_final_cases"] == 2
    assert manifest["metric_transform"] == "none"
    assert manifest["paper_reference_verification"]["csv_hash_verified"]
    for artifact in manifest["outputs"]:
        assert hashlib.sha256((out / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    with (out / "performance_curves.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    strict = next(r for r in rows if r["population"] == "complete_grid")
    assert strict["coefficient_error_mcse"] == ""
    with (out / "per_seed.csv").open() as handle:
        seeds = list(csv.DictReader(handle))
    assert len(seeds) == 4 and all(r["init_penalty"] == "0.03" for r in seeds)
    assert "standard errors" in (out / "README.md").read_text()
