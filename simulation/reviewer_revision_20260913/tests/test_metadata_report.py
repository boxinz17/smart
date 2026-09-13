"""Synthetic JSON receipts only; no numerical models or array archives."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("metadata_report_fixture", Path(__file__).parents[1] / "metadata_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _fixture(root):
    case = dict(case_id="fitted_tiny", family="fitted_source", level=100, p=4, q=3,
                n_train=6, source_mode="fitted", n_source=100)
    tasks = [dict(case_index=0, seed=seed, variants=["full_caps"]) for seed in (1, 2)]
    config = dict(init_penalties=[.1, .3], penalties_u=[0., .01, .04], penalties_v=[0., .01, .04])
    plan = dict(cases=[case], tasks=tasks, configuration=config)
    _save(root / "plan.json", plan)
    digest = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    _save(root / "audit.json", dict(audit_passed=True, all_planned_tasks_complete=True,
        plan_sha256=digest, n_planned_tasks=2,
        tasks=[dict(task_index=i, state="complete") for i in range(2)]))
    for index, task in enumerate(tasks):
        metadata = dict(case_id=case["case_id"], seed=task["seed"], case=case,
            latent_source_rank=2, operational_source_rank=2, source_numerical_rank=3,
            source_error_frobenius=1.+2*index, source_error_operator=.5+index,
            target_population_signal=5., target_coefficient_energy=25., source_coefficient_energy=100.,
            containment_left=0., containment_right=0., containment_joint=0.,
            observed_source_singular_values=[5., 2., .1], source_internal_gaps=[3.], target_internal_gaps=[1.],
            left_angles_deg=[0., 30.], right_angles_deg=[0., 0.],
            alignment_left=dict(tail_after_top2=.1), alignment_right=dict(tail_after_top2=.2),
            source_fit=dict(selected_rank=2+index, selected_alpha=.1*index, n_refit=100,
                            candidates=[dict(selected_rank=999, validation_mse=999.)]))
        transfer = dict(success=True, selected_method="sparse_smart_v2", selected_iteration=50,
                        parameters=dict(init_penalty=.3, penalty=[0., .04]))
        endpoint = dict(success=True, selected_method="target_rrr", selected_iteration=0,
                        parameters=dict(init_penalty=0., penalty=[0., 0.]))
        _save(root / "tasks" / f"task-{index:06d}" / "result.json", dict(complete=True,
              task=task, case=case, plan_sha256=digest, metadata=metadata,
              methods=dict(v2=endpoint if index == 0 else deepcopy(transfer), v2_transfer_only=transfer)))
    return root


def test_metadata_means_mcse_and_boundary_denominators(tmp_path):
    _fixture(tmp_path)
    plan, rows, raw, selections, provenance = report.load_study(tmp_path)
    summary = report.summarize_rows(plan, rows, selections)
    source_error = summary["conditions"][0]["metrics"]["source_error_frobenius"]
    assert source_error["mean"] == 2 and source_error["mcse"] == 1
    assert source_error["n"] == source_error["n_planned"] == 2
    assert rows[0]["observed_source_retained_boundary_gap"] == pytest.approx(1.9)
    assert rows[0]["left_angles_deg.maximum"] == 30
    assert not any("candidates" in key for key in rows[0])
    assert raw[0]["metadata"]["source_fit"]["candidates"][0]["selected_rank"] == 999
    edge = next(row for row in summary["penalty_boundaries"] if row["method"] == "v2" and row["parameter"] == "penalty_u")
    assert edge["n_planned"] == 2 and edge["n_endpoint"] == 1 and edge["n_applicable"] == 1
    assert edge["lower_count"] == 1 and edge["upper_count"] == 0
    assert provenance["n_receipts"] == 2


def test_build_emits_complete_json_csv_and_tex(tmp_path):
    _fixture(tmp_path)
    prefix = tmp_path / "report" / "realized"
    result = report.build_report(tmp_path, prefix)
    assert result["n_conditions"] == 1
    for suffix in (".json", "-per-task.csv", "-condition-metrics.csv", "-penalty-boundaries.csv", "-fitted-source.tex", "-structure.tex"):
        assert Path(str(prefix)+suffix).stat().st_size > 0


@pytest.mark.parametrize("mutation", ("audit_failed", "hash", "metadata_identity", "task_identity", "grid", "missing_metadata"))
def test_invalid_receipts_do_not_produce_outputs(tmp_path, mutation):
    _fixture(tmp_path)
    path = tmp_path / ("audit.json" if mutation in ("audit_failed", "hash") else "tasks/task-000000/result.json")
    value = json.loads(path.read_text())
    if mutation == "audit_failed":
        value["audit_passed"] = False
    elif mutation == "hash":
        value["plan_sha256"] = "wrong"
    elif mutation == "metadata_identity":
        value["metadata"]["seed"] = 99
    elif mutation == "task_identity":
        value["task"]["seed"] = 99
    elif mutation == "grid":
        value["methods"]["v2_transfer_only"]["parameters"]["penalty"][1] = 99
    else:
        del value["metadata"]["source_error_operator"]
    _save(path, value)
    with pytest.raises(ValueError):
        report.build_report(tmp_path, tmp_path / "uncreated")
    assert not (tmp_path / "uncreated.json").exists()


def test_nonfinite_metadata_is_rejected_and_single_seed_has_no_mcse():
    with pytest.raises(ValueError):
        report.flatten_numeric(dict(gaps=[1., float("nan")]))
    assert report._stats([1.], 1)["mcse"] is None
