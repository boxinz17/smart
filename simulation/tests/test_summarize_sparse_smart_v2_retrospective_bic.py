"""Retrospective aggregation preserves paired seeds and rejects broken provenance."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location("retrospective_summary_under_test", Path(__file__).resolve().parents[1] / "summarize_sparse_smart_v2_retrospective_bic.py")
summary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary
spec.loader.exec_module(summary)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_json(path, value)


def seal(root, result):
    cid = result["case_id"]
    path = root / "cases" / f"{cid}.json"
    write(path, result)
    status = {k: result[k] for k in ("schema_version", "method", "case_id", "plan_fingerprint", "source_manifest_sha256", "input_sha256", "success", "candidate_inputs_fingerprint")}
    write(root / "cases" / f"{cid}.status.json", dict(status, status="finished", result_sha256=summary.sha(path)))


def winner(tid, endpoint, iteration, error, *, rrr=False):
    return dict(task_id=tid, task=dict(task_id=tid, init_penalty=.03, penalty_u=0., penalty_v=.01),
                endpoint=endpoint, iteration=iteration, fit_method="target_rrr" if rrr else "sparse_smart_v2",
                termination_reason="target_rrr_closed_form" if rrr else "validation_stop", optimization_converged=rrr,
                bic=dict(score=-10. + tid, rss=1., model_dimension=5. if rrr else 7.),
                metrics=dict(coefficient_rmse=error, coefficient_frobenius_squared=error ** 2 * 16,
                             training_prediction_mse=error ** 2, validation_mse=error + .1),
                result_sha256="a" * 64, states_sha256="b" * 64)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "retrospective"
    root.mkdir()
    source = root / "source/example.py"
    source.parent.mkdir(); source.write_text("# Frozen scorer fixture\n")
    write(root / "source-manifest.json", dict(files={"example.py": summary.sha(source)}))
    entries = []
    for seed in (0, 1, 2):
        cid = f"m0_e0_k0_s{seed}"
        case = dict(case_id=cid, model_id=0, experiment_id=0, setting_index=0, seed_id=seed,
                    n_train=200, p=4, q=4, rank=1, source_rank=2, sigma0=.1, free_directions=[2, 2])
        entries.append(dict(case_id=cid, case=case, source_root=str(tmp_path / "reference/model1-exp1")))
    plan = dict(schema_version=1, method=summary.METHOD, root=str(root), source_campaign_root=str(tmp_path / "reference"),
                source_manifest_sha256=summary.sha(root / "source-manifest.json"), cases=entries, n_cases=len(entries))
    plan["plan_fingerprint"] = summary.digest(plan)
    write(root / "plan.json", plan)
    prepared, results = [], []
    for entry in entries:
        cid, seed = entry["case_id"], entry["case"]["seed_id"]
        input_path = root / "inputs" / f"{cid}.json"
        write(input_path, dict(case_id=cid, case=entry["case"], source_root=entry["source_root"]))
        input_sha = summary.sha(input_path)
        prepared.append(dict(case_id=cid, input_sha256=input_sha))
        result = dict(schema_version=1, method=summary.METHOD, case_id=cid, case=entry["case"],
                      plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"],
                      input_sha256=input_sha, reference_subroot=entry["source_root"], reference_plan_sha256="c" * 64,
                      success=True, complete_execution_coverage=True, complete_tuning_coverage=False,
                      errors=[], counts=dict(eligible=2, initializer_exclusion=1), planned_tasks=3, eligible_tasks=2,
                      candidate_inputs_fingerprint="d" * 64,
                      arms={summary.ARMS[0]: winner(0, "selected", 50, .3 + seed * .1),
                            summary.ARMS[1]: winner(1, "selected", 0, .2 + seed * .2, rrr=True),
                            summary.ARMS[2]: winner(0, "terminal", 500, .5 + seed * .1)})
        seal(root, result); results.append(result)
    write(root / "preparation.json", dict(schema_version=1, method=summary.METHOD, plan_sha256=summary.sha(root / "plan.json"),
          plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"],
          status="complete", success=True, n_cases=len(entries), cases=prepared))
    return root, plan, results


def test_complete_summary_pairs_seeds_reports_both_endpoints_and_scientific_exclusions(campaign):
    root, plan, _ = campaign
    report = summary.summarize(root, make_plots=False)
    assert report["success"], report["errors"]
    assert report["completed_cases"] == 3 and report["complete_execution_coverage"]
    assert not report["complete_tuning_coverage"]  # Legitimate initializer exclusions.
    assert report["task_counts"] == {"eligible": 6, "initializer_exclusion": 3}
    assert report["selection_counts"][summary.ARMS[1]]["rrr_selected"] == 3
    assert report["selection_counts"][summary.ARMS[1]]["candidate_changed_from_validation"] == 3
    assert report["selection_counts"][summary.ARMS[2]]["candidate_changed_from_validation"] == 0
    assert report["selection_counts"][summary.ARMS[2]]["state_changed_from_validation"] == 3
    curve = next(r for r in report["setting_results"] if r["arm"] == summary.ARMS[0])
    assert curve["coefficient_rmse_mean"] == pytest.approx(.4)
    assert curve["coefficient_rmse_se"] == pytest.approx(.1 / 3 ** .5)
    delta = next(r for r in report["paired_differences"] if r["arm"] == summary.ARMS[1] and r["reference"] == summary.ARMS[0] and r["metric"] == "coefficient_rmse")
    assert delta["difference_mean"] == pytest.approx(0.)
    assert delta["difference_se"] == pytest.approx(.1 / 3 ** .5)
    assert delta["paired_seed_ids"] == [0, 1, 2]
    assert "not training-only BIC" in (root / "summary/REPORT.md").read_text()
    assert len(report["input_hashes"]) == 3
    assert all(summary.sha(root / "summary" / name) == digest for name, digest in report["output_hashes"].items())
    assert not list((root / "summary").glob("*.npz"))


@pytest.mark.parametrize("damage", ["result_hash", "status_identity", "input_hash", "case_identity", "endpoint", "metrics", "candidate_fingerprint", "counts", "success_lie"])
def test_corrupt_case_is_explicit_and_never_silently_paired(campaign, damage):
    root, _, results = campaign
    result = deepcopy(results[1]); cid = result["case_id"]
    if damage == "result_hash":
        (root / "cases" / f"{cid}.json").write_text("{}")
    elif damage == "status_identity":
        path = root / "cases" / f"{cid}.status.json"
        status = summary.read(path); status["case_id"] = "wrong"
        write(path, status)
    elif damage == "input_hash":
        (root / "inputs" / f"{cid}.json").write_text("{}")
    else:
        if damage == "case_identity": result["case"]["seed_id"] = 77
        elif damage == "endpoint": result["arms"][summary.ARMS[2]]["endpoint"] = "selected"
        elif damage == "metrics": result["arms"][summary.ARMS[1]]["metrics"]["coefficient_rmse"] = -1
        elif damage == "candidate_fingerprint": result["candidate_inputs_fingerprint"] = "bad"
        elif damage == "counts": result["counts"]["eligible"] = 10
        elif damage == "success_lie": result["complete_execution_coverage"] = False
        seal(root, result)
    report = summary.summarize(root, make_plots=False)
    assert not report["success"] and report["case_counts"] == {"complete": 2, "corrupt": 1}
    assert all(r["available_seeds"] == 2 and r["missing_seed_ids"] == [1] for r in report["setting_results"])
    assert all(r["paired_seed_ids"] == [0, 2] and r["requested_seeds"] == 3 for r in report["paired_differences"])
    assert len((root / "summary/per-seed-results.csv").read_text().splitlines()) == 10


def test_missing_and_incomplete_case_remain_distinct_and_partial_results_are_labelled(campaign):
    root, _, results = campaign
    (root / "cases" / f"{results[0]['case_id']}.json").unlink()
    partial = deepcopy(results[1]); partial.update(success=False, complete_execution_coverage=False, errors=[{"task_id": 2, "error": "missing task"}])
    partial["counts"] = dict(eligible=2, missing=1)
    seal(root, partial)
    report = summary.summarize(root, make_plots=False)
    assert not report["success"]
    assert report["case_counts"] == {"missing": 1, "incomplete": 1, "complete": 1}
    assert report["selection_counts"][summary.ARMS[0]]["available_cases"] == 2
    assert all(not row["complete_execution_coverage"] for row in report["setting_results"])
    assert summary.main(["--root", str(root), "--no-plots"]) == 1


def test_rrr_selected_and_terminal_at_iteration_zero_count_as_same_state():
    a, b = winner(3, "selected", 0, .1, rrr=True), winner(3, "terminal", 0, .1, rrr=True)
    assert summary.changed(a, b) == (False, False)


def test_plan_source_tampering_and_unsafe_output_are_rejected(campaign):
    root, _, _ = campaign
    (root / "source/example.py").write_text("# changed\n")
    report = summary.summarize(root, make_plots=False)
    assert not report["success"] and report["errors"][0]["classification"] == "campaign_provenance"
    with pytest.raises(ValueError, match="cannot overwrite"):
        summary.summarize(root, root / "inputs")
    with pytest.raises(ValueError, match="subdirectory"):
        summary.summarize(root, root.parent)


def test_plot_outputs_are_standalone_and_hash_bound(campaign):
    root, _, _ = campaign
    report = summary.summarize(root, make_plots=True)
    assert report["success"], report["errors"]
    for name in ("model_1_coefficient_rmse.png", "model_1_training_prediction_mse.pdf", "comparison.pdf"):
        assert (root / "summary" / name).stat().st_size > 1000
        assert name in report["output_hashes"]
