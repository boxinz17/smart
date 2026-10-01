"""Five-arm summaries bind compact scores, preserve old endpoints, and block seeds."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
spec = importlib.util.spec_from_file_location("retained_summary_under_test", SIM / "summarize_sparse_smart_v2_retained_state_bic.py")
summary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary
spec.loader.exec_module(summary)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_json(path, data)


def seal(root, result):
    cid = result["case_id"]
    path = root / "cases" / f"{cid}.json"
    write(path, result)
    status = {k: result[k] for k in ("schema_version", "method", "case_id", "plan_fingerprint", "source_manifest_sha256", "input_sha256", "success", "candidate_inputs_fingerprint")}
    write(root / "cases" / f"{cid}.status.json", dict(status, status="finished", result_sha256=summary.sha(path)))


def endpoint(tid, name, iteration, score, error, *, direct=False):
    return dict(task_id=tid, task=dict(task_id=tid, init_penalty=.03, penalty_u=0., penalty_v=0. if direct else .01),
                endpoint=name, iteration=iteration, fit_method="target_rrr" if direct else "sparse_smart_v2",
                termination_reason="target_rrr_closed_form" if direct else "validation_stop", optimization_converged=direct,
                bic=dict(score=score, rss=1., model_dimension=5. if direct else 7.),
                metrics=dict(coefficient_rmse=error, coefficient_frobenius_squared=error ** 2 * 16,
                             training_prediction_mse=error ** 2, validation_mse=error + .1),
                result_sha256="a" * 64, states_sha256="b" * 64)


def state(base, sid, origins):
    return dict(base, state_id=sid, history_sha256="e"*64,
                state_pointer=dict(state_array=sid + "_state", chart_prefix=sid, archive="states.npz", history="history.json.gz"),
                origins=[dict(kind=kind, saved_at_iteration=saved, actual_iteration=base["iteration"]) for kind, saved in origins])


def rebuild_inventory(result):
    cs = result["candidates"]
    states = [s for c in cs for s in c["states"]]
    result["inventory"] = dict(eligible_task_ids=sorted(c["selected"]["task_id"] for c in cs),
        states_total=sum(len(c["states"]) for c in cs),
        states_by_origin=dict(Counter(kind for c in cs for s in c["states"] for kind in {o["kind"] for o in s["origins"]})),
        candidate_states_fingerprint=summary.digest([(c["selected"]["task_id"], c["states"]) for c in cs]),
        current_coverage_complete=True, all_retained_coverage_complete=True,
        expected_checkpoint_count=3, available_checkpoint_count=3, available_incumbent_checkpoint_count=2,
        unreached_checkpoint_count=4, validation_evaluation_count=6, unsaved_validation_evaluation_count=2,
        raw_saved_state_count=sum(len(s["origins"]) for s in states),
        distinct_current_state_count=sum(len(c["current_state_ids"]) for c in cs), distinct_all_retained_state_count=len(states))


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "retained"
    prior = tmp_path / "previous"
    source = root / "source/example.py"
    source.parent.mkdir(parents=True); source.write_text("# frozen scorer\n")
    write(root / "source-manifest.json", dict(files={"example.py": summary.sha(source)}))
    write(prior / "plan.json", dict(method=summary.historical.METHOD))
    write(prior / "source-manifest.json", dict(files={}))
    entries = []
    for seed in (0, 10, 11):
        cid = f"m0_e0_k0_s{seed}"
        case = dict(case_id=cid, model_id=0, experiment_id=0, setting_index=0, seed_id=seed,
                    n_train=200, p=4, q=4, rank=1, source_rank=2, sigma0=.1, free_directions=[2, 2])
        entries.append(dict(case_id=cid, case=case, source_root=str(tmp_path / "reference")))
    plan = dict(schema_version=1, method=summary.METHOD, root=str(root), selection_arms=list(summary.ARMS),
                source_manifest_sha256=summary.sha(root / "source-manifest.json"), n_cases=3, cases=entries,
                previous_retrospective_root=str(prior), previous_plan_sha256=summary.sha(prior / "plan.json"),
                previous_source_manifest_sha256=summary.sha(prior / "source-manifest.json"))
    plan["plan_fingerprint"] = summary.digest(plan)
    write(root / "plan.json", plan)
    prepared, results = [], []
    for i, entry in enumerate(entries):
        cid = entry["case_id"]
        input_path = root / "inputs" / f"{cid}.json"
        write(input_path, dict(case=entry["case"], source_root=entry["source_root"]))
        input_sha = summary.sha(input_path)
        prepared.append(dict(case_id=cid, input_sha256=input_sha))
        selected = endpoint(0, "selected", 350, -10., .30 + i*.1)
        terminal = endpoint(0, "terminal", 500, -9., .50 + i*.1)
        current = endpoint(0, "checkpoint_current", 250, -12., .20 + i*.2)
        incumbent = endpoint(0, "checkpoint_incumbent", 100, -13., .25 + i*.1)
        a = state(selected, "selected", [("selected", 500), ("checkpoint_incumbent", 500)])
        b = state(terminal, "terminal", [("terminal", 500), ("checkpoint_current", 500)])
        c = state(current, "checkpoint_250_current", [("checkpoint_current", 250)])
        d = state(incumbent, "checkpoint_250_incumbent", [("checkpoint_incumbent", 250)])
        rrr = endpoint(1, "selected", 0, -8., .4 + i*.1, direct=True)
        rrr_terminal = dict(rrr, endpoint="terminal")
        rrr_state = state(rrr, "rrr", [("rrr", 0), ("selected", 0), ("terminal", 0)])
        inventory = dict(current_coverage_complete=True, all_retained_coverage_complete=True)
        candidates = [dict(selected=selected, terminal=terminal, states=[a,b,c,d], current_state_ids=[b["state_id"], c["state_id"]], inventory=inventory),
                      dict(selected=rrr, terminal=rrr_terminal, states=[rrr_state], current_state_ids=[rrr_state["state_id"]], inventory=inventory)]
        arms = dict(zip(summary.ARMS, (selected, selected, terminal, c, d)))
        previous = dict(schema_version=1, method=summary.historical.METHOD, case=entry["case"], success=True,
                        arms={arm: arms[arm] for arm in summary.ARMS[:3]}, candidate_inputs_fingerprint="c"*64,
                        candidates=[{key: candidate[key] for key in ("selected", "terminal")} for candidate in candidates])
        prior_path, prior_status = prior / "cases" / f"{cid}.json", prior / "cases" / f"{cid}.status.json"
        write(prior_path, previous); write(prior_status, dict(result_sha256=summary.sha(prior_path)))
        write(input_path, dict(case=entry["case"], source_root=entry["source_root"],
                               previous_case_sha256=summary.sha(prior_path), previous_status_sha256=summary.sha(prior_status)))
        input_sha = summary.sha(input_path)
        prepared[-1]["input_sha256"] = input_sha
        result = dict(schema_version=1, method=summary.METHOD, case_id=cid, case=entry["case"],
            reference_subroot=entry["source_root"], plan_fingerprint=plan["plan_fingerprint"],
            source_manifest_sha256=plan["source_manifest_sha256"], input_sha256=input_sha,
            candidate_inputs_fingerprint="c"*64, success=True, complete_execution_coverage=True,
            complete_tuning_coverage=False, planned_tasks=3, eligible_tasks=2,
            counts=dict(eligible=2, initializer_exclusion=1), errors=[], candidates=candidates, arms=arms,
            elapsed_seconds=10.+i, process_cpu_seconds=5.+i,
            historical_comparison=dict(passed=True, previous_case_sha256=summary.sha(prior_path), previous_status_sha256=summary.sha(prior_status)))
        rebuild_inventory(result); seal(root, result); results.append(result)
    write(root / "preparation.json", dict(schema_version=1, method=summary.METHOD, status="complete", success=True,
        plan_sha256=summary.sha(root / "plan.json"), plan_fingerprint=plan["plan_fingerprint"],
        source_manifest_sha256=plan["source_manifest_sha256"], n_cases=3, cases=prepared))
    return root, plan, results


def test_complete_five_arm_summary_preserves_endpoints_and_hashes(campaign):
    root, _, _ = campaign
    report = summary.summarize(root, make_plots=False)
    assert report["success"], report["errors"]
    assert report["completed_cases"] == 3 and report["historical_endpoint_comparisons"] == 9
    assert report["task_counts"] == {"eligible": 6, "initializer_exclusion": 3}
    assert report["complete_execution_coverage"] and not report["complete_tuning_coverage"]
    assert len(report["minimum_bic_checks"]) == 12
    assert report["inventory_totals"]["expected_checkpoint_count"] == 9
    assert report["inventory_totals"]["available_checkpoint_count"] == 9
    assert report["inventory_totals"]["available_incumbent_checkpoint_count"] == 6
    assert report["inventory_totals"]["unreached_checkpoint_count"] == 12
    assert report["inventory_totals"]["unsaved_validation_evaluation_count"] == 6
    assert report["inventory_totals"]["validation_evaluation_count"] == 18
    assert report["inventory_totals"]["raw_saved_state_count"] == 27
    assert report["inventory_totals"]["distinct_current_state_count"] == 9
    assert report["inventory_totals"]["distinct_all_retained_state_count"] == 15
    assert report["worker_process_cpu_seconds"] == 18. and report["worker_elapsed_seconds_sum"] == 33.
    assert report["worker_timings_available_cases"] == 3
    assert report["summary_elapsed_seconds"] > 0 and report["summary_process_cpu_seconds"] > 0
    assert report["finished_utc"] >= report["started_utc"]
    assert len((root / "summary/case-runtime.csv").read_text().splitlines()) == 4
    assert all(check["passed"] for check in report["minimum_bic_checks"])
    curves = {r["arm"]: r for r in report["setting_results"]}
    assert curves[summary.ARMS[0]]["coefficient_rmse_mean"] == pytest.approx(.4)
    assert curves[summary.ARMS[3]]["coefficient_rmse_mean"] == pytest.approx(.4)
    assert curves[summary.ARMS[4]]["coefficient_rmse_mean"] == pytest.approx(.35)
    assert report["selection_counts"][summary.ARMS[4]]["selected_origin_presence"] == {"checkpoint_incumbent": 3}
    assert report["selection_counts"][summary.ARMS[2]]["last_available_current_checkpoint_selected"] == 3
    assert all("candidates" not in record for record in report["case_results"])
    assert all(summary.sha(root / "summary" / name) == expected for name, expected in report["output_hashes"].items())
    assert "validation-filtered incumbents" in (root / "summary/REPORT.md").read_text()
    assert not list((root / "summary").glob("*.npz"))


@pytest.mark.parametrize("damage", ["hash", "input", "origin", "pointer", "state_fingerprint", "state_count", "current_ids", "incumbent_as_current", "new_tie", "historical_arm", "previous_case", "coverage", "partial_winner", "lost_candidate", "missing_count", "negative_timing", "missing_timing"])
def test_damaged_case_never_supplies_a_winner(campaign, damage):
    root, _, results = campaign
    result = deepcopy(results[1]); cid = result["case_id"]
    if damage == "hash": (root / "cases" / f"{cid}.json").write_text("{}")
    elif damage == "input": (root / "inputs" / f"{cid}.json").write_text("{}")
    elif damage == "previous_case":
        p = Path(result["historical_comparison"].get("path", root.parent / "previous")) / "cases" / f"{cid}.json"
        p.write_text("{}")
    else:
        if damage == "origin": result["candidates"][0]["states"][0]["origins"][0]["actual_iteration"] = 123
        elif damage == "pointer": result["candidates"][0]["states"][0]["state_pointer"]["archive"] = "other.npz"
        elif damage == "state_fingerprint": result["inventory"]["candidate_states_fingerprint"] = "f"*64
        elif damage == "state_count": result["inventory"]["states_total"] = 99
        elif damage == "current_ids": result["candidates"][0]["current_state_ids"] = ["invented"]
        elif damage == "incumbent_as_current": result["candidates"][0]["current_state_ids"].append("checkpoint_250_incumbent")
        elif damage == "new_tie": result["arms"][summary.ARMS[4]] = result["arms"][summary.ARMS[3]]
        elif damage == "historical_arm": result["arms"][summary.ARMS[0]]["metrics"]["validation_mse"] = .001
        elif damage == "coverage": result["inventory"]["all_retained_coverage_complete"] = False
        elif damage == "partial_winner": result.update(success=False, errors=[{"error":"missing state"}])
        elif damage == "missing_count": result["inventory"].pop("unsaved_validation_evaluation_count")
        elif damage == "negative_timing": result["process_cpu_seconds"] = -1.
        elif damage == "missing_timing": result.pop("elapsed_seconds")
        elif damage == "lost_candidate":
            result["candidates"].pop()
            result["eligible_tasks"] = 1
            result["counts"] = dict(eligible=1, initializer_exclusion=2)
            rebuild_inventory(result)
        seal(root, result)
    report = summary.summarize(root, make_plots=False)
    assert not report["success"] and report["completed_cases"] == 2
    assert all(c["available_seeds"] == 2 and c["missing_seed_ids"] == [10] for c in report["setting_results"])
    assert all(report["case_results"][1]["arms"][arm] is None for arm in summary.ARMS)


def test_missing_and_honest_incomplete_results_are_explicit(campaign):
    root, _, results = campaign
    (root / "cases" / f"{results[0]['case_id']}.json").unlink()
    result = deepcopy(results[1]); result.update(success=False, errors=[{"error":"missing task"}],
        complete_execution_coverage=False, arms=dict.fromkeys(summary.ARMS), counts=dict(eligible=2, missing=1))
    result["inventory"]["all_retained_coverage_complete"] = False
    seal(root, result)
    report = summary.summarize(root, make_plots=False)
    assert report["case_counts"] == {"missing":1, "incomplete":1, "complete":1}
    assert all(r["available_seeds"] == 1 for r in report["setting_results"])
    assert summary.main(["--run-dir", str(root), "--no-plots"]) == 1


def test_seed_blocks_average_correlated_settings_before_uncertainty(campaign):
    _, _, results = campaign
    records = []
    for result in results:
        for k in (0,1):
            record = deepcopy(result)
            record["case"]["setting_index"] = k
            record["case"]["case_id"] = f"m0_e0_k{k}_s{record['case']['seed_id']}"
            record["classification"] = "complete"
            records.append(record)
    _, _, _, blocks = summary.build_tables(records)
    row = next(r for r in blocks if r["experiment_id"] == 0 and r["cohort"] == "all" and
               r["arm"] == summary.ARMS[3] and r["reference"] == summary.ARMS[0] and r["metric"] == "coefficient_rmse")
    assert row["n"] == 3 and row["settings_per_seed"] == 2
    assert row["mean"] == pytest.approx(0.) and row["se"] == pytest.approx(.1 / 3**.5)
    fresh = next(r for r in blocks if r["experiment_id"] == 0 and r["cohort"] == "seeds_10_99" and
               r["arm"] == summary.ARMS[3] and r["reference"] == summary.ARMS[0] and r["metric"] == "coefficient_rmse")
    assert fresh["seed_ids"] == [10,11] and fresh["n"] == 2
    records[2]["arms"][summary.ARMS[3]] = None
    _, _, _, blocks = summary.build_tables(records)
    row = next(r for r in blocks if r["experiment_id"] == 0 and r["cohort"] == "all" and
               r["arm"] == summary.ARMS[3] and r["reference"] == summary.ARMS[0] and r["metric"] == "coefficient_rmse")
    assert row["seed_ids"] == [0,11] and row["omitted_seed_ids"] == [10]


def test_new_ties_use_actual_iteration_then_task_and_state_id():
    a = dict(bic={"score":-10.}, iteration=50, task_id=2, state_id="b")
    b = dict(a, iteration=20, task_id=3)
    c = dict(b, task_id=2, state_id="z")
    d = dict(c, state_id="a")
    assert sorted([a,b,c,d], key=summary.new_key) == [d,c,b,a]


def test_source_tampering_and_unsafe_output(campaign):
    root, _, _ = campaign
    (root / "source/example.py").write_text("changed")
    report = summary.summarize(root, make_plots=False)
    assert not report["success"] and report["errors"][0]["classification"] == "campaign_provenance"
    with pytest.raises(ValueError, match="overwrite"): summary.summarize(root, root / "cases")
    with pytest.raises(ValueError, match="subdirectory"): summary.summarize(root, root.parent)


def test_figures_are_exported_and_hash_bound(campaign):
    root, _, _ = campaign
    report = summary.summarize(root)
    assert report["success"], report["errors"]
    for name in ("model_1_coefficient_rmse.png", "model_1_training_prediction_mse.pdf", "comparison.pdf"):
        assert (root / "summary" / name).stat().st_size > 1000
        assert name in report["output_hashes"]
