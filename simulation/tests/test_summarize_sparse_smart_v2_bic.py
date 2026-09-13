"""BIC selection, paired evaluation, and corrupt-output rejection without fits."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bic_summary_under_test", HERE.parent / "summarize_sparse_smart_v2_bic.py")
summary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary
spec.loader.exec_module(summary)
from sparse_smart.chart import AnchorChart
from sparse_smart_v2.support import choose_free_rows


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_json(path, value)


def seal(root, result):
    folder = root / "tasks/000000"
    if (folder / "states.npz").exists():
        result["files"]["states.npz"] = summary.sha(folder / "states.npz")
    write(folder / "result.json", result)
    identity = {key: result[key] for key in ("schema_version", "method", "plan_fingerprint", "source_manifest_sha256",
                                            "task", "group_id", "success", "execution_success")}
    write(folder / "status.json", dict(identity, status="finished", result_sha256=summary.sha(folder / "result.json")))
    for prefix in ("process", "launcher"):
        (folder / f"{prefix}-exit-code.txt").write_text("0\n")
        columns = "job_id\tstep_id\texit_code\tfinished_utc\n"
        (folder / f"{prefix}-status.tsv").write_text(columns + "123\t7\t0\t2026-09-13T00:00:00Z\n")


@pytest.fixture
def campaign(tmp_path):
    root, reference = tmp_path / "bic", tmp_path / "reference"
    root.mkdir()
    n, p, q = 7, 4, 4
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, p))
    c = np.zeros((p, q)); c[0, 0] = 1.
    data = dict(X=x, Y=x @ c + .1 * rng.normal(size=(n, q)), C0=c,
                C_star=c, X_validation=x.copy(), Y_validation=x @ c + .2 * rng.normal(size=(n, q)))
    source = dict(left=np.eye(p), right=np.eye(q), source_singular_values=np.array([1., 0., 0., 0.]))
    case = dict(case_id="m0_e0_k0_s0", group_id="m0_e0_k0_s0", model_id=0, experiment_id=0,
                setting_index=0, seed_id=0, random_seed=1234, n_train=n, p=p, q=q, sigma0=.01,
                rank=1, source_rank=2, free_directions=[2, 2])
    group = dict(case, margins=dict(d_lower=.05, d_upper=12., gap=.01, anchor_min=.04))
    task = dict(task_id=0, group_id=case["group_id"], rank=1, initializer_source_rank=2,
                init_penalty=.03, free_counts=[2, 3], include_rrr=True)
    source_file = root / "source/example.py"
    source_file.parent.mkdir(); source_file.write_text("# Frozen fixture source\n")
    write(root / "source-manifest.json", dict(files={"example.py": summary.sha(source_file)}))
    plan = dict(root=str(root), schema_version=1, method="SparseSMARTv2BIC", reference_root=str(reference),
                source_manifest_sha256=summary.sha(root / "source-manifest.json"), groups=[group],
                display_cases=[case], tasks=[task], n_groups=1, n_tasks=1,
                configuration=dict(selection="bic_terminal", validation_patience=None, iterations=2000,
                                   penalties_u=[0., .1], penalties_v=[.1]))
    plan["plan_fingerprint"] = summary.digest(plan)
    write(root / "plan.json", plan)
    (root / "work-items.tsv").write_text("0\n")
    old_root = reference / "model1-exp1"
    old_case_root = old_root / "cases" / case["case_id"]
    old_case_root.mkdir(parents=True)
    np.savez(old_case_root / "data.npz", **data)
    np.savez(old_case_root / "source.npz", **source)
    old_case = {key: value for key, value in case.items() if key != "group_id"}
    old_meta = dict(case=old_case, fingerprints=summary.fingerprints(data),
                    files={name: summary.sha(old_case_root / name) for name in ("data.npz", "source.npz")})
    write(old_case_root / "case.json", old_meta)
    meta = dict(group=group, plan_fingerprint=plan["plan_fingerprint"],
                reference_case_json_path=str(old_case_root / "case.json"),
                reference_case_json_sha256=summary.sha(old_case_root / "case.json"),
                files={name: dict(path=str(old_case_root / name), sha256=old_meta["files"][name]) for name in old_meta["files"]},
                fingerprints=old_meta["fingerprints"], source_fingerprint=summary.array_fingerprint(source, tuple(sorted(source))))
    group_path = root / "groups" / case["group_id"] / "group.json"
    write(group_path, meta)
    write(root / "preparation.json", dict(success=True, status="complete", plan_fingerprint=plan["plan_fingerprint"],
          plan_sha256=summary.sha(root / "plan.json"), source_manifest_sha256=plan["source_manifest_sha256"],
          n_groups=1, n_tasks=1, groups=[dict(group_id=case["group_id"], group_json_sha256=summary.sha(group_path))]))
    write(old_root / "plan.json", {"fixture": "legacy"})
    old_plan_sha = summary.sha(old_root / "plan.json")
    write(old_root / "preparation.json", dict(success=True, status="complete", plan_sha256=old_plan_sha,
          cases=[dict(case_id=case["case_id"], case_json_sha256=summary.sha(old_case_root / "case.json"))]))
    legacy_winner = dict(task_id=7, coefficient_rmse=.05, coefficient_frobenius_squared=.04,
                         training_prediction_mse=.03, fit_method="sparse_smart_v2", init_penalty=.03,
                         penalty_u=0., penalty_v=.1, selected_iteration=50, n_iter=500,
                         termination_reason="validation_stop", optimization_converged=False, elapsed_seconds=1.)
    write(old_root / "summary/summary.json", dict(success=True, complete_execution_coverage=True,
          plan_sha256=old_plan_sha, case_results=[dict(case=old_case, winner=legacy_winner)]))
    write(old_root / "summary/numerical-audit.json", dict(success=True))
    write(old_root / "summary/postprocess.json", dict(success=True,
          summary_sha256=summary.sha(old_root / "summary/summary.json"), audit_sha256=summary.sha(old_root / "summary/numerical-audit.json")))
    write(reference / "campaign-index.json", dict(roots=[dict(root=str(old_root))]))
    rows, states = [], {}
    chart = AnchorChart(p, q, [0], [0], np.eye(1), np.eye(1))
    for free in task["free_counts"]:
        for penalty in plan["configuration"]["penalties_u"]:
            prefix = f"c{len(rows):04d}_"
            d = np.array([.8 + penalty])
            state = chart.pack(np.zeros(0), np.zeros(0), d, np.zeros((p-1, 1)), np.zeros((q-1, 1)))
            P, d, Q = chart.reconstruct(state)
            coefficient = (P * d) @ Q.T
            mask = choose_free_rows(chart, (free, free))
            arrays = dict(coefficient=coefficient, P=P, d=d, Q=Q, state=state,
                weighted_u=chart.unpack(state)[-2], weighted_v=chart.unpack(state)[-1],
                penalized_u=mask.penalized_u, penalized_v=mask.penalized_v,
                free_rows_u=mask.rows_u, free_rows_v=mask.rows_v, anchors_u=chart.anchors_u,
                anchors_v=chart.anchors_v, center_u=chart.center_u, center_v=chart.center_v)
            score = summary.bic_score(x, data["Y"], coefficient, rank=1, free_directions=(free, free),
                weighted_u=arrays["weighted_u"], weighted_v=arrays["weighted_v"],
                penalized_u=arrays["penalized_u"], penalized_v=arrays["penalized_v"]).as_dict()
            error = coefficient - c
            metrics = dict(coefficient_rmse=float(np.linalg.norm(error)/4), coefficient_frobenius_squared=float(np.sum(error**2)),
                training_prediction_mse=float(np.mean((x @ error)**2)), validation_mse=float(np.mean((data["Y_validation"]-x@coefficient)**2)))
            rows.append(dict(candidate_id=f"c{len(rows)}", rank=1, free_directions=[free, free], init_penalty=.03,
                penalty_u=penalty, penalty_v=.1, success=True, execution_success=True, fit_method="sparse_smart_v2",
                fit_status="completed", termination_reason="max_iterations", n_iter=2000, selected_iteration=2000,
                optimization_converged=False, elapsed_seconds=.1, selection=score, metrics=metrics, state_key=prefix))
            states.update({prefix+name: value for name, value in arrays.items()})
    coefficient = c * .5  # A distinct direct candidate; selection never uses its truth error.
    error = coefficient - c
    prefix = "c0004_"
    states[prefix+"coefficient"] = coefficient
    rows.append(dict(candidate_id="rrr", rank=1, free_directions=[p,q], init_penalty=.03, penalty_u=0., penalty_v=0.,
        success=True, execution_success=True, fit_method="target_rrr", fit_status="completed", n_iter=0, selected_iteration=0,
        termination_reason="target_rrr_closed_form", optimization_converged=True, elapsed_seconds=.1, state_key=prefix,
        rrr_certificate=dict(certified=True),
        selection=summary.bic_score(x,data["Y"],coefficient,rank=1,direct_rrr=True).as_dict(),
        metrics=dict(coefficient_rmse=float(np.linalg.norm(error)/4), coefficient_frobenius_squared=float(np.sum(error**2)),
        training_prediction_mse=float(np.mean((x@error)**2)), validation_mse=float(np.mean((data["Y_validation"]-x@coefficient)**2)))))
    folder = root / "tasks/000000"; folder.mkdir(parents=True)
    np.savez(folder / "states.npz", **states)
    result = dict(schema_version=1, method="SparseSMARTv2BIC", plan_fingerprint=plan["plan_fingerprint"],
        source_manifest_sha256=plan["source_manifest_sha256"], task=task, group_id=case["group_id"],
        fingerprints=meta["fingerprints"], slurm=dict(job_id="123",step_id="7"), status="complete", success=True,
        execution_success=True, files={}, outcomes=rows, validation_used_for_fit=False, selection_rule="bic_terminal")
    seal(root, result)
    return root, reference, plan, result


def test_complete_aggregation_audits_states_and_produces_paired_tables(campaign):
    root, _, plan, result = campaign
    report = summary.summarize(root, make_plots=False)
    assert report["success"], report["errors"]
    assert report["complete_execution_coverage"] and report["complete_tuning_coverage"]
    assert report["outcome_counts"] == {"eligible": 5}
    expected = min(result["outcomes"], key=lambda row: (row["selection"]["score"], row["candidate_id"]))
    assert report["case_results"][0]["fixed_bic"]["candidate_id"] == expected["candidate_id"]
    assert report["audited_unique_winners"] == 1
    assert not report["truth_used_for_selection"] and not report["validation_used_for_selection"]
    paired = report["paired_differences"][0]
    assert paired["paired_seeds"] == 1 and paired["difference_se"] is None
    assert paired["difference_mean"] == pytest.approx(expected["metrics"]["coefficient_rmse"] - .05)
    assert not list((root / "summary").glob("*.npz"))


@pytest.mark.parametrize("damage", ["missing", "hash", "coverage", "validation_stop", "checkpoint", "bad_bic", "status_identity", "launcher"])
def test_missing_corrupt_or_validation_dependent_candidates_prevent_success(campaign, damage):
    root, _, _, result = campaign
    folder = root / "tasks/000000"
    if damage == "missing":
        (folder / "result.json").unlink()
    elif damage == "hash":
        (folder / "states.npz").write_bytes(b"corrupt")
    elif damage == "launcher":
        (folder / "launcher-exit-code.txt").write_text("7\n")
    elif damage == "status_identity":
        status = summary.read(folder / "status.json"); status["task"]["rank"] = 8
        write(folder / "status.json", status)
    else:
        if damage == "coverage":
            result["outcomes"].pop()
        elif damage == "validation_stop":
            result["outcomes"][0]["termination_reason"] = "validation_stop"
        elif damage == "checkpoint":
            result["outcomes"][0]["selected_iteration"] = 50
        elif damage == "bad_bic":
            result["outcomes"][0]["selection"]["score"] -= 10.
        seal(root, result)
    report = summary.summarize(root, make_plots=False)
    assert not report["success"] and not report["complete_execution_coverage"]
    assert report["errors"]


def test_selected_state_corruption_is_not_silently_replaced_with_another_winner(campaign):
    root, _, _, result = campaign
    winner = min(result["outcomes"], key=lambda row: row["selection"]["score"])
    folder = root / "tasks/000000"
    states = summary.load_npz(folder / "states.npz")
    states[winner["state_key"]+"coefficient"] *= 2
    np.savez(folder / "states.npz", **states)
    seal(root, result)  # Self-consistent hashes do not excuse numerical disagreement.
    report = summary.summarize(root, make_plots=False)
    assert not report["success"] and report["complete_execution_coverage"]
    assert report["case_results"][0]["fixed_bic"] is None
    assert any("selected-state audit" in item.get("error", "") for item in report["errors"])


def test_expected_initializer_exclusion_does_not_require_refit(campaign):
    root, _, _, result = campaign
    loser = max(result["outcomes"][:-1], key=lambda row: row["selection"]["score"])
    loser.update(success=False, fit_status="initialization_outside_domain", selection=None, metrics=None,
                 n_iter=0, selected_iteration=None, state_key=None)
    seal(root, result)
    report = summary.summarize(root, make_plots=False)
    assert report["success"], report["errors"]
    assert report["complete_execution_coverage"] and not report["complete_tuning_coverage"]
    assert report["outcome_counts"]["initializer_exclusion"] == 1


def test_reference_data_mismatch_fails_pairing(campaign):
    root, reference, _, _ = campaign
    path = reference / "model1-exp1/cases/m0_e0_k0_s0/case.json"
    meta = summary.read(path); meta["fingerprints"]["training_observed_input_fingerprint"] = "different"
    write(path, meta)
    report = summary.summarize(root, make_plots=False)
    assert not report["success"] and report["errors"]


def test_selection_ignores_evaluation_error_and_fixed_cases_accept_matching_rrr():
    rows = []
    for index, (rank, free, score, error, method) in enumerate(((1,[2,2],-10.,100.,"sparse_smart_v2"),
                (2,[3,3],-20.,200.,"sparse_smart_v2"), (1,[10,8],-11.,300.,"target_rrr"))):
        rows.append(dict(task_id=index, candidate_id=str(index), rank=rank, free_directions=free,
                         selection=dict(score=score), metrics=dict(coefficient_rmse=error,validation_mse=error),
                         eligible=True, fit_method=method))
    assert summary.select_winner(rows)["candidate_id"] == "1"
    assert summary.select_winner(rows, dict(rank=1,free_directions=[2,2]))["candidate_id"] == "2"
    rows[2]["selection"]["score"] = -10.
    assert summary.select_winner(rows, dict(rank=1,free_directions=[2,2]))["candidate_id"] == "0"


def test_paired_standard_error_is_for_differences_not_independent_error_bars():
    records = []
    for seed, (new, old) in enumerate(((2.,1.), (4.,3.), (6.,5.))):
        case = dict(case_id=f"s{seed}",group_id=f"g{seed}",model_id=0,experiment_id=0,setting_index=0,seed_id=seed,n_train=20)
        def winner(value):
            return dict(metrics=dict(coefficient_rmse=value),rank=1,free_directions=[2,2],fit_method="sparse_smart_v2",
                        selected_iteration=10,selection=dict(score=0.))
        records.append(dict(case=case, fixed_bic=winner(new), automatic_bic=winner(new), previous_validation=winner(old),
                            complete_execution_coverage=True))
    _, curves, paired = summary.build_tables(records)
    assert curves[0]["coefficient_rmse_se"] > 0
    assert paired[0]["difference_mean"] == 1.
    assert paired[0]["difference_se"] == 0.


def test_comparison_figure_smoke_test(campaign):
    root, _, _, _ = campaign
    report = summary.summarize(root, make_plots=True)
    assert report["success"], report["errors"]
    for name in ("comparison.pdf", "model_1_comparison.png", "model_1_comparison.pdf"):
        assert (root / "summary" / name).stat().st_size > 1000
        assert name in report["output_hashes"]
