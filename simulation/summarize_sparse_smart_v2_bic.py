#!/usr/bin/env python3
"""Audit and compare fixed/automatic training-BIC selection with validation v2.

Read frozen outputs in place: no refitting, raw-state copying, or validation
selection occurs here. Failed/missing task coverage is explicit. An excluded
initializer is a scientific outcome, not an execution failure. The automatic
curve reuses a group winner across the fixed-rank/free-count sensitivity axis;
those repeated appearances are not independent observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
for folder in (CODE / "sparse-smart/src", CODE / "sparse-smart-v2/src"):
    sys.path.insert(0, str(folder))
import numpy as np
from sparse_smart.chart import AnchorChart
from sparse_smart_v2.support import FreeRows, _check_masks
from sparse_smart_v2.selection import bic_from_rss, bic_score

METHODS = ("fixed_bic", "automatic_bic", "previous_validation")
SELECTION_RULE = "smallest terminal training BIC; ties use frozen task ID and candidate ID"
ERROR_CLASSES = {"missing", "corrupt", "execution_failure"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def check_hash(path, expected):
    require(isinstance(expected, str) and sha(path) == expected, f"SHA256 mismatch: {path}")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def relative(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
            f"unsafe relative path: {name}")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), f"path escapes root: {name}")
    return path


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def load_plan(root):
    plan = read(root / "plan.json")
    require(plan.get("schema_version") == 1 and plan.get("method") == "SparseSMARTv2BIC",
            "unsupported BIC plan schema")
    require(plan.get("root") == str(root), "plan root identity mismatch")
    require(digest({key: value for key, value in plan.items() if key != "plan_fingerprint"}) ==
            plan.get("plan_fingerprint"), "plan fingerprint mismatch")
    require(plan["configuration"].get("selection") == "bic_terminal", "plan is not terminal BIC")
    require(not plan["configuration"].get("validation_stopping", False), "validation stopping is enabled")
    require(plan["configuration"].get("validation_patience") is None, "validation patience is enabled")
    groups = {group["group_id"]: group for group in plan["groups"]}
    require(len(groups) == len(plan["groups"]) == plan["n_groups"] > 0, "invalid group coverage")
    require(len(plan["tasks"]) == plan["n_tasks"] > 0, "invalid task count")
    require((root / "work-items.tsv").read_text().splitlines() ==
            [str(i) for i in range(plan["n_tasks"])], "work table differs from frozen task IDs")
    by_group = defaultdict(list)
    for tid, task in enumerate(plan["tasks"]):
        require(task.get("task_id") == tid and type(task["task_id"]) is int and task["group_id"] in groups,
                "task identity mismatch")
        by_group[task["group_id"]].append(task)
    require(set(by_group) == set(groups), "group without planned tasks")
    grid = plan["configuration"]
    pairs = grid.get("penalty_pairs", [(u, v) for u in grid["penalties_u"] for v in grid["penalties_v"] if u != 0 or v != 0])
    expected_candidates = sum(len(task["free_counts"]) * len(pairs) + int(task["include_rrr"]) for task in plan["tasks"])
    if "n_candidates" in plan:
        require(plan["n_candidates"] == expected_candidates, "planned candidate count differs from task grid")
    cases = {case["case_id"]: case for case in plan["display_cases"]}
    require(len(cases) == len(plan["display_cases"]) > 0, "duplicate or absent display cases")
    require(all(case["group_id"] in groups for case in cases.values()), "display case has unknown group")
    check_hash(root / "source-manifest.json", plan["source_manifest_sha256"])
    manifest = read(root / "source-manifest.json")
    require(bool(manifest.get("files")), "source manifest is empty")
    for name, expected in manifest["files"].items():
        check_hash(relative(root / "source", name), expected)
    prep = read(root / "preparation.json")
    require(prep.get("success") is True and prep.get("status") == "complete", "preparation incomplete")
    for key, expected in (("plan_fingerprint", plan["plan_fingerprint"]), ("plan_sha256", sha(root / "plan.json")),
                          ("source_manifest_sha256", plan["source_manifest_sha256"]),
                          ("n_groups", plan["n_groups"]), ("n_tasks", plan["n_tasks"])):
        require(prep.get(key) == expected, f"preparation identity mismatch: {key}")
    prepared = {record["group_id"]: record for record in prep["groups"]}
    require(len(prepared) == len(prep["groups"]) and set(prepared) == set(groups), "prepared group coverage mismatch")
    return plan, by_group, prepared


def array_fingerprint(data, names):
    value = hashlib.sha256()
    for name in names:
        array = np.ascontiguousarray(data[name])
        value.update(json.dumps((name, array.shape, array.dtype.str)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def fingerprints(data):
    return dict(training_observed_input_fingerprint=array_fingerprint(data, ("X", "Y", "C0")),
                validation_observed_input_fingerprint=array_fingerprint(data, ("X_validation", "Y_validation")),
                evaluation_truth_fingerprint=array_fingerprint(data, ("C_star",)))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    for name, value in data.items():
        require(value.dtype.kind in "biuf" and np.isfinite(value).all(), f"invalid numerical array: {name}")
    return data


def load_group(root, plan, group, prepared):
    path = relative(root / "groups", group["group_id"]) / "group.json"
    check_hash(path, prepared["group_json_sha256"])
    meta = read(path)
    require(meta.get("group") == group and meta.get("plan_fingerprint") == plan["plan_fingerprint"],
            "group metadata identity mismatch")
    reference = Path(meta["reference_case_json_path"])
    check_hash(reference, meta["reference_case_json_sha256"])
    old = read(reference)
    require(old["fingerprints"] == meta["fingerprints"], "reference/group data fingerprints differ")
    for name, info in meta["files"].items():
        require(Path(info["path"]).is_absolute(), "shared data path is not absolute")
        check_hash(info["path"], info["sha256"])
        require(old["files"][name] == info["sha256"], "reference file hash differs")
    data = load_npz(meta["files"]["data.npz"]["path"])
    require(fingerprints(data) == meta["fingerprints"], "observed/truth data fingerprint mismatch")
    n, p, q = group["n_train"], group["p"], group["q"]
    for name, shape in {"X": (n, p), "Y": (n, q), "C0": (p, q), "C_star": (p, q)}.items():
        require(data[name].shape == shape, f"data shape mismatch: {name}")
    source = load_npz(meta["files"]["source.npz"]["path"])
    require(array_fingerprint(source, tuple(sorted(source))) == meta["source_fingerprint"],
            "source-frame fingerprint mismatch")
    close(source["left"].T @ source["left"], np.eye(p), "source left frame")
    close(source["right"].T @ source["right"], np.eye(q), "source right frame")
    return meta, data, source


def close(actual, expected, name, *, atol=2e-8, rtol=2e-8):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), f"numerical mismatch: {name}")


def outcome_key(outcome):
    return (outcome["selection"]["score"], outcome["task_id"], outcome["candidate_id"])


def selection_identity(outcome):
    return (outcome["rank"], tuple(outcome["free_directions"]), outcome.get("init_penalty"),
            outcome.get("penalty_u"), outcome.get("penalty_v"), outcome["fit_method"])


def expected_identities(task, group, config):
    pairs = config.get("penalty_pairs")
    if pairs is None:
        pairs = [(u, v) for u in config["penalties_u"] for v in config["penalties_v"] if u != 0 or v != 0]
    expected = {(task["rank"], (free, free), task["init_penalty"], u, v, "sparse_smart_v2")
                for free in task["free_counts"] for u, v in pairs}
    return expected


def check_task(root, plan, group, task, meta):
    folder = root / "tasks" / f"{task['task_id']:06d}"
    result, status = read(folder / "result.json"), read(folder / "status.json")
    check_hash(folder / "result.json", status.get("result_sha256"))
    for key, expected in (("schema_version", 1), ("method", "SparseSMARTv2BIC"),
                          ("plan_fingerprint", plan["plan_fingerprint"]),
                          ("source_manifest_sha256", plan["source_manifest_sha256"]),
                          ("task", task), ("group_id", group["group_id"]), ("fingerprints", meta["fingerprints"])):
        require(result.get(key) == expected, f"task result identity mismatch: {key}")
    require(status.get("status") == "finished", "task status is not finished")
    for key in ("schema_version", "method", "plan_fingerprint", "source_manifest_sha256", "task", "group_id", "execution_success", "success"):
        require(status.get(key) == result.get(key), f"task status/result mismatch: {key}")
    require(result.get("status") == "complete" and result.get("execution_success") is True,
            "task execution did not complete successfully")
    require(result.get("validation_used_for_fit") is False and result.get("selection_rule") == "bic_terminal",
            "task fitting/selection policy differs")
    require(isinstance(result.get("files"), dict), "missing task artifact hash mapping")
    if any(outcome.get("success") is True for outcome in result["outcomes"]):
        require("states.npz" in result["files"], "eligible task is missing bound retained states")
    for name, expected in result["files"].items():
        check_hash(relative(folder, name), expected)
    for prefix in ("process", "launcher"):
        exit_path = folder / f"{prefix}-exit-code.txt"
        require(exit_path.exists() and exit_path.read_text().strip() == "0", f"nonzero/missing {prefix} exit marker")
        with (folder / f"{prefix}-status.tsv").open(newline="") as stream:
            exit_rows = list(csv.DictReader(stream, delimiter="\t"))
        require(len(exit_rows) == 1 and exit_rows[0].get("exit_code") == "0" and
                exit_rows[0].get("job_id") == str(result["slurm"]["job_id"]) and bool(exit_rows[0].get("finished_utc")),
                f"invalid {prefix} exit status binding")
        if prefix == "process":
            require(exit_rows[0].get("step_id") == str(result["slurm"]["step_id"]), "process step binding differs")
    rows, identities, ids, rrr_count = [], set(), set(), 0
    for outcome in result["outcomes"]:
        require(isinstance(outcome.get("candidate_id"), str) and outcome["candidate_id"] not in ids,
                "invalid/duplicate candidate ID")
        ids.add(outcome["candidate_id"])
        require(outcome.get("rank") == task["rank"], "candidate rank differs from task")
        direct = outcome.get("fit_method") == "target_rrr"
        if direct:
            rrr_count += 1
            require(outcome["free_directions"] == [group["p"], group["q"]] and
                    outcome.get("penalty_u") == outcome.get("penalty_v") == 0, "invalid RRR identity")
        else:
            identity = selection_identity(outcome)
            require(identity not in identities, "duplicate candidate tuning tuple")
            identities.add(identity)
        require(type(outcome.get("success")) is bool and type(outcome.get("execution_success")) is bool,
                "invalid candidate success flags")
        row = dict(outcome, task_id=task["task_id"], group_id=group["group_id"], eligible=False)
        if not outcome["execution_success"]:
            row["classification"] = "execution_failure"
        elif not outcome["success"]:
            status_text = str(outcome.get("fit_status", ""))
            row["classification"] = ("initializer_exclusion" if "initial" in status_text or
                    outcome.get("refinement_skipped") == "initializer_failed_strict_preflight" else
                    "numerical_stagnation" if status_text in ("numerical_stagnation", "line_search_failed") else
                    "other_scientific_failure")
            require(outcome.get("selection") is None, "ineligible candidate has a BIC score")
        else:
            require(outcome.get("fit_status") in ("completed", "converged"), "successful candidate has failure status")
            require(type(outcome["n_iter"]) is int and 0 <= outcome["n_iter"] <= plan["configuration"]["iterations"],
                    "invalid iteration count")
            require(outcome["selected_iteration"] == outcome["n_iter"], "BIC candidate is not the terminal iterate")
            require(outcome.get("termination_reason") != "validation_stop", "candidate used validation stopping")
            if direct:
                require(outcome["n_iter"] == 0 and outcome.get("optimization_converged") is True and
                        outcome.get("termination_reason") == "target_rrr_closed_form" and
                        outcome.get("rrr_certificate", {}).get("certified") is True,
                        "invalid direct RRR completion/certificate")
            score = outcome["selection"]
            require(score.get("criterion") == "bic" and score.get("method") == outcome["fit_method"], "invalid BIC record")
            for key in ("score", "rss", "model_dimension"):
                require(finite(score.get(key)), f"nonfinite BIC {key}")
            for key, expected in (("n", group["n_train"]), ("p", group["p"]), ("q", group["q"]), ("rank", task["rank"])):
                require(score.get(key) == expected, f"BIC identity differs: {key}")
            close(bic_from_rss(score["rss"], n=score["n"], q=score["q"], model_dimension=score["model_dimension"]),
                  score["score"], "recorded BIC arithmetic")
            require(all(finite(outcome["metrics"].get(key)) and outcome["metrics"][key] >= 0 for key in
                        ("coefficient_rmse", "coefficient_frobenius_squared", "training_prediction_mse", "validation_mse")),
                    "invalid evaluation metrics")
            row.update(classification="eligible", eligible=True)
        rows.append(row)
    require(identities == expected_identities(task, group, plan["configuration"]), "candidate tuning coverage differs from plan")
    require(rrr_count == int(task["include_rrr"]), "RRR coverage differs from plan")
    return rows


def select_winner(rows, case=None):
    eligible = (row for row in rows if row["eligible"] and (case is None or
                row["rank"] == case["rank"] and (row["fit_method"] == "target_rrr" or
                row["free_directions"] == case["free_directions"])))
    return min(eligible, key=outcome_key, default=None)


def audit_winner(root, winner, data, source, group):
    require(isinstance(winner.get("state_key"), str), "selected candidate has no retained state")
    folder = root / "tasks" / f"{winner['task_id']:06d}"
    prefix = winner["state_key"]
    with np.load(folder / "states.npz", allow_pickle=False) as archive:
        state = {name[len(prefix):]: archive[name] for name in archive.files if name.startswith(prefix)}
    for name, value in state.items():
        require(value.dtype.kind in "biuf" and np.isfinite(value).all(), f"invalid selected state array: {name}")
    coefficient = state["coefficient"]
    require(coefficient.shape == data["C_star"].shape, "selected coefficient shape mismatch")
    direct = winner["fit_method"] == "target_rrr"
    arguments = dict(rank=winner["rank"], direct_rrr=direct,
                     support_tolerance=winner["selection"]["support_tolerance"])
    if not direct:
        arguments.update(free_directions=winner["free_directions"],
                         **{name: state[name] for name in ("weighted_u", "weighted_v", "penalized_u", "penalized_v")})
        p, q = coefficient.shape
        chart = AnchorChart(p, q, state["anchors_u"], state["anchors_v"], state["center_u"], state["center_v"])
        domain = {name: group["margins"][name] for name in ("d_lower", "d_upper", "gap", "anchor_min")}
        reason = chart.domain_reason(state["state"], **domain)
        require(reason is None, f"selected chart is infeasible: {reason}")
        reconstructed = chart.reconstruct(state["state"])
        for name, value in zip(("P", "d", "Q"), reconstructed):
            close(value, state[name], f"saved chart reconstruction {name}")
        for name, value in zip(("weighted_u", "weighted_v"), chart.unpack(state["state"])[-2:]):
            close(value, state[name], f"saved chart support block {name}", atol=0, rtol=0)
        masks = FreeRows(state["free_rows_u"], state["free_rows_v"], state["penalized_u"], state["penalized_v"])
        _check_masks(chart, masks)
        require([len(masks.rows_u), len(masks.rows_v)] == winner["free_directions"], "saved free counts differ")
    score = bic_score(data["X"], data["Y"], coefficient, **arguments).as_dict()
    for name, value in score.items():
        if finite(value):
            close(value, winner["selection"].get(name), f"independent BIC {name}")
        else:
            require(value == winner["selection"].get(name), f"independent BIC metadata mismatch: {name}")
    if all(name in state for name in ("P", "d", "Q")):
        factored = (state["P"] * state["d"]) @ state["Q"].T
        if not direct:
            factored = source["left"] @ factored @ source["right"].T
        close(factored, coefficient, "factor coefficient reconstruction")
        close(state["P"].T @ state["P"], np.eye(len(state["d"])), "left factor orthogonality")
        close(state["Q"].T @ state["Q"], np.eye(len(state["d"])), "right factor orthogonality")
    error = coefficient - data["C_star"]
    metrics = dict(coefficient_rmse=float(np.linalg.norm(error) / math.sqrt(error.size)),
                   coefficient_frobenius_squared=float(np.sum(error * error)),
                   training_prediction_mse=float(np.mean((data["X"] @ error) ** 2)),
                   validation_mse=float(np.mean((data["Y_validation"] - data["X_validation"] @ coefficient) ** 2)))
    for name, value in metrics.items():
        close(value, winner["metrics"][name], f"independent evaluation {name}")
    return dict(task_id=winner["task_id"], candidate_id=winner["candidate_id"], passed=True,
                audited="coefficient, observed RSS, BIC support/dimension, evaluation metrics, available factor reconstruction")


def load_reference(reference_root):
    root = Path(reference_root).resolve()
    index = read(root / "campaign-index.json")
    records, inputs = {}, {"campaign-index.json": sha(root / "campaign-index.json")}
    for entry in index["roots"]:
        folder = root / Path(entry["root"]).name
        summary_path = folder / "summary/summary.json"
        summary = read(summary_path)
        require(summary.get("success") and summary.get("complete_execution_coverage"), "reference summary incomplete")
        post_path = folder / "summary/postprocess.json"
        post = read(post_path)
        require(post.get("success"), "reference postprocessing failed")
        check_hash(summary_path, post["summary_sha256"])
        check_hash(folder / "summary/numerical-audit.json", post["audit_sha256"])
        require(read(folder / "summary/numerical-audit.json").get("success"), "reference audit failed")
        inputs[str(summary_path.relative_to(root))] = sha(summary_path)
        prep = read(folder / "preparation.json")
        require(prep.get("success") and prep.get("status") == "complete", "reference preparation incomplete")
        check_hash(folder / "plan.json", prep["plan_sha256"])
        require(summary["plan_sha256"] == prep["plan_sha256"], "reference summary/preparation plan differs")
        prepared = {record["case_id"]: record for record in prep["cases"]}
        for record in summary["case_results"]:
            case = record["case"]
            require(case["case_id"] not in records and record["winner"] is not None, "reference case absent/duplicated")
            records[case["case_id"]] = dict(record, case_json_path=str(folder / "cases" / case["case_id"] / "case.json"),
                                           case_json_sha256=prepared[case["case_id"]]["case_json_sha256"])
    return records, inputs


def reference_winner(case, record, group_meta):
    old_case, winner = record["case"], record["winner"]
    for name in ("case_id", "model_id", "experiment_id", "setting_index", "seed_id", "random_seed", "n_train", "p", "q", "sigma0", "rank", "free_directions"):
        require(old_case.get(name) == case.get(name), f"reference/display case mismatch: {name}")
    check_hash(record["case_json_path"], record["case_json_sha256"])
    old_meta = read(record["case_json_path"])
    require(old_meta.get("case") == old_case, "reference metadata case identity differs")
    require(old_meta["fingerprints"] == group_meta["fingerprints"], "paired data fingerprint mismatch")
    require(finite(winner["coefficient_rmse"]), "reference winner has nonfinite error")
    return dict(candidate_id=f"validation_task_{winner['task_id']}", rank=case["rank"],
                free_directions=case["free_directions"], fit_method=winner["fit_method"],
                init_penalty=winner["init_penalty"], penalty_u=winner["penalty_u"], penalty_v=winner["penalty_v"],
                selected_iteration=winner["selected_iteration"], n_iter=winner["n_iter"],
                termination_reason=winner["termination_reason"], optimization_converged=winner["optimization_converged"],
                selection=None, metrics={key: winner[key] for key in ("coefficient_rmse", "coefficient_frobenius_squared", "training_prediction_mse")},
                elapsed_seconds=winner["elapsed_seconds"])


def statistics_of(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                se=statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None)


def xvalue(case):
    return case[("n_train", "rank", "source_rank", "sigma0")[case["experiment_id"]]]


def build_tables(case_results):
    per_seed, curves, differences = [], [], []
    groups = defaultdict(list)
    for record in case_results:
        case = record["case"]
        groups[(case["model_id"], case["experiment_id"], case["setting_index"])].append(record)
        for method in METHODS:
            winner = record.get(method)
            row = dict(case_id=case["case_id"], group_id=case["group_id"], model_id=case["model_id"],
                       experiment_id=case["experiment_id"], setting_index=case["setting_index"], seed_id=case["seed_id"],
                       x=xvalue(case), method=method, available=winner is not None,
                       complete_execution_coverage=record["complete_execution_coverage"])
            if winner is not None:
                row.update(coefficient_rmse=winner["metrics"]["coefficient_rmse"], selected_rank=winner["rank"],
                           selected_free_u=winner["free_directions"][0], selected_free_v=winner["free_directions"][1],
                           fit_method=winner["fit_method"], selected_iteration=winner["selected_iteration"],
                           bic=None if winner["selection"] is None else winner["selection"]["score"],
                           init_penalty=winner.get("init_penalty"), penalty_u=winner.get("penalty_u"), penalty_v=winner.get("penalty_v"),
                           termination_reason=winner.get("termination_reason"), optimization_converged=winner.get("optimization_converged"))
            per_seed.append(row)
    for key, members in sorted(groups.items()):
        common = dict(model_id=key[0], experiment_id=key[1], setting_index=key[2], x=xvalue(members[0]["case"]),
                      requested_seeds=len(members), complete_execution_coverage=all(r["complete_execution_coverage"] for r in members))
        for method in METHODS:
            winners = [r[method] for r in members if r.get(method) is not None]
            stats = statistics_of([w["metrics"]["coefficient_rmse"] for w in winners])
            curves.append(dict(common, method=method, available_seeds=stats["n"], coefficient_rmse_mean=stats["mean"],
                               coefficient_rmse_se=stats["se"], selected_rank_counts=dict(Counter(w["rank"] for w in winners)),
                               selected_free_counts=dict(Counter(str(w["free_directions"]) for w in winners)),
                               rrr_selected=sum(w["fit_method"] == "target_rrr" for w in winners)))
        for method, reference in (("fixed_bic", "previous_validation"), ("automatic_bic", "previous_validation"), ("automatic_bic", "fixed_bic")):
            pairs = [r for r in members if r.get(method) is not None and r.get(reference) is not None]
            deltas = [r[method]["metrics"]["coefficient_rmse"] - r[reference]["metrics"]["coefficient_rmse"] for r in pairs]
            stats = statistics_of(deltas)
            differences.append(dict(common, method=method, reference=reference, paired_seeds=stats["n"],
                                    difference_mean=stats["mean"], difference_se=stats["se"],
                                    interpretation="negative means lower error for method"))
    return per_seed, curves, differences


def plot_curves(output, curves):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import t
    labels = dict(fixed_bic="Fixed ranks/counts: BIC", automatic_bic="Automatic ranks/counts: BIC",
                  previous_validation="Previous v2: validation selection")
    colors = dict(fixed_bic="#087E8B", automatic_bic="#A84B26", previous_validation="#677286")
    xlabels = ("Training observations", "Fixed fitted rank", "Fixed free/source directions", "Source-noise standard deviation")
    outputs = []
    with PdfPages(output / "comparison.pdf") as pdf:
        for model in sorted({row["model_id"] for row in curves}):
            fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
            fig.subplots_adjust(top=.85, bottom=.13, hspace=.38, wspace=.24)
            fig.suptitle(f"SparseSMART v2 — Model {('I', 'II', 'III')[model]}", fontsize=19, y=.97)
            handles = []
            for experiment, ax in enumerate(axes.flat):
                subset = [row for row in curves if row["model_id"] == model and row["experiment_id"] == experiment]
                ticks = sorted({row["x"] for row in subset})
                for method in METHODS:
                    rows = sorted((row for row in subset if row["method"] == method and row["coefficient_rmse_mean"] is not None), key=lambda row: row["x"])
                    xs = [ticks.index(row["x"]) if experiment == 3 else row["x"] for row in rows]
                    ys = [row["coefficient_rmse_mean"] for row in rows]
                    ci = [t.ppf(.975, row["available_seeds"] - 1) * row["coefficient_rmse_se"] if row["coefficient_rmse_se"] is not None else 0 for row in rows]
                    line = ax.errorbar(xs, ys, yerr=ci, color=colors[method], marker="o", ms=4, lw=1.8,
                                       linestyle="--" if method == "previous_validation" else "-", capsize=2, label=labels[method])
                    if experiment == 0:
                        handles.append(line)
                ax.set_title(f"Experiment {experiment + 1}", loc="left", fontsize=13)
                ax.set_xlabel(xlabels[experiment]); ax.set_ylabel("Coefficient RMSE")
                ax.grid(alpha=.22); ax.spines[["top", "right"]].set_visible(False)
                if experiment == 3:
                    ax.set_xticks(range(len(ticks)), [f"{value:g}" for value in ticks])
                else:
                    ax.set_xticks(ticks)
                if experiment == 1 and any(row["coefficient_rmse_mean"] and row["coefficient_rmse_mean"] > 0 for row in subset):
                    ax.set_yscale("log")
            fig.legend(handles=handles, labels=[labels[method] for method in METHODS], loc="upper center", bbox_to_anchor=(.5,.925), ncol=3, frameon=False)
            fig.text(.06, .055, "Mean coefficient error ||C − C*||F / √(pq); bars: 95% t intervals across available seeds. Lower is better.\n"
                     "Automatic winners are reused along fixed-rank/count sensitivity axes. Source-noise ticks are equally spaced.\n"
                     "BIC uses terminal training fits; the earlier v2 used validation for selection and stopping. See coverage and paired differences.", fontsize=9)
            for suffix in ("png", "pdf"):
                path = output / f"model_{model + 1}_comparison.{suffix}"
                fig.savefig(path, dpi=170, bbox_inches="tight")
                outputs.append(path.name)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    outputs.append("comparison.pdf")
    return outputs


def summarize(root, output_root=None, reference_root=None, *, make_plots=True):
    root = Path(root).resolve()
    output = Path(output_root).resolve() if output_root else root / "summary"
    require(output != root and not root.is_relative_to(output), "summary must not replace the campaign root")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, method="SparseSMARTv2BIC", result_root=str(root), output_root=str(output),
                  checked_utc=datetime.now(timezone.utc).isoformat(), success=False, errors=[], group_results=[], case_results=[],
                  selection_rule=SELECTION_RULE, truth_used_for_selection=False, validation_used_for_selection=False,
                  complete_execution_coverage=False, complete_tuning_coverage=False,
                  audit_scope="all unique fixed-BIC and automatic-BIC selected states; observed data and evaluation metrics")
    counts, task_counts, audits = Counter(), Counter(), []
    output_files = []
    outcomes_path = output / "task-outcomes.jsonl"
    temporary = outcomes_path.with_suffix(".jsonl.tmp")
    try:
        plan, by_group, prepared = load_plan(root)
        reference_root = Path(reference_root or plan["reference_root"]).resolve()
        reference, reference_inputs = load_reference(reference_root)
        report.update(plan_fingerprint=plan["plan_fingerprint"], plan_sha256=sha(root / "plan.json"),
                      source_manifest_sha256=plan["source_manifest_sha256"], configuration=plan["configuration"],
                      reference_root=str(reference_root), reference_inputs=reference_inputs,
                      planned_groups=plan["n_groups"], planned_tasks=plan["n_tasks"], planned_cases=len(plan["display_cases"]),
                      planned_candidates=plan.get("n_candidates"))
        cases_by_group = defaultdict(list)
        for case in plan["display_cases"]:
            cases_by_group[case["group_id"]].append(case)
        with temporary.open("w") as stream:
            for group in plan["groups"]:
                gid = group["group_id"]
                members, rows, group_counts = by_group[gid], [], Counter()
                try:
                    meta, data, source = load_group(root, plan, group, prepared[gid])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    meta = data = source = None
                    report["errors"].append(dict(group_id=gid, error=f"group provenance: {error}"))
                for task in members:
                    try:
                        require(meta is not None, "group provenance invalid")
                        task_rows = check_task(root, plan, group, task, meta)
                        task_class = "complete"
                    except FileNotFoundError as error:
                        task_rows, task_class = [], "missing"
                        report["errors"].append(dict(task_id=task["task_id"], group_id=gid, classification=task_class, error=str(error)))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        task_rows, task_class = [], "corrupt"
                        report["errors"].append(dict(task_id=task["task_id"], group_id=gid, classification=task_class, error=str(error)))
                    task_counts[task_class] += 1
                    for row in task_rows:
                        counts[row["classification"]] += 1
                        group_counts[row["classification"]] += 1
                        if row["classification"] in ERROR_CLASSES:
                            report["errors"].append(dict(task_id=task["task_id"], candidate_id=row["candidate_id"], classification=row["classification"]))
                    rows.extend(task_rows)
                    stream.write(json.dumps(dict(task_id=task["task_id"], group_id=gid, classification=task_class,
                                                 outcome_counts=dict(Counter(row["classification"] for row in task_rows))), sort_keys=True) + "\n")
                complete = (len(rows) > 0 and all(row["classification"] not in ERROR_CLASSES for row in rows)
                            and len({row["task_id"] for row in rows}) == len(members))
                automatic = select_winner(rows)
                fixed = {case["case_id"]: select_winner(rows, case) for case in cases_by_group[gid]}
                selected = {winner["candidate_id"]: winner for winner in (automatic, *fixed.values()) if winner is not None}
                invalid = set()
                for cid, winner in selected.items():
                    try:
                        audits.append(audit_winner(root, winner, data, source, group))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        invalid.add(cid)
                        report["errors"].append(dict(group_id=gid, candidate_id=cid, error=f"selected-state audit: {error}"))
                if automatic is None or automatic["candidate_id"] in invalid:
                    automatic = None
                    report["errors"].append(dict(group_id=gid, error="no valid automatic winner"))
                report["group_results"].append(dict(group=group, planned_tasks=len(members), outcome_counts=dict(group_counts),
                    complete_execution_coverage=complete, automatic_bic=automatic))
                for case in cases_by_group[gid]:
                    winner = fixed[case["case_id"]]
                    if winner is None or winner["candidate_id"] in invalid:
                        winner = None
                        report["errors"].append(dict(case_id=case["case_id"], error="no valid fixed winner"))
                    try:
                        require(meta is not None, "group provenance invalid")
                        old = reference_winner(case, reference[case["case_id"]], meta)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        old = None
                        report["errors"].append(dict(case_id=case["case_id"], error=f"reference pairing: {error}"))
                    report["case_results"].append(dict(case=case, fixed_bic=winner, automatic_bic=automatic,
                                                       previous_validation=old, complete_execution_coverage=complete))
                if len(report["group_results"]) % 100 == 0 or len(report["group_results"]) == plan["n_groups"]:
                    print(json.dumps(dict(summarized_groups=len(report["group_results"]), planned_groups=plan["n_groups"],
                                          audited_winners=len(audits), errors=len(report["errors"])), sort_keys=True), flush=True)
        temporary.replace(outcomes_path)
        report.update(outcome_counts=dict(counts), task_counts=dict(task_counts), selected_state_audits=audits,
                      audited_unique_winners=len(audits), complete_execution_coverage=task_counts["complete"] == plan["n_tasks"] and not counts["execution_failure"],
                      complete_tuning_coverage=not any(counts[key] for key in counts if key != "eligible") and task_counts["complete"] == plan["n_tasks"])
        output_files.append(outcomes_path.name)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(dict(error=f"campaign provenance/summary: {error}"))
    finally:
        temporary.unlink(missing_ok=True)
    per_seed, curves, differences = build_tables(report["case_results"])
    for name, rows in (("per-seed-results.csv", per_seed), ("performance-curves.csv", curves), ("paired-differences.csv", differences)):
        write_csv(output / name, rows); output_files.append(name)
    report["setting_results"] = curves
    report["paired_differences"] = differences
    report["cases_with_winner"] = {method: sum(record.get(method) is not None for record in report["case_results"]) for method in METHODS}
    # Counts over independent canonical data groups avoid double-counting the
    # automatic winner repeated on the rank/free-count sensitivity figures.
    auto = [record["automatic_bic"] for record in report["group_results"] if record["automatic_bic"] is not None]
    report["automatic_group_selection"] = dict(n_groups=len(auto), rank_counts=dict(Counter(w["rank"] for w in auto)),
        free_count_counts=dict(Counter(str(w["free_directions"]) for w in auto)), rrr_selected=sum(w["fit_method"] == "target_rrr" for w in auto))
    if make_plots and curves:
        try:
            output_files.extend(plot_curves(output, curves))
        except (ImportError, OSError, ValueError) as error:
            report["errors"].append(dict(error=f"plotting: {error}"))
    report["success"] = not report["errors"] and report["complete_execution_coverage"]
    report["output_hashes"] = {name: sha(output / name) for name in output_files}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--result-root", dest="root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = summarize(args.root, args.output_root, args.reference_root, make_plots=not args.no_plots)
    print(json.dumps({key: report.get(key) for key in ("success", "planned_tasks", "task_counts", "outcome_counts", "cases_with_winner", "audited_unique_winners")}, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
