#!/usr/bin/env python3
"""Freeze a training-BIC initializer pair, then assess its independent seed bank.

Development never loads validation/truth arrays or reads candidate metrics.
All candidate banks have the same 200-update budget. Display aliases are never
treated as extra independent groups; uncertainty uses Monte Carlo seed blocks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_sparse_smart_v2_bic as base
import numpy as np

INITIALIZERS = (.003, .03, .1, .3, 1., 3.)
PENALTIES = ((.001, .001), (.0025, .0025), (.01, .01))
PAIRS = tuple(combinations(INITIALIZERS, 2))
METRICS = {
    "coefficient_rmse": "Coefficient RMSE",
    "validation_mse": "Validation response MSE (includes response noise)",
    "training_prediction_mse": "Training signal prediction MSE (uses coefficient truth)",
}
require, read, digest = base.require, base.read, base.digest


def scientific_identity(plan):
    """Settings common to both independently seeded phases, excluding data IDs."""
    return {key: plan["configuration"].get(key) for key in (
        "iterations", "init_penalties", "penalty_pairs", "inverse_step", "stationarity_tol",
        "max_backtracks", "adaptive_anchors", "max_anchor_switches", "refinement_solver",
        "support_tolerance", "selection", "selection_rule", "protocols", "margins", "initialization_spectrum")}


def validate_phase(plan, phase):
    config = plan["configuration"]
    phase_name = "development" if phase == "develop" else "assessment"
    require(config.get("cheap_bic_phase") == phase_name and plan.get("cheap_bic", {}).get("phase") == phase_name,
            "plan is not the requested frozen cheap-pilot phase")
    seeds = set(range(5)) if phase == "develop" else set(range(5, 10))
    require(config["iterations"] == 200, "cheap pilot requires the exact 200-update budget")
    require(tuple(config["init_penalties"]) == INITIALIZERS, "requires all six frozen initializers")
    require(tuple(map(tuple, config["penalty_pairs"])) == PENALTIES, "requires the three tied positive penalties")
    require(config.get("selection_rule") == "bic_terminal" and config.get("validation_patience") is None,
            "pilot must use terminal training BIC without validation stopping")
    groups = plan["groups"]
    require(len(groups) == 165, "phase must contain all 165 dataset/protocol groups")
    identities = {(g["model_id"], g["n_train"], g["sigma0"], g["seed_id"], g["protocol"]) for g in groups}
    require(len(identities) == len(groups), "duplicate dataset/protocol group")
    expected = {(model, seed, protocol): count for model in range(3) for seed in seeds
                for protocol, count in (("low_floor", 5), ("standard_floor", 6))}
    actual = Counter((g["model_id"], g["seed_id"], g["protocol"]) for g in groups)
    require(actual == expected, "phase model/seed/protocol cohort differs from the frozen pilot")
    by_group = defaultdict(list)
    for task in plan["tasks"]:
        by_group[task["group_id"]].append(task)
    for group in groups:
        tasks = by_group[group["group_id"]]
        require(len(tasks) == 6 and sorted(t["init_penalty"] for t in tasks) == list(INITIALIZERS),
                "each group requires exactly six initializer tasks")
        ranks = {t["rank"] for t in tasks}
        require(len(ranks) == 1, "a group must use one frozen RSC-selected rank")
        rank = next(iter(ranks))
        require(type(rank) is int and 1 <= rank <= min(group["p"], group["q"]), "invalid selected rank")
        rank_record = plan["cheap_bic"]["rank_records"][group["dataset_id"]]
        require(group["selected_target_rank"] == rank_record["selected_rank"] == rank and
                group["rank_selection_sha256"] == digest(rank_record), "group/task rank differs from frozen RSC record")
        counts = sorted({rank, *(v for v in (5, 7, 10, 15, 20) if rank <= v <= min(group["p"], group["q"]))})
        counts = [value for value in counts if not value == group["p"] == group["q"]]
        require(all(t["initializer_source_rank"] == max(10, rank) and t["free_counts"] == counts for t in tasks),
                "initializer dimension/free grid differs from frozen rule")
        require(sum(bool(t["include_rrr"]) for t in tasks) == 1, "each group requires exactly one RRR endpoint")
    return sorted(seeds)


def load_training_group(root, plan, group, prepared):
    """Verify shared provenance without opening any evaluation arrays."""
    path = base.relative(root / "groups", group["group_id"]) / "group.json"
    base.check_hash(path, prepared["group_json_sha256"])
    meta = read(path)
    require(meta.get("group") == group and meta.get("plan_fingerprint") == plan["plan_fingerprint"],
            "group identity mismatch")
    base.check_hash(meta["reference_case_json_path"], meta["reference_case_json_sha256"])
    old = read(meta["reference_case_json_path"])
    require(old["fingerprints"] == meta["fingerprints"], "reference/group fingerprints differ")
    rank_record = plan["cheap_bic"]["rank_records"][group["dataset_id"]]
    require(rank_record["training_observed_input_fingerprint"] == meta["fingerprints"]["training_observed_input_fingerprint"],
            "RSC selection and candidate bank use different training observations")
    for name, record in meta["files"].items():
        require(Path(record["path"]).is_absolute(), "shared artifact path is not absolute")
        base.check_hash(record["path"], record["sha256"])
        require(old["files"][name] == record["sha256"], "reference artifact identity differs")
    return meta


def check_training_task(root, plan, group, task, meta):
    """Equivalent task/score checks to the exhaustive auditor, without metrics."""
    folder = root / "tasks" / f"{task['task_id']:06d}"
    result, status = read(folder / "result.json"), read(folder / "status.json")
    base.check_hash(folder / "result.json", status.get("result_sha256"))
    for key, expected in (("schema_version", 1), ("method", "SparseSMARTv2BIC"),
            ("plan_fingerprint", plan["plan_fingerprint"]), ("source_manifest_sha256", plan["source_manifest_sha256"]),
            ("task", task), ("group_id", group["group_id"]), ("fingerprints", meta["fingerprints"])):
        require(result.get(key) == expected, f"task identity mismatch: {key}")
    require(result.get("status") == "complete" and result.get("execution_success") is True,
            "task execution incomplete or failed")
    require(status.get("status") == "finished", "task status is not finished")
    for key in ("schema_version", "method", "plan_fingerprint", "source_manifest_sha256", "task", "group_id",
                "execution_success", "success"):
        require(status.get(key) == result.get(key), f"task/status mismatch: {key}")
    require(result.get("validation_used_for_fit") is False and result.get("selection_rule") == "bic_terminal",
            "task used a different fitting/selection policy")
    for name, expected in result["files"].items():
        base.check_hash(base.relative(folder, name), expected)
    if any(row.get("success") for row in result["outcomes"]):
        require("states.npz" in result["files"], "eligible task has no bound states")
    for prefix in ("process", "launcher"):
        require((folder / f"{prefix}-exit-code.txt").read_text().strip() == "0", "nonzero worker/launcher exit")
        import csv
        with (folder / f"{prefix}-status.tsv").open(newline="") as stream:
            markers = list(csv.DictReader(stream, delimiter="\t"))
        require(len(markers) == 1 and markers[0].get("exit_code") == "0" and
                markers[0].get("job_id") == str(result["slurm"]["job_id"]) and markers[0].get("finished_utc"),
                "invalid worker/launcher status binding")
        if prefix == "process":
            require(markers[0].get("step_id") == str(result["slurm"]["step_id"]), "process step mismatch")
    rows, identities, ids, rrr_count = [], set(), set(), 0
    for original in result["outcomes"]:
        # Deliberately do not inspect, copy, or validate evaluation metric values.
        row = {key: original[key] for key in original if key != "metrics"}
        require(isinstance(row.get("candidate_id"), str) and row["candidate_id"] not in ids, "duplicate candidate ID")
        ids.add(row["candidate_id"])
        require(row["rank"] == task["rank"], "candidate rank mismatch")
        direct = row["fit_method"] == "target_rrr"
        if direct:
            rrr_count += 1
            require(row["free_directions"] == [group["p"], group["q"]] and row["penalty_u"] == row["penalty_v"] == 0,
                    "invalid RRR identity")
        else:
            identity = base.selection_identity(row)
            require(identity not in identities, "duplicate candidate tuning tuple")
            identities.add(identity)
        require(type(row.get("execution_success")) is bool and row["execution_success"], "candidate execution failed")
        require(type(row.get("success")) is bool, "invalid scientific success flag")
        row.update(task_id=task["task_id"], group_id=group["group_id"], eligible=row["success"])
        if not row["success"]:
            require(row.get("selection") is None, "ineligible candidate has score")
            text = str(row.get("fit_status", ""))
            row["classification"] = ("initializer_exclusion" if "initial" in text or text == "lasso_not_converged"
                else "numerical_stagnation" if text in ("numerical_stagnation", "line_search_failed")
                else "other_scientific_failure")
        else:
            require(row["fit_status"] in ("completed", "converged"), "eligible candidate has failure status")
            require(type(row["n_iter"]) is int and 0 <= row["n_iter"] <= 200 and
                    row["selected_iteration"] == row["n_iter"] and row["termination_reason"] != "validation_stop",
                    "candidate is not a terminal 200-budget fit")
            if direct:
                require(row["n_iter"] == 0 and row.get("optimization_converged") is True and
                    row["termination_reason"] == "target_rrr_closed_form" and row.get("rrr_certificate", {}).get("certified") is True,
                    "uncertified RRR endpoint")
            score = row["selection"]
            require(score["criterion"] == "bic" and score["method"] == row["fit_method"], "BIC method mismatch")
            require(score.get("support_tolerance") == plan["configuration"].get("support_tolerance", 0.),
                    "BIC support tolerance differs from the frozen protocol")
            for key, value in (("n", group["n_train"]), ("p", group["p"]), ("q", group["q"]), ("rank", task["rank"])):
                require(score[key] == value, f"BIC identity mismatch: {key}")
            require(all(base.finite(score[k]) for k in ("score", "rss", "model_dimension")), "nonfinite BIC terms")
            base.close(base.bic_from_rss(score["rss"], n=score["n"], q=score["q"], model_dimension=score["model_dimension"]),
                       score["score"], "candidate BIC arithmetic")
            row["classification"] = "eligible"
        rows.append(row)
    require(identities == base.expected_identities(task, group, plan["configuration"]), "candidate tuning coverage incomplete")
    require(rrr_count == int(task["include_rrr"]), "RRR coverage mismatch")
    return rows


def winner(rows, pair=None):
    allowed = [row for row in rows if row["eligible"] and
               (pair is None or row["fit_method"] == "target_rrr" or row["init_penalty"] in pair)]
    require(bool(allowed), "no eligible candidate, including RRR")
    return min(allowed, key=base.outcome_key)


def audit_training_winner(root, row, meta, group):
    """Reconstruct the retained coefficient and BIC using observed training only."""
    require(isinstance(row.get("state_key"), str), "selected subset winner has no retained state")
    prefix = row["state_key"]
    with np.load(root / "tasks" / f"{row['task_id']:06d}" / "states.npz", allow_pickle=False) as archive:
        state = {name[len(prefix):]: archive[name] for name in archive.files if name.startswith(prefix)}
    require(all(v.dtype.kind in "biuf" and np.isfinite(v).all() for v in state.values()), "invalid retained state")
    coefficient = state["coefficient"]
    require(coefficient.shape == (group["p"], group["q"]), "retained coefficient shape mismatch")
    with np.load(meta["files"]["data.npz"]["path"], allow_pickle=False) as archive:
        x, y = archive["X"], archive["Y"]
    require(x.shape == (group["n_train"], group["p"]) and y.shape == (group["n_train"], group["q"]), "training array shape mismatch")
    direct = row["fit_method"] == "target_rrr"
    arguments = dict(rank=row["rank"], direct_rrr=direct, support_tolerance=row["selection"]["support_tolerance"])
    if not direct:
        chart = base.AnchorChart(group["p"], group["q"], state["anchors_u"], state["anchors_v"], state["center_u"], state["center_v"])
        require(chart.domain_reason(state["state"], **{k: group["margins"][k] for k in
                ("d_lower", "d_upper", "gap", "anchor_min")}) is None, "selected chart outside domain")
        p, d, q = chart.reconstruct(state["state"])
        for name, value in zip(("P", "d", "Q"), (p, d, q)):
            base.close(value, state[name], f"selected factor {name}")
        for name, value in zip(("weighted_u", "weighted_v"), chart.unpack(state["state"])[-2:]):
            base.close(value, state[name], f"selected support {name}", atol=0, rtol=0)
        masks = base.FreeRows(state["free_rows_u"], state["free_rows_v"], state["penalized_u"], state["penalized_v"])
        base._check_masks(chart, masks)
        require([len(masks.rows_u), len(masks.rows_v)] == row["free_directions"], "selected free count mismatch")
        with np.load(meta["files"]["source.npz"]["path"], allow_pickle=False) as archive:
            reconstructed = archive["left"] @ ((p * d) @ q.T) @ archive["right"].T
        base.close(reconstructed, coefficient, "selected coefficient reconstruction")
        arguments.update(free_directions=row["free_directions"], **{k: state[k] for k in
                         ("weighted_u", "weighted_v", "penalized_u", "penalized_v")})
    actual = base.bic_score(x, y, coefficient, **arguments).as_dict()
    for key, value in actual.items():
        if base.finite(value):
            base.close(value, row["selection"].get(key), f"independent training BIC {key}")
        else:
            require(value == row["selection"].get(key), f"independent training BIC metadata {key}")
    return dict(task_id=row["task_id"], candidate_id=row["candidate_id"], passed=True, evaluation_data_read=False)


def pair_scores(records):
    require(len(records) > 0, "empty development cohort")
    require(len({r["group"]["group_id"] for r in records}) == len(records), "duplicate development groups")
    table = []
    for pair in PAIRS:
        regrets, rrr = [], 0
        for record in records:
            selected, full = winner(record["rows"], pair), winner(record["rows"])
            delta = (selected["selection"]["score"] - full["selection"]["score"]) / (record["group"]["n_train"] * record["group"]["q"])
            require(delta >= -1e-12, "negative nested-library BIC regret")
            regrets.append(max(0., delta))
            rrr += selected["fit_method"] == "target_rrr"
        table.append(dict(init_penalties=list(pair), groups=len(records), mean_normalized_bic_regret=statistics.mean(regrets),
                          max_normalized_bic_regret=max(regrets), rrr_selected=rrr))
    chosen = min(table, key=lambda row: (row["mean_normalized_bic_regret"], tuple(row["init_penalties"])))
    return table, chosen


def read_bank(root, phase):
    plan, by_group, prepared = base.load_plan(root)
    validate_phase(plan, phase)
    records, errors = [], []
    for group in plan["groups"]:
        try:
            meta = load_training_group(root, plan, group, prepared[group["group_id"]])
            rows = []
            for task in by_group[group["group_id"]]:
                rows.extend(check_training_task(root, plan, group, task, meta))
            require(sum(row["fit_method"] == "target_rrr" for row in rows) == 1, "group RRR is not unique")
            winner(rows)
            records.append(dict(group=group, rows=rows, meta=meta))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(group_id=group["group_id"], error=str(error)))
    return plan, records, errors


def inventory(records):
    return dict(complete_groups=len(records), outcomes=dict(Counter(row["classification"] for record in records for row in record["rows"])),
        termination_reasons=dict(Counter(str(row.get("termination_reason") or "not_reported") for record in records for row in record["rows"])),
        initializer_excluded_group_strengths=len({(record["group"]["group_id"], row["init_penalty"]) for record in records
            for row in record["rows"] if row["classification"] == "initializer_exclusion"}),
        rrr_only_groups=sum(not any(row["eligible"] and row["fit_method"] != "target_rrr" for row in record["rows"]) for record in records),
        C_rrr_selected=sum(winner(record["rows"])["fit_method"] == "target_rrr" for record in records))


def output_path(root, output_root, default):
    output = Path(output_root).resolve() if output_root else root / default
    require(output != root and not root.is_relative_to(output), "output must not replace campaign root")
    output.mkdir(parents=True, exist_ok=True)
    return output


def develop(root, output_root=None):
    root = Path(root).resolve()
    output = output_path(root, output_root, "development")
    report = dict(schema_version=1, phase="develop", success=False, errors=[], evaluation_metrics_used=False,
                  truth_used_for_selection=False, validation_used_for_selection=False,
                  complete_execution_coverage=False, complete_tuning_coverage=False)
    try:
        plan, records, errors = read_bank(root, "develop")
        report.update(inventory(records), errors=errors, planned_groups=165, plan_fingerprint=plan["plan_fingerprint"],
                      complete_execution_coverage=not errors and len(records) == 165,
                      complete_tuning_coverage=not errors and len(records) == 165 and
                          all(row["eligible"] for record in records for row in record["rows"]))
        require(not errors and len(records) == 165, "pair selection requires the same complete 165-group cohort for every pair")
        table, selected = pair_scores(records)
        audits = []
        for record in records:
            selections = {row["candidate_id"]: row for row in [winner(record["rows"]),
                          *(winner(record["rows"], pair) for pair in PAIRS)]}
            for row in selections.values():
                audits.append(audit_training_winner(root, row, record["meta"], record["group"]))
        artifact = dict(schema_version=1, method="SparseSMARTv2CheapBICPair", phase="develop", iterations=200,
            selected_init_penalties=selected["init_penalties"], selection_criterion="group_equal_mean_(pair_BIC-C_BIC)/(n*q)",
            tie_rule="lexicographic ascending initializer strengths", initializers=list(INITIALIZERS),
            penalty_pairs=[list(p) for p in PENALTIES], development_seed_ids=list(range(5)), assessment_seed_ids=list(range(5, 10)),
            development_group_ids=sorted(r["group"]["group_id"] for r in records), development_groups=165,
            development_root=str(root), development_plan_fingerprint=plan["plan_fingerprint"],
            development_source_manifest_sha256=plan["source_manifest_sha256"],
            split_fingerprint=plan["cheap_bic"]["split_fingerprint"],
            source_files_fingerprint=digest(read(root / "source-manifest.json")["files"]),
            scientific_identity=scientific_identity(plan), pair_scores=table,
            evaluation_metrics_used=False, truth_used_for_selection=False, validation_used_for_selection=False)
        artifact["artifact_fingerprint"] = digest(artifact)
        path = output / "selected-pair.json"
        contents = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if path.exists():
            require(not path.is_symlink() and path.read_text() == contents, "frozen pair artifact already exists with different content")
        else:
            with path.open("x") as stream:
                stream.write(contents)
        base.write_csv(output / "initializer-pair-scores.csv", table)
        report.update(success=True, selected_pair=artifact["selected_init_penalties"], pair_file=str(path),
                      pair_file_sha256=base.sha(path), selected_state_audits=audits, pair_scores=table)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(dict(error=str(error)))
    base.write_json(output / "development-report.json", report)
    return report


def load_pair(path):
    artifact = read(path)
    require(artifact.get("schema_version") == 1 and artifact.get("method") == "SparseSMARTv2CheapBICPair",
            "unsupported initializer pair artifact")
    require(digest({k: v for k, v in artifact.items() if k != "artifact_fingerprint"}) == artifact.get("artifact_fingerprint"),
            "initializer pair artifact fingerprint mismatch")
    require(tuple(artifact["selected_init_penalties"]) in PAIRS and artifact.get("iterations") == 200,
            "invalid frozen initializer pair or budget")
    require(artifact.get("initializers") == list(INITIALIZERS) and
            tuple(map(tuple, artifact["penalty_pairs"])) == PENALTIES and
            artifact.get("selection_criterion") == "group_equal_mean_(pair_BIC-C_BIC)/(n*q)" and
            artifact.get("tie_rule") == "lexicographic ascending initializer strengths",
            "pair artifact uses a different grid or selection rule")
    require(artifact.get("development_groups") == 165 and len(set(artifact["development_group_ids"])) == 165 and
            artifact["development_seed_ids"] == list(range(5)) and artifact["assessment_seed_ids"] == list(range(5, 10)),
            "invalid frozen development/assessment cohorts")
    require(all(artifact.get(key) is False for key in
                ("evaluation_metrics_used", "truth_used_for_selection", "validation_used_for_selection")),
            "pair was not selected using training BIC only")
    table = artifact["pair_scores"]
    require(len(table) == 15 and {tuple(row["init_penalties"]) for row in table} == set(PAIRS) and
            all(row["groups"] == 165 and base.finite(row["mean_normalized_bic_regret"]) and row["mean_normalized_bic_regret"] >= 0 for row in table),
            "incomplete pair comparison cohort")
    selected = min(table, key=lambda row: (row["mean_normalized_bic_regret"], tuple(row["init_penalties"])))
    require(selected["init_penalties"] == artifact["selected_init_penalties"], "frozen pair differs from deterministic selection")
    return artifact


def seed_block_statistics(rows, value_key, *, expected_seeds=None):
    blocks = defaultdict(list)
    for row in rows:
        require(base.finite(row[value_key]), f"nonfinite assessment metric: {value_key}")
        blocks[row["seed_id"]].append(row[value_key])
    if expected_seeds is not None:
        require(set(blocks) == set(expected_seeds), "incomplete seed blocks")
        require(len({len(values) for values in blocks.values()}) == 1, "unequal group coverage across seed blocks")
    values = [statistics.mean(blocks[seed]) for seed in sorted(blocks)]
    mean = statistics.mean(values) if values else None
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    if se is not None:
        from scipy.stats import t
        radius = float(t.ppf(.975, len(values) - 1)) * se
    else:
        radius = None
    return dict(seed_blocks=len(values), groups=len(rows), mean=mean, se=se,
                ci95_low=None if radius is None else mean-radius, ci95_high=None if radius is None else mean+radius,
                ci_method="Student t interval over equally weighted Monte Carlo seed-block means",
                seed_block_means={str(seed): statistics.mean(blocks[seed]) for seed in sorted(blocks)})


def assessment_tables(plan, groups):
    by_group = {record["group"]["group_id"]: record for record in groups}
    group_rows, alias_rows = [], []
    for record in groups:
        group = record["group"]
        for method in ("C", "D"):
            selected = record[method]
            group_rows.append(dict(group_id=group["group_id"], model_id=group["model_id"], seed_id=group["seed_id"],
                protocol=group["protocol"], method=method, candidate_id=selected["candidate_id"], task_id=selected["task_id"],
                selected_rank=selected["rank"], free_directions=selected["free_directions"], init_penalty=selected["init_penalty"],
                penalty_u=selected["penalty_u"], penalty_v=selected["penalty_v"], fit_method=selected["fit_method"],
                bic=selected["selection"]["score"], selected_iteration=selected["selected_iteration"],
                termination_reason=selected["termination_reason"], optimization_converged=selected["optimization_converged"],
                **{key: selected["metrics"][key] for key in METRICS}))
    for case in plan["display_cases"]:
        if case["group_id"] not in by_group:
            continue
        record = by_group[case["group_id"]]
        for method in ("C", "D"):
            alias_rows.append(dict(case_id=case["case_id"], group_id=case["group_id"], model_id=case["model_id"],
                experiment_id=case["experiment_id"], setting_index=case["setting_index"], seed_id=case["seed_id"],
                x=base.xvalue(case), method=method, **{key: record[method]["metrics"][key] for key in METRICS}))
    paired = []
    for record in groups:
        paired.append(dict(group_id=record["group"]["group_id"], model_id=record["group"]["model_id"],
            seed_id=record["group"]["seed_id"], protocol=record["group"]["protocol"],
            **{key: record["D"]["metrics"][key]-record["C"]["metrics"][key] for key in METRICS}))
    curves = []
    bins = defaultdict(list)
    for row in alias_rows:
        bins[(row["model_id"], row["experiment_id"], row["setting_index"], row["method"])].append(row)
    for key, rows in sorted(bins.items()):
        for metric in METRICS:
            curves.append(dict(model_id=key[0], experiment_id=key[1], setting_index=key[2], method=key[3], x=rows[0]["x"],
                metric=metric, **seed_block_statistics(rows, metric, expected_seeds=range(5, 10))))
    overall = [{"comparison": "D_minus_C", "metric": metric,
                **seed_block_statistics(paired, metric, expected_seeds=range(5, 10))} for metric in METRICS]
    return group_rows, alias_rows, paired, curves, overall


def plot_assessment(output, curves):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    files = []
    labels = ("Training observations", "Supplied rank (display alias)", "Supplied free count (display alias)", "Source-noise standard deviation")
    with PdfPages(output / "assessment-comparison.pdf") as pdf:
        for model in range(3):
            for metric, ylabel in METRICS.items():
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                for experiment, ax in enumerate(axes.flat):
                    subset = [r for r in curves if r["model_id"] == model and r["experiment_id"] == experiment and r["metric"] == metric]
                    ticks = sorted({r["x"] for r in subset})
                    for method, color in (("C", "#087E8B"), ("D", "#A84B26")):
                        rows = sorted((r for r in subset if r["method"] == method), key=lambda r: r["x"])
                        ax.errorbar([ticks.index(r["x"]) if experiment == 3 else r["x"] for r in rows],
                            [r["mean"] for r in rows], yerr=[r["ci95_high"]-r["mean"] for r in rows],
                            marker="o", capsize=2, color=color, label="C: six initializers" if method == "C" else "D: frozen pair")
                    ax.set_xlabel(labels[experiment]); ax.set_ylabel(ylabel.replace(" (", "\n(")); ax.grid(alpha=.2)
                    if experiment == 3:
                        ax.set_xticks(range(len(ticks)), [f"{v:g}" for v in ticks])
                    else:
                        ax.set_xticks(ticks)
                    ax.set_title(f"Experiment {experiment + 1}", loc="left")
                fig.suptitle(f"Model {('I', 'II', 'III')[model]} · RSC rank · terminal training BIC · T = 200")
                axes.flat[0].legend(frameon=False)
                fig.text(.03, .015, "95% t intervals across five assessment seed blocks. Repeated rank/free-count aliases are not independent observations.", fontsize=9)
                fig.tight_layout(rect=(0, .035, 1, .96))
                name = f"model-{model+1}-{metric}.png"
                fig.savefig(output / name, dpi=160); files.append(name); pdf.savefig(fig); plt.close(fig)
    return files + ["assessment-comparison.pdf"]


def exhaustive_inventory(root, plan):
    if root is None:
        return dict(A="unavailable_no_library", B="unavailable_no_library", performance_used=False)
    try:
        old = read(Path(root) / "plan.json")
        base.require(digest({k: v for k, v in old.items() if k != "plan_fingerprint"}) == old["plan_fingerprint"], "exhaustive plan hash mismatch")
        budget = old["configuration"]["iterations"]
        # A terminal result under a 2,000 budget cannot substitute for a T=200 fit.
        # Earlier stationarity can be considered only after a separate exact
        # candidate/provenance inventory; it does not establish whole-bank coverage.
        status = "unavailable_exact_200_budget" if budget != 200 else "unavailable_pending_complete_compatible_candidate_audit"
        return dict(A=status, B=status, performance_used=False, root=str(root), iterations=budget,
                    plan_fingerprint=old["plan_fingerprint"],
                    early_stationarity_potential="candidate-level compatibility audit required; no terminal-2000 substitution or forced fits")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(A="unavailable_invalid_library", B="unavailable_invalid_library", performance_used=False, error=str(error))


def timing_summary(root, plan, pair):
    """Actual C work and reconstructed D work; neither is allocation wall time."""
    def total(values):
        return sum(values) if all(base.finite(v) and v >= 0 for v in values) else None
    by_group = defaultdict(list)
    for task in plan["tasks"]:
        by_group[task["group_id"]].append(task)
    rows = []
    for group in plan["groups"]:
        results = [(task, read(root / "tasks" / f"{task['task_id']:06d}" / "result.json"))
                   for task in by_group[group["group_id"]]]
        selected = [(task, result) for task, result in results if task["init_penalty"] in pair]
        rrr = next(row for _, result in results for row in result["outcomes"] if row["fit_method"] == "target_rrr")
        included = any(task["include_rrr"] for task, _ in selected)
        rrr_cpu = 0. if included else rrr.get("process_cpu_seconds")
        rrr_wall = 0. if included else rrr.get("elapsed_seconds")
        rows.append(dict(group_id=group["group_id"], dataset_id=group["dataset_id"], seed_id=group["seed_id"],
            C_actual_task_process_cpu_seconds=total([r.get("process_cpu_seconds") for _, r in results]),
            C_actual_task_elapsed_work_seconds=total([r.get("elapsed_seconds") for _, r in results]),
            D_reconstructed_process_cpu_seconds=total([r.get("process_cpu_seconds") for _, r in selected] + [rrr_cpu]),
            D_reconstructed_elapsed_work_seconds=total([r.get("elapsed_seconds") for _, r in selected] + [rrr_wall]),
            D_selected_initializer_tasks=[task["task_id"] for task, _ in selected],
            D_RRR_already_in_selected_task=included, D_RRR_additional_process_cpu_seconds=rrr_cpu,
            D_RRR_additional_elapsed_seconds=rrr_wall, RRR_reused=bool(rrr.get("reuse_provenance")),
            RRR_original_elapsed_seconds=rrr.get("original_elapsed_seconds"),
            RRR_current_elapsed_seconds=rrr.get("elapsed_seconds")))
    rank_path = root.parent / "rank-selection-timing.json"
    rank = dict(available=False, process_cpu_seconds=None, elapsed_work_seconds=None,
                accounting="one rank estimate per unique dataset, shared across spectral protocols")
    if rank_path.exists():
        records = read(rank_path)["records"]
        wanted = {g["dataset_id"] for g in plan["groups"]}
        records = [record for record in records if record["dataset_id"] in wanted]
        complete = len(records) == len(wanted) and {r["dataset_id"] for r in records} == wanted
        rank.update(available=complete, datasets=len(records), expected_datasets=len(wanted),
            process_cpu_seconds=total([r.get("process_cpu_seconds") for r in records]) if complete else None,
            elapsed_work_seconds=total([r.get("elapsed_seconds") for r in records]) if complete else None,
            source_path=str(rank_path), source_sha256=base.sha(rank_path))
    totals = {key: total([row[key] for row in rows]) for key in (
        "C_actual_task_process_cpu_seconds", "C_actual_task_elapsed_work_seconds",
        "D_reconstructed_process_cpu_seconds", "D_reconstructed_elapsed_work_seconds")}
    totals["C_cpu_plus_once_per_dataset_rank_seconds"] = total([totals["C_actual_task_process_cpu_seconds"], rank["process_cpu_seconds"]])
    totals["D_reconstructed_cpu_plus_once_per_dataset_rank_seconds"] = total([totals["D_reconstructed_process_cpu_seconds"], rank["process_cpu_seconds"]])
    return dict(group_timings=rows, totals=totals, rank_estimation=rank,
        D_interpretation="reconstructed from the two selected initializer tasks, plus one RRR endpoint if absent from those tasks; includes initializer and task overhead",
        C_interpretation="actual six-initializer bank task work, including any audited reuse of RRR",
        missing_cpu_policy="CPU totals are unavailable if a required task or separately added RRR CPU measurement is absent; wall seconds are never substituted for CPU",
        standalone_D_runtime_measured=False,
        standalone_note="D was selected from the measured C bank, not timed as an independent run. Historical RRR compute time is reported separately from current reuse work.",
        retry_note="task timers cover the invocation that finalized result.json; work in earlier interrupted invocations whose blocks were reused is not included unless separately inventoried",
        wall_time_note="sum of task elapsed times measures work, not parallel allocation elapsed time; no scheduler or source preparation time is included")


def assess(root, pair_file, output_root=None, *, exhaustive_root=None, make_plots=True):
    # This guard intentionally precedes reading the assessment plan or outcomes.
    pair = load_pair(pair_file)
    root = Path(root).resolve()
    output = output_path(root, output_root, "assessment")
    report = dict(schema_version=1, phase="assess", success=False, errors=[], pair_file=str(Path(pair_file).resolve()),
                  pair_file_sha256=base.sha(pair_file), selected_init_penalties=pair["selected_init_penalties"],
                  pair_frozen_before_assessment=True, selection_uses_evaluation_metrics=False, metric_labels=METRICS,
                  complete_execution_coverage=False, complete_tuning_coverage=False)
    try:
        plan, records, errors = read_bank(root, "assess")
        require(scientific_identity(plan) == pair["scientific_identity"], "assessment scientific settings differ from development")
        require(plan["cheap_bic"]["split_fingerprint"] == pair["split_fingerprint"], "assessment split differs from frozen development split")
        require(digest(read(root / "source-manifest.json")["files"]) == pair["source_files_fingerprint"], "assessment source differs from development")
        require(not set(pair["development_group_ids"]) & {g["group_id"] for g in plan["groups"]}, "development/assessment group overlap")
        report.update(inventory(records), errors=errors, planned_groups=165, plan_fingerprint=plan["plan_fingerprint"],
                      complete_execution_coverage=not errors and len(records) == 165,
                      complete_tuning_coverage=not errors and len(records) == 165 and
                          all(row["eligible"] for record in records for row in record["rows"]))
        require(not errors and len(records) == 165, "assessment requires all 165 complete groups")
        selections, audits = [], []
        for record in records:
            c, d = winner(record["rows"]), winner(record["rows"], pair["selected_init_penalties"])
            # Evaluation arrays and metrics enter only after pair validation and
            # all training-only coverage/selection checks above have succeeded.
            group = record["group"]
            prep = read(root / "preparation.json")
            prepared = next(v for v in prep["groups"] if v["group_id"] == group["group_id"])
            _, data, source = base.load_group(root, plan, group, prepared)
            full = {}
            for row in {r["candidate_id"]: r for r in (c, d)}.values():
                result = read(root / "tasks" / f"{row['task_id']:06d}" / "result.json")
                outcome = next(v for v in result["outcomes"] if v["candidate_id"] == row["candidate_id"])
                chosen = dict(row, metrics=outcome["metrics"])
                audits.append(base.audit_winner(root, chosen, data, source, group))
                full[row["candidate_id"]] = chosen
            selections.append(dict(group=group, C=full[c["candidate_id"]], D=full[d["candidate_id"]]))
        group_rows, aliases, paired, curves, overall = assessment_tables(plan, selections)
        for name, rows in (("group-results.csv", group_rows), ("display-alias-results.csv", aliases),
                ("paired-group-differences.csv", paired), ("performance-curves.csv", curves), ("paired-seed-block-summary.csv", overall)):
            base.write_csv(output / name, rows)
        report.update(success=True, groups=selections, selected_state_audits=audits, paired_seed_block_summary=overall,
            selected_rrr_counts={method: sum(r[method]["fit_method"] == "target_rrr" for r in selections) for method in ("C", "D")},
            selected_termination_counts={method: dict(Counter(r[method]["termination_reason"] for r in selections)) for method in ("C", "D")},
            exhaustive_comparators=exhaustive_inventory(exhaustive_root or plan.get("exhaustive_root"), plan))
        report["runtime"] = timing_summary(root, plan, pair["selected_init_penalties"])
        base.write_csv(output / "group-runtime.csv", report["runtime"]["group_timings"])
        report["figures"] = plot_assessment(output, curves) if make_plots else []
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["success"] = False
        report["errors"].append(dict(error=str(error)))
    base.write_json(output / "assessment-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("develop", "assess"):
        sub = commands.add_parser(command)
        sub.add_argument("--root", required=True, type=Path)
        sub.add_argument("--output-root", type=Path)
        if command == "assess":
            sub.add_argument("--pair-file", required=True, type=Path)
            sub.add_argument("--exhaustive-root", type=Path)
            sub.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = develop(args.root, args.output_root) if args.command == "develop" else assess(
            args.root, args.pair_file, args.output_root, exhaustive_root=args.exhaustive_root, make_plots=not args.no_plots)
        print(json.dumps({key: result.get(key) for key in ("phase", "success", "complete_groups", "selected_pair", "selected_init_penalties", "errors")}, indent=2))
        return 0 if result["success"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
