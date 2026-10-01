#!/usr/bin/env python3
"""Read-in-place checkpoint/terminal BIC audits and paired campaign reporting.

All winner selection uses training BIC. Evaluation data enter only state audits
and reporting. Checkpoint scores authenticate the saved score inventory; only
retained winning coefficients can be independently rescored from saved states.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import statistics

import summarize_sparse_smart_v2_bic as base

METHODS = ("fixed_checkpoint_bic", "automatic_checkpoint_bic", "fixed_terminal_bic",
           "automatic_terminal_bic", "previous_validation")
POLICY = "bic_checkpoint"


def terminal_view(row):
    result = dict(row)
    if row["eligible"]:
        result.update(selection=row["terminal_selection"], metrics=row["terminal_metrics"],
                      selected_iteration=row["n_iter"], state_key=row.get("terminal_state_key"))
    return result


def select(rows, group, case=None, *, terminal=False):
    """RSC automatic scope cannot accidentally select a fixed-only rank/count."""
    candidates = []
    for row in rows:
        if not row["eligible"]:
            continue
        rank = group["selected_target_rank"] if case is None else case["rank"]
        if row["rank"] != rank:
            continue
        if row["fit_method"] != "target_rrr":
            counts = row["free_directions"]
            if case is not None and counts != case["free_directions"]:
                continue
            if case is None and (counts[0] != counts[1] or counts[0] not in group["automatic_free_counts"]):
                continue
        candidates.append(terminal_view(row) if terminal else row)
    return min(candidates, key=base.outcome_key, default=None)


def compact(row):
    if row is None:
        return None
    return {key: value for key, value in row.items() if key not in
            ("checkpoint_scores", "terminal_metrics", "terminal_selection", "terminal_state_key",
             "initialization_diagnostics", "numerical_work", "traceback")}


def _load(root):
    plan, by_group, prepared = base.load_plan(root, selection_rule=POLICY)
    config = plan["configuration"]
    base.require(config["iterations"] == 200 and config["init_penalties"] == [.003, .1],
                 "checkpoint campaign estimator configuration changed")
    base.require(config["checkpoint_iterations"] == [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200],
                 "checkpoint grid changed")
    base.require(config["penalty_pairs"] == [[.001, .001], [.0025, .0025], [.01, .01]],
                 "checkpoint campaign penalty grid changed")
    return plan, by_group, prepared


def audit(root, task_ids=None):
    """Operational gate: full task provenance and every retained canary state.

Scientific initializer exclusions remain reported outcomes. Missing executions,
invalid scores, or failed saved-state audits prevent production release.
"""
    root = Path(root).resolve()
    report = dict(success=False, errors=[], outcome_counts={}, audited_states=0,
                  completed_tasks=0, requested_task_ids=task_ids)
    counts = Counter()
    try:
        plan, by_group, prepared = _load(root)
        wanted = set(range(plan["n_tasks"]) if task_ids is None else task_ids)
        base.require(bool(wanted) and all(type(tid) is int and 0 <= tid < plan["n_tasks"] for tid in wanted),
                     "invalid audit task subset")
        if task_ids is not None:
            base.require(len(wanted) == len(task_ids), "duplicate audit task IDs")
        report.update(plan_fingerprint=plan["plan_fingerprint"], expected_tasks=len(wanted))
        for group in plan["groups"]:
            members = [task for task in by_group[group["group_id"]] if task["task_id"] in wanted]
            if not members:
                continue
            meta, data, source = base.load_group(root, plan, group, prepared[group["group_id"]])
            for task in members:
                rows = base.check_task(root, plan, group, task, meta, selection_rule=POLICY)
                counts.update(row["classification"] for row in rows)
                blocks = defaultdict(list)
                for row in rows:
                    if row["eligible"]:
                        blocks[(row["rank"], tuple(row["free_directions"]), row["fit_method"])].append(row)
                for members in blocks.values():
                    for winner in (min(members, key=base.outcome_key),
                                   min((terminal_view(r) for r in members), key=base.outcome_key)):
                        base.require(isinstance(winner.get("state_key"), str),
                                     "required checkpoint/terminal block winner has no retained state")
                checked = set()
                for original in rows:
                    if not original["eligible"]:
                        continue
                    for row in (original, terminal_view(original)):
                        if row.get("state_key") is not None and row["state_key"] not in checked:
                            base.audit_winner(root, row, data, source, group)
                            checked.add(row["state_key"])
                            report["audited_states"] += 1
                report["completed_tasks"] += 1
        base.require(report["completed_tasks"] == len(wanted), "incomplete task audit")
        base.require(not counts["execution_failure"], "candidate execution failure")
        base.require(counts["eligible"] > 0 and report["audited_states"] > 0, "no usable audited candidate")
        report["success"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(str(error))
    report["outcome_counts"] = dict(counts)
    return report


def _stats(values):
    from scipy.stats import t
    result = base.statistics_of(values)
    hw = float(t.ppf(.975, len(values)-1)) * result["se"] if len(values) > 1 else None
    result.update(lower95=None if hw is None else result["mean"]-hw,
                  upper95=None if hw is None else result["mean"]+hw)
    return result


def _baseline_records(root):
    path = root / "comparison-baselines.json"
    if not path.exists():
        return {}, None
    payload = base.read(path)
    config = base.read(root / "campaign.json")
    expected = config.get("comparison_baselines_sha256")
    base.require(expected is not None, "comparison baselines lack frozen campaign hash")
    base.check_hash(path, expected)
    rows = payload["case_results"]
    mapping = defaultdict(dict)
    for row in rows:
        base.require(row["method"] not in mapping[row["case_id"]], "duplicate comparator case/method")
        mapping[row["case_id"]][row["method"]] = row
    return mapping, dict(path=str(path), sha256=expected, source_summary_sha256=payload["source_summary_sha256"])


def build_tables(records):
    flat, curves, differences, seed_blocks = [], [], [], []
    settings = defaultdict(list)
    blocks = defaultdict(list)
    for record in records:
        case = record["case"]
        settings[(case["model_id"], case["experiment_id"], case["setting_index"])].append(record)
        common = {k: case[k] for k in ("case_id", "group_id", "model_id", "experiment_id", "setting_index", "seed_id")}
        for method, winner in record["methods"].items():
            row = dict(common, x=base.xvalue(case), method=method, available=winner is not None,
                       complete_execution_coverage=record["complete_execution_coverage"])
            if winner is not None:
                row.update(winner["metrics"], selected_rank=winner.get("rank"),
                    free_directions=winner.get("free_directions"), selected_iteration=winner.get("selected_iteration"),
                    init_penalty=winner.get("init_penalty"), penalty_u=winner.get("penalty_u"), penalty_v=winner.get("penalty_v"),
                    fit_method=winner.get("fit_method"), termination_reason=winner.get("termination_reason"),
                    optimization_converged=winner.get("optimization_converged"),
                    bic=None if winner.get("selection") is None else winner["selection"]["score"])
            flat.append(row)
        targets = [name for name in record["methods"] if name not in ("automatic_checkpoint_bic",)]
        chosen = record["methods"].get("automatic_checkpoint_bic")
        for target in targets:
            ref = record["methods"].get(target)
            if chosen is not None and ref is not None:
                for metric in ("coefficient_rmse", "training_prediction_mse"):
                    if metric in ref["metrics"]:
                        delta = chosen["metrics"][metric]-ref["metrics"][metric]
                        blocks[(case["model_id"], case["experiment_id"], case["seed_id"], target, metric)].append(delta)
    for key, members in sorted(settings.items()):
        methods = sorted({name for record in members for name in record["methods"]})
        for method in methods:
            wins = [record["methods"].get(method) for record in members]
            wins = [w for w in wins if w is not None]
            for metric in ("coefficient_rmse", "training_prediction_mse", "validation_mse"):
                vals = [w["metrics"][metric] for w in wins if metric in w["metrics"]]
                if not vals:
                    continue
                curves.append(dict(model_id=key[0], experiment_id=key[1], setting_index=key[2],
                    x=base.xvalue(members[0]["case"]), method=method, metric=metric,
                    requested_seeds=len(members), **_stats(vals)))
        for method, target in (("automatic_checkpoint_bic", "automatic_terminal_bic"),
                               ("fixed_checkpoint_bic", "fixed_terminal_bic"),
                               ("automatic_checkpoint_bic", "previous_validation"),
                               ("fixed_checkpoint_bic", "previous_validation")):
            pairs = [(r["methods"].get(method), r["methods"].get(target)) for r in members]
            vals = [a["metrics"]["coefficient_rmse"]-b["metrics"]["coefficient_rmse"]
                    for a, b in pairs if a is not None and b is not None]
            differences.append(dict(model_id=key[0], experiment_id=key[1], setting_index=key[2],
                method=method, reference=target, metric="coefficient_rmse", **_stats(vals)))
    grouped = defaultdict(list)
    for (model, experiment, seed, reference, metric), values in blocks.items():
        # Average repeated settings within seed before computing uncertainty.
        delta = statistics.mean(values)
        for cohort in ("all", "design_seeds_0_9" if seed < 10 else "seeds_10_99"):
            grouped[(model, experiment, reference, metric, cohort)].append(delta)
    for (model, experiment, reference, metric, cohort), values in sorted(grouped.items()):
        seed_blocks.append(dict(model_id=model, experiment_id=experiment, reference=reference,
            metric=metric, cohort=cohort, method="automatic_checkpoint_bic", **_stats(values)))
    return flat, curves, differences, seed_blocks


def plot(output, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"automatic_checkpoint_bic": "Automatic: checkpoint BIC",
             "fixed_checkpoint_bic": "Fixed: checkpoint BIC", "automatic_terminal_bic": "Automatic: terminal BIC",
             "fixed_terminal_bic": "Fixed: terminal BIC", "previous_validation": "Previous v2: validation",
             "source_subspace_ridge_rrr": "Source-subspace ridge RRR", "target_ridge_rrr": "Target ridge RRR",
             "initializer_only": "Raw Lasso-SVD initializer"}
    colors = ["#c45325", "#146b93", "#d8a28b", "#91bacd", "#20844f", "#79589f", "#555555", "#ac8b12"]
    paths = []
    for model in range(3):
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
        for experiment, ax in enumerate(axes):
            for (method, label), color in zip(names.items(), colors):
                items = sorted((r for r in curves if r["model_id"] == model and r["experiment_id"] == experiment
                                and r["method"] == method and r["metric"] == "coefficient_rmse"), key=lambda r: r["x"])
                if items:
                    ax.errorbar([r["x"] for r in items], [r["mean"] for r in items],
                                yerr=[0 if r["lower95"] is None else r["mean"]-r["lower95"] for r in items],
                                color=color, marker="o", markersize=3, linewidth=1.4,
                                linestyle="--" if "terminal" in method else "-", label=label)
            ax.set_xlabel(("Training sample size", "Imposed target rank", "Imposed free count", "Source noise SD")[experiment])
            ax.set_title(f"Experiment {experiment+1}"); ax.grid(alpha=.2)
            if experiment == 0:
                ax.set_ylabel("Coefficient RMSE")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=9)
        fig.suptitle(f"Model {model+1}: paired frozen-grid comparison; pointwise 95% seed intervals", fontsize=12)
        name = f"model-{model+1}-coefficient-rmse.png"
        fig.savefig(output / name, dpi=160); plt.close(fig); paths.append(name)
    return paths


def summarize(root, output=None, *, make_plots=True):
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, method="SparseSMARTv2CheckpointBIC", root=str(root), success=False,
        checked_utc=base.datetime.now(base.timezone.utc).isoformat(), errors=[], case_results=[], group_results=[],
        selection_rule="minimum training BIC over candidate/checkpoint; ties by task ID and candidate ID, then earliest checkpoint within trajectory",
        evaluation_used_for_selection=False, validation_mse_is_selection_set_error_for_comparators=True,
        uncertainty="pointwise Student t intervals; repeated settings averaged within simulation seed",
        audit_scope="all task/source/data hashes and score inventories; independently rescore retained selected checkpoint and terminal winners",
        early_checkpoint_selection_does_not_shorten_actual_trajectory=True)
    counts, tasks, audits, runtimes = Counter(), Counter(), [], []
    try:
        plan, by_group, prepared = _load(root)
        reference, reference_hashes = base.load_reference(plan["reference_root"])
        baselines, baseline_provenance = _baseline_records(root)
        report.update(plan_fingerprint=plan["plan_fingerprint"], configuration=plan["configuration"],
            planned_tasks=plan["n_tasks"], planned_candidates=plan["n_candidates"],
            reference_hashes=reference_hashes, baseline_provenance=baseline_provenance)
        cases_by_group = defaultdict(list)
        for case in plan["display_cases"]:
            cases_by_group[case["group_id"]].append(case)
        for group in plan["groups"]:
            gid = group["group_id"]
            meta, data, source = base.load_group(root, plan, group, prepared[gid])
            rows, complete = [], True
            for task in by_group[gid]:
                try:
                    rows.extend(base.check_task(root, plan, group, task, meta, selection_rule=POLICY))
                    task_result = base.read(root / "tasks" / f"{task['task_id']:06d}" / "result.json")
                    runtimes.append(dict(task_id=task["task_id"], group_id=gid,
                        elapsed_seconds=task_result["elapsed_seconds"], process_cpu_seconds=task_result.get("process_cpu_seconds")))
                    tasks["complete"] += 1
                except (OSError, ValueError, KeyError, TypeError) as error:
                    complete = False; tasks["failed_or_missing"] += 1
                    report["errors"].append(dict(task_id=task["task_id"], error=str(error)))
            counts.update(row["classification"] for row in rows)
            complete = complete and not any(row["classification"] == "execution_failure" for row in rows)
            auto = select(rows, group) if complete else None
            terminal = select(rows, group, terminal=True) if complete else None
            selected = {}
            def register(winner):
                if winner is not None:
                    selected[(winner["task_id"], winner["state_key"])] = winner
                return compact(winner)
            group_record = dict(group=group, automatic_checkpoint_bic=register(auto),
                automatic_terminal_bic=register(terminal), complete_execution_coverage=complete,
                outcome_counts=dict(Counter(row["classification"] for row in rows)))
            report["group_results"].append(group_record)
            for case in cases_by_group[gid]:
                old = base.reference_winner(case, reference[case["case_id"]], meta)
                methods = dict(automatic_checkpoint_bic=compact(auto), automatic_terminal_bic=compact(terminal),
                    fixed_checkpoint_bic=register(select(rows, group, case)) if complete else None,
                    fixed_terminal_bic=register(select(rows, group, case, terminal=True)) if complete else None,
                    previous_validation=old)
                for name, other in baselines.get(case["case_id"], {}).items():
                    for key in ("model_id", "experiment_id", "setting_index", "seed_id", "random_seed", "n_train", "p", "q", "sigma0"):
                        base.require(other[key] == case[key], f"comparator identity differs: {case['case_id']} {key}")
                    methods[name] = dict(metrics=other["metrics"], rank=other.get("rank"), fit_method=name,
                                         selection=None, selected_iteration=None)
                if any(methods[name] is None for name in METHODS):
                    report["errors"].append(dict(case_id=case["case_id"], error="missing complete-library winner"))
                report["case_results"].append(dict(case=case, methods=methods, complete_execution_coverage=complete))
            for winner in selected.values():
                audits.append(base.audit_winner(root, winner, data, source, group))
            if len(report["group_results"]) % 100 == 0:
                print(json.dumps(dict(summarized_groups=len(report["group_results"]), errors=len(report["errors"])), sort_keys=True), flush=True)
        report["complete_execution_coverage"] = tasks["complete"] == plan["n_tasks"] and not counts["execution_failure"]
        report["complete_tuning_coverage"] = report["complete_execution_coverage"] and all(k == "eligible" or v == 0 for k, v in counts.items())
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(dict(error=str(error)))
        report["complete_execution_coverage"] = False
    report.update(outcome_counts=dict(counts), task_counts=dict(tasks), selected_state_audits=audits,
        task_cpu_seconds=sum(r["process_cpu_seconds"] or 0 for r in runtimes),
        task_elapsed_seconds_sum=sum(r["elapsed_seconds"] for r in runtimes))
    for name in ("rank-selection-timing.json", "campaign-preparation.json"):
        if (root / name).exists():
            report[name.removesuffix(".json")] = base.read(root / name)
    flat, curves, differences, seed_blocks = build_tables(report["case_results"])
    files = []
    for name, values in (("per-seed-results.csv", flat), ("performance-curves.csv", curves),
                         ("paired-differences.csv", differences), ("paired-seed-blocks.csv", seed_blocks),
                         ("task-runtime.csv", runtimes)):
        base.write_csv(output / name, values); files.append(name)
    report["setting_results"] = curves
    report["paired_seed_blocks"] = seed_blocks
    report["automatic_selection"] = {}
    for policy in ("automatic_checkpoint_bic", "automatic_terminal_bic"):
        winners = [g[policy] for g in report["group_results"] if g[policy] is not None]
        report["automatic_selection"][policy] = dict(n_groups=len(winners),
            ranks=dict(Counter(w["rank"] for w in winners)),
            free_counts=dict(Counter(str(w["free_directions"]) for w in winners)),
            iterations=dict(Counter(w["selected_iteration"] for w in winners)),
            initializers=dict(Counter(str(w["init_penalty"]) for w in winners if w["fit_method"] != "target_rrr")),
            penalties=dict(Counter(str(w["penalty_u"]) for w in winners if w["fit_method"] != "target_rrr")),
            rrr_selected=sum(w["fit_method"] == "target_rrr" for w in winners))
    if make_plots and curves:
        files.extend(plot(output, curves))
    report["success"] = not report["errors"] and report["complete_execution_coverage"]
    report["output_hashes"] = {name: base.sha(output / name) for name in files}
    base.write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    result = audit(args.root, args.task_ids) if args.audit else summarize(args.root, args.output_root, make_plots=not args.no_plots)
    print(json.dumps({key: result.get(key) for key in ("success", "errors", "task_counts", "outcome_counts", "audited_states")}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
