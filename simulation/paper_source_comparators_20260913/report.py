"""Summarize saved paper-grid comparator outputs without numerical refitting.

Every displayed point uses one observation per original simulation seed.
Experiment-3 aliases reuse fitted tasks, not additional independent replications.
Intervals are pointwise normal Monte Carlo intervals, not simultaneous bands.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


LABELS = {
    "v2": "SparseSMART v2",
    "initializer_only": "Raw Lasso–SVD initializer",
    "target_rrr": "Target RRR",
    "target_ridge_rrr": "Target ridge RRR",
    "source_subspace_rrr": "Source-subspace RRR",
    "source_subspace_ridge_rrr": "Source-subspace ridge RRR",
    "ridge_to_source": "Ridge toward source",
    "source_target_mixture": "Source–target mixture",
    "nuclear_contrast": "Nuclear contrast",
}
BAD_EXECUTION = {"missing", "corrupt", "execution_failure"}
INTERVAL = "pointwise normal 95% Monte Carlo interval; independent unit is simulation seed"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _relative(folder, name):
    path = (folder / name).resolve()
    _require(path.is_relative_to(folder.resolve()), "artifact path escapes task folder")
    return path


def _stats(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, mcse=None, lower95=None, upper95=None)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, mcse=se,
                lower95=None if se is None else mean - 1.959963984540054 * se,
                upper95=None if se is None else mean + 1.959963984540054 * se)


def _paired_diagnostics(rows):
    n = len(rows)
    differences = [row["paired_difference"] for row in rows]
    harmed = [value for value in differences if value > 1e-12]
    k, z = len(harmed), 1.959963984540054
    lower = upper = None
    if n:
        center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        lower, upper = (0. if k == 0 else center - radius), (1. if k == n else center + radius)
    positive = [row for row in rows if row["coefficient_rmse"] > 0 and row["v2_coefficient_rmse"] > 0]
    logs = _stats(math.log(row["coefficient_rmse"]) - math.log(row["v2_coefficient_rmse"]) for row in positive)
    ratio = {key: math.exp(logs[key]) if logs[key] is not None else None for key in ("mean", "lower95", "upper95")}
    ratio.update(n=logs["n"], log_mean=logs["mean"], log_mcse=logs["mcse"], excluded_zero_pairs=n-len(positive),
                 comparator_zero_pairs=sum(row["coefficient_rmse"] == 0 for row in rows),
                 v2_zero_pairs=sum(row["v2_coefficient_rmse"] == 0 for row in rows))
    return dict(geometric_rmse_ratio_vs_v2=ratio,
                higher_rmse_frequency=dict(n=n, count=k, proportion=k/n if n else None,
                    lower95=lower, upper95=upper, interval="pointwise Wilson 95%", tolerance=1e-12),
                positive_excess_rmse_over_all_pairs=_stats(max(0., value) for value in differences),
                excess_rmse_conditional_on_higher=_stats(harmed))


def _exit(folder, prefix, slurm):
    code = (folder / f"{prefix}-exit-code.txt").read_text().strip()
    _require(code.isdecimal(), f"invalid {prefix} exit code")
    with (folder / f"{prefix}-status.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    _require(len(rows) == 1 and rows[0].get("exit_code") == code,
             f"{prefix} status/exit-code mismatch")
    _require(rows[0].get("job_id") == str(slurm["job_id"]) and rows[0].get("finished_utc"),
             f"{prefix} Slurm job/time binding mismatch")
    if prefix == "process":
        _require(rows[0].get("step_id") == str(slurm["step_id"]), "process Slurm step mismatch")
    return int(code)


def check_task(root, plan, task, common):
    """Verify recorded execution/provenance and its selected-state audit receipt."""
    folder = root / "tasks" / f"{task['task_id']:06d}"
    result, status = common.read(folder / "result.json"), common.read(folder / "status.json")
    _require(status.get("result_sha256") == common.sha(folder / "result.json"),
             "result/status hash mismatch")
    identity = dict(task=task, group_id=task["group_id"],
                    plan_fingerprint=plan["plan_fingerprint"],
                    source_manifest_sha256=plan["source_manifest_sha256"])
    for key, expected in identity.items():
        _require(result.get(key) == expected and status.get(key) == expected,
                 f"result/status {key} identity mismatch")
    _require(isinstance(result.get("execution_success"), bool), "missing execution-success flag")
    _require(status.get("status") == "finished" and status.get("execution_success") == result["execution_success"],
             "execution status receipt mismatch")
    _require(isinstance(result.get("slurm"), dict) and result["slurm"] == status.get("slurm"), "Slurm result/status binding mismatch")
    exits = {prefix: _exit(folder, prefix, result["slurm"]) for prefix in ("process", "launcher")}
    files = result.get("files")
    _require(isinstance(files, dict), "missing result file manifest")
    for name, expected in files.items():
        _require(common.sha(_relative(folder, name)) == expected, f"file hash mismatch: {name}")
    row = dict(task_id=task["task_id"], group_id=task["group_id"], method=task["method"],
               rank=task.get("rank"), classification="execution_failure", eligible=False,
               elapsed_seconds=result.get("elapsed_seconds"), result_sha256=common.sha(folder / "result.json"),
               n_candidates=0, candidate_status_counts={}, audit_success=False)
    if not result["execution_success"] or any(exits.values()):
        row["error"] = "nonzero process/launcher exit or unsuccessful execution"
        return row
    _require(result.get("status") == "complete", "result is not complete")
    selected = result.get("selection")
    _require(isinstance(selected, dict) and selected.get("status") in
             ("success", "no_eligible_candidate", "selection_verification_failed"), "invalid selection record")
    candidates = selected.get("candidate_results")
    _require(isinstance(candidates, list), "missing candidate census")
    _require(selected.get("method") == task["method"] and selected.get("refit") is False,
             "selection method/refit mismatch")
    _require(selected.get("n_candidates") == len(candidates), "candidate count mismatch")
    eligible = []
    for index, candidate in enumerate(candidates):
        _require(candidate.get("index") == index and candidate.get("status") in
                 ("eligible", "uncertified", "failed"), "invalid candidate index/status")
        if candidate["status"] == "eligible":
            _require(_finite(candidate.get("validation_mse")) and candidate["validation_mse"] >= 0,
                     "invalid eligible validation score")
            eligible.append(index)
    _require(selected.get("n_eligible") == len(eligible), "eligible candidate count mismatch")
    row.update(n_candidates=len(candidates),
               candidate_status_counts=dict(Counter(str(c.get("status", "unknown")) for c in candidates)))
    _require(selected["status"] != "selection_verification_failed", "selection verification failed")
    if selected["status"] != "success":
        _require(not eligible, "failed selection contains eligible candidate")
        row.update(classification="method_failure", error=selected.get("error", selected.get("status")))
        return row
    _require(isinstance(result.get("audit"), dict) and result["audit"].get("success") is True,
             "selected-state audit did not pass")
    _require("selected.npz" in files, "successful fit lacks selected coefficient artifact")
    metrics = result.get("metrics", {})
    for name in ("coefficient_rmse", "coefficient_frobenius_squared", "validation_mse", "training_prediction_mse"):
        _require(_finite(metrics.get(name)) and metrics[name] >= 0, f"invalid selected metric {name}")
    index = selected.get("selected_index")
    _require(type(index) is int and 0 <= index < len(candidates), "invalid selected candidate index")
    _require(eligible and index == min(eligible, key=lambda i: (candidates[i]["validation_mse"], i)),
             "selected candidate is not the metadata validation winner")
    for score in (selected.get("validation_mse"), selected.get("validation_mse_recomputed"),
                  candidates[index]["validation_mse"]):
        _require(_finite(score) and math.isclose(score, metrics["validation_mse"], rel_tol=2e-9, abs_tol=2e-9),
                 "selected validation score mismatch")
    row.update(classification="eligible", eligible=True, audit_success=True,
               selected_index=index, selection_status=selected.get("status"),
               selected_parameters=selected.get("selected_parameters"),
               boundary_indicators=selected.get("boundary_indicators", {}),
               effective_rank=selected.get("effective_rank"), source_dimensions=selected.get("source_dimensions"), **metrics)
    return row


def _load_references(root, plan, common):
    preparation = common.read(root / "preparation.json")
    _require(preparation.get("success") is True and preparation.get("status") == "complete" and
             preparation.get("plan_sha256") == common.sha(root / "plan.json") and
             preparation.get("plan_fingerprint") == plan["plan_fingerprint"] and
             preparation.get("source_manifest_sha256") == plan["source_manifest_sha256"],
             "preparation plan/source binding mismatch")
    references = preparation.get("references")
    _require(isinstance(references, dict), "preparation lacks reference hash bindings")
    loaded, errors = {}, {}
    for reference in sorted({item["reference_root"] for item in plan["display_cases"]}):
        try:
            binding = references[reference]
            folder = Path(reference)
            for filename, key in (("plan.json", "plan_sha256"),
                                  ("preparation.json", "preparation_sha256"),
                                  ("summary/summary.json", "summary_sha256")):
                _require(common.sha(folder / filename) == binding[key], f"reference {filename} hash mismatch")
            original_plan = common.read(folder / "plan.json")
            _require(original_plan.get("method") == "SparseSMARTv2Pilot", "reference is not external-validation v2 campaign (BIC is excluded)")
            summary = common.read(folder / "summary/summary.json")
            _require(summary.get("plan_sha256") == binding["plan_sha256"] and
                     summary.get("plan_fingerprint") == original_plan.get("plan_fingerprint"),
                     "reference summary/plan identity mismatch")
            records = summary["case_results"]
            mapping = {record["case"]["case_id"]: record for record in records}
            _require(len(mapping) == len(records), "duplicate reference summary case")
            if "source_manifest_sha256" in binding:
                _require(common.sha(folder / "source-manifest.json") == binding["source_manifest_sha256"],
                         "reference source-manifest hash mismatch")
            if "numerical_audit_sha256" in binding:
                audit_path = folder / "summary/numerical-audit.json"
                _require(common.sha(audit_path) == binding["numerical_audit_sha256"], "reference numerical-audit hash mismatch")
                audit = common.read(audit_path)
                _require(audit.get("success") is True and not audit.get("errors"), "reference numerical audit failed")
                ledger_path = folder / "summary/numerical-audit-cases.jsonl"
                _require(common.sha(ledger_path) == binding["numerical_audit_cases_sha256"],
                         "reference numerical-audit case-ledger hash mismatch")
                checked, ledger_cases = {}, set()
                with ledger_path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        cid = row["case_id"]
                        _require(cid not in ledger_cases, "duplicate case in reference numerical audit")
                        ledger_cases.add(cid)
                        _require(row.get("success") is True and not row.get("errors"), "reference case numerical audit failed")
                        audited_tasks = row["tasks"]
                        _require(len({task["task_id"] for task in audited_tasks}) == len(audited_tasks) and
                                 set(row["task_ids"]) == {task["task_id"] for task in audited_tasks},
                                 "reference numerical audit task coverage differs")
                        for task in audited_tasks:
                            _require(task.get("success") is True and task.get("case_id") == cid,
                                     "reference numerical audit task failed or has wrong case")
                            checked[(cid, task["task_id"])] = task
                for display in plan["display_cases"]:
                    if display["reference_root"] != reference:
                        continue
                    record = mapping.get(display["case"]["case_id"], {})
                    winner = record.get("winner")
                    if winner is not None:
                        row = checked.get((display["case"]["case_id"], winner["task_id"]))
                        _require(row is not None,
                                 "reference winner is absent from the passing numerical audit")
            loaded[reference] = mapping
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors[reference] = str(error)
    return loaded, errors


def _case_columns(case):
    return {key: case.get(key) for key in ("case_id", "model_id", "experiment_id", "setting_index",
            "seed_id", "random_seed", "n_train", "p", "q", "rank", "source_rank", "sigma0")}


def display_rows(plan, tasks, references, reference_errors):
    """Expand aliases for display while retaining the unique task and seed IDs."""
    rows, seen = [], set()
    for display in plan["display_cases"]:
        case, reference = display["case"], display["reference_root"]
        columns = _case_columns(case)
        base = dict(columns, reference_root=reference, group_id=display["group_id"])
        ref = dict(base, method="v2", task_id=None, eligible=False, classification="missing",
                   coefficient_rmse=None, error=reference_errors.get(reference))
        if reference in references:
            record = references[reference].get(case["case_id"])
            if record is not None:
                if record.get("case") != case:
                    ref.update(classification="corrupt", error="reference display-case mismatch")
                else:
                    winner = record.get("winner")
                    if winner is not None and _finite(winner.get("coefficient_rmse")) and winner["coefficient_rmse"] >= 0:
                        ref.update(winner, **base, method="v2", eligible=True,
                                   classification="eligible" if record.get("complete_execution_coverage") is True
                                   else "partial_reference", reference_task_id=winner.get("task_id"), task_id=None)
                    else:
                        ref.update(classification="method_failure", error="reference has no eligible winner")
        rows.append(ref)
        for method, task_id in display["task_ids"].items():
            task = tasks[task_id]
            _require(task["group_id"] == display["group_id"] and task["method"] == method,
                     "display alias references an incompatible task")
            row = dict(task, **base, method=method)
            paired = row["eligible"] and ref["eligible"]
            row.update(v2_classification=ref["classification"],
                       v2_coefficient_rmse=ref.get("coefficient_rmse"),
                       paired_difference=(row["coefficient_rmse"] - ref["coefficient_rmse"]) if paired else None,
                       paired=paired)
            rows.append(row)
        for row in rows[-(len(display["task_ids"]) + 1):]:
            key = tuple(row.get(k) for k in ("model_id", "experiment_id", "setting_index", "seed_id", "method"))
            _require(key not in seen, "duplicate simulation seed within a displayed setting/method")
            seen.add(key)
    return rows


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_id"], row["experiment_id"], row["setting_index"], row["method"])].append(row)
    results = []
    for key, members in sorted(groups.items()):
        valid = [row for row in members if row["eligible"]]
        pairs = [row for row in members if row.get("paired")]
        stats = _stats(row["coefficient_rmse"] for row in valid)
        paired = _stats(row["paired_difference"] for row in pairs)
        counts = Counter(row["classification"] for row in members)
        setting = {k: members[0].get(k) for k in ("n_train", "p", "q", "rank", "source_rank", "sigma0")}
        parameter_counts = Counter(json.dumps(row["selected_parameters"], sort_keys=True) for row in valid
                                   if row.get("selected_parameters") is not None)
        boundary_counts = defaultdict(Counter)
        for row in valid:
            for parameter, boundary in row.get("boundary_indicators", {}).items():
                boundary_counts[parameter]["n_selected"] += 1
                boundary_counts[parameter]["at_lower"] += boundary.get("at_lower") is True
                boundary_counts[parameter]["at_upper"] += boundary.get("at_upper") is True
        unique_timed = {row["task_id"]: row for row in members if row.get("task_id") is not None and _finite(row.get("elapsed_seconds"))}
        results.append(dict(model_id=key[0], experiment_id=key[1], setting_index=key[2], method=key[3],
            **setting, planned=len(members), successful=len(valid), failed=counts["method_failure"],
            missing=counts["missing"], corrupt=counts["corrupt"], execution_failed=counts["execution_failure"],
            partial_reference=counts["partial_reference"], seed_ids=[r["seed_id"] for r in members],
            n_unique_fit_tasks=len({row.get("task_id") for row in members if row.get("task_id") is not None}),
            coefficient_rmse=stats, paired_difference_vs_v2=paired,
            paired_higher_rmse_count=sum(row["paired_difference"] > 1e-12 for row in pairs),
            **_paired_diagnostics(pairs),
            elapsed_seconds_per_unique_task=_stats(row["elapsed_seconds"] for row in unique_timed.values()),
            selected_parameter_frequencies=[dict(parameters=json.loads(parameters), count=count)
                                           for parameters, count in sorted(parameter_counts.items())],
            selected_boundary_counts={parameter: dict(counts) for parameter, counts in sorted(boundary_counts.items())},
            complete=all(row["classification"] == "eligible" for row in members),
            outcome_counts=dict(counts)))
    return results


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _plot(settings, output, complete):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(3, 4, figsize=(21, 13), squeeze=False)
    methods = [name for name in LABELS if any(row["method"] == name for row in settings)]
    methods += sorted({row["method"] for row in settings} - set(methods))
    colors = dict(zip(methods, plt.get_cmap("tab10").colors))
    handles = {}
    for model in range(3):
        for experiment in range(4):
            ax = axes[model][experiment]
            panel = [row for row in settings if row["model_id"] == model and row["experiment_id"] == experiment]
            ax.set_title(f"Model {('I','II','III')[model]} · Experiment {experiment + 1}")
            ax.set_xlabel(("Training sample size", "Imposed target rank", "SMART protected coordinates", "Source noise SD")[experiment])
            ax.set_ylabel("Coefficient RMSE")
            if not panel:
                ax.text(.5, .5, "Not included in this plan", ha="center", va="center", transform=ax.transAxes, color="0.45")
                continue
            keys = ("n_train", "rank", "source_rank", "sigma0")
            incomplete = any(not row["complete"] for row in panel)
            for method in methods:
                selected = sorted((row for row in panel if row["method"] == method), key=lambda row: row["setting_index"])
                if not selected:
                    continue
                x = [row[keys[experiment]] for row in selected]
                means = [row["coefficient_rmse"]["mean"] if row["coefficient_rmse"]["n"] else float("nan") for row in selected]
                errors = [1.959963984540054 * row["coefficient_rmse"]["mcse"] if row["coefficient_rmse"]["mcse"] is not None else 0. for row in selected]
                line = ax.errorbar(x, means, yerr=errors, color=colors[method], marker="o", markersize=3,
                                  linewidth=1.6 if method == "v2" else 1., capsize=2, label=LABELS.get(method, method))
                handles[method] = line
            ax.grid(alpha=.18)
            if incomplete:
                ax.text(.01, .98, "PARTIAL: missing/failed/provisional outputs\nMeans use available seeds; see coverage table",
                        ha="left", va="top", transform=ax.transAxes, fontsize=8, color="#9b2226",
                        bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    figure.suptitle("Original paper grids: shared-holdout comparator evidence" + ("" if complete else " — INCOMPLETE"), fontsize=17)
    if handles:
        figure.legend([handles[m] for m in methods if m in handles], [LABELS.get(m, m) for m in methods if m in handles],
                      loc="upper center", bbox_to_anchor=(.5, .957), ncol=3, frameon=False)
    figure.text(.5, .008, "Pointwise normal 95% Monte Carlo intervals; one replication per simulation seed. Reused Experiment-3 fits do not add independent evidence.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .025, 1, .88))
    for extension in ("pdf", "png"):
        figure.savefig(output / f"settings.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(root):
    """Write a complete ledger, including absent and unsuccessful planned tasks."""
    from . import common
    root = Path(root).resolve()
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    plan = common.load_plan(root, full_source=False)
    _require(plan.get("method") == "PaperSourceComparators", "unsupported comparator plan")
    _require(len(plan["tasks"]) == plan["n_tasks"] and len(plan["groups"]) == plan["n_groups"], "plan counts mismatch")
    tasks, errors = {}, []
    for task in plan["tasks"]:
        _require(task["task_id"] not in tasks, "duplicate planned task ID")
        try:
            row = check_task(root, plan, task, common)
        except FileNotFoundError as error:
            row = dict(task, classification="missing", eligible=False, error=str(error))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row = dict(task, classification="corrupt", eligible=False, error=str(error))
        tasks[task["task_id"]] = row
        if row["classification"] in BAD_EXECUTION:
            errors.append(dict(task_id=task["task_id"], classification=row["classification"], error=row.get("error")))
    references, reference_errors = _load_references(root, plan, common)
    rows = display_rows(plan, tasks, references, reference_errors)
    settings = aggregate_rows(rows)
    counts = Counter(row["classification"] for row in tasks.values())
    execution_complete = not any(counts[key] for key in BAD_EXECUTION)
    references_complete = all(row["classification"] == "eligible" for row in rows if row["method"] == "v2")
    complete = execution_complete and references_complete and not reference_errors
    report = dict(schema_version=1, method=plan["method"], root=str(root), checked_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=common.sha(root / "plan.json"), plan_fingerprint=plan["plan_fingerprint"],
        source_manifest_sha256=plan["source_manifest_sha256"], success=complete,
        all_planned_tasks_accounted=True, complete_execution_coverage=execution_complete,
        complete_reference_coverage=references_complete, all_methods_successful=counts["eligible"] == plan["n_tasks"],
        planned_tasks=plan["n_tasks"], planned_groups=plan["n_groups"], display_cases=len(plan["display_cases"]),
        unique_task_outcome_counts=dict(counts), errors=errors, reference_errors=reference_errors,
        interval=INTERVAL, paired_difference_definition="competitor coefficient RMSE minus v2 coefficient RMSE; positive favors v2",
        geometric_ratio_definition="competitor RMSE divided by v2 RMSE; zero pairs excluded with explicit counts",
        higher_rmse_definition="competitor RMSE exceeds v2 RMSE by more than 1e-12; this is comparator harm relative to v2",
        unique_task_timing={method: _stats(row["elapsed_seconds"] for row in tasks.values()
                if row["method"] == method and _finite(row.get("elapsed_seconds")))
                for method in sorted({row["method"] for row in tasks.values()})},
        truth_used_for_selection=False, refit_on_validation=False,
        audit_scope="Runtime selected-score and selected-state audit receipts plus candidate metadata, provenance and execution checks; reporting does not refit or independently reexecute all candidates.",
        reference_audit_scope="Frozen external-validation campaign summary and plan/preparation hash bindings; numerical-audit hash and selected-winner coverage are checked when bound by preparation. No BIC result is merged.",
        independent_unit="simulation seed within setting; display aliases and different settings are not additional independent replications",
        case_results=rows, setting_results=settings)
    common.write(output / "summary.json", report)
    _csv(output / "per-case.csv", rows)
    _csv(output / "per-setting.csv", settings)
    _csv(output / "task-outcomes.csv", list(tasks.values()))
    _plot(settings, output, complete)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    summary = summarize(arguments.root)
    print(json.dumps({key: summary[key] for key in ("success", "planned_tasks", "unique_task_outcome_counts", "complete_reference_coverage")}, sort_keys=True))
