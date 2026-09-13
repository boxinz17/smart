#!/usr/bin/env python3
"""Summarize BIC re-selection of saved validation-stopped SparseSMART v2 fits.

This reads compact, per-case scoring records. It neither fits models nor copies
raw states. Both BIC arms remain validation-dependent because their retained
trajectories were stopped using validation. They are not training-only BIC.
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
import re
import statistics

METHOD = "SparseSMARTv2RetrospectiveBIC"
ARMS = ("original_validation", "bic_validation_selected", "bic_validation_terminal")
LABELS = {
    "original_validation": "Original validation selection",
    "bic_validation_selected": "BIC: validation-selected states",
    "bic_validation_terminal": "BIC: validation-stopped terminal states",
}
METRICS = ("coefficient_rmse", "training_prediction_mse", "validation_mse")
PAIRS = ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1]))
LIMITATION = (
    "Both retrospective BIC arms use trajectories stopped using validation data. "
    "The selected-state arm also uses validation to choose the endpoint within each "
    "trajectory. These are validation-stopped/BIC-selected hybrids, not training-only "
    "BIC experiments. Only the previously saved selected and terminal states are "
    "compared; no fits, initialization, data generation, unsaved-checkpoint search, "
    "or automatic rank/free-count grid expansion is performed."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def check_hash(path, expected):
    require(isinstance(expected, str) and sha(path) == expected, f"SHA256 mismatch: {path}")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def file_inside(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
            f"unsafe relative path: {name}")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), f"path escapes root: {name}")
    return path


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def load_plan(root):
    plan = read(root / "plan.json")
    require(plan.get("schema_version") == 1 and plan.get("method") == METHOD, "unsupported retrospective plan")
    require(plan.get("root") == str(root), "plan root identity mismatch")
    require(digest({key: value for key, value in plan.items() if key != "plan_fingerprint"}) ==
            plan.get("plan_fingerprint"), "plan fingerprint mismatch")
    cases = plan["cases"]
    require(isinstance(cases, list) and len(cases) == plan["n_cases"] > 0, "invalid planned case count")
    ids = [entry["case_id"] for entry in cases]
    require(len(set(ids)) == len(ids), "duplicate planned case IDs")
    identities = set()
    for entry in cases:
        cid, case = entry["case_id"], entry["case"]
        require(re.fullmatch(r"m[0-9]+_e[0-9]+_k[0-9]+_s[0-9]+", cid) is not None and case["case_id"] == cid,
                "invalid case identity")
        for key in ("model_id", "experiment_id", "setting_index", "seed_id"):
            require(type(case[key]) is int and case[key] >= 0, f"invalid case coordinate: {key}")
        require(case["model_id"] < 3 and case["experiment_id"] < 4, "unsupported model/experiment")
        coordinate = tuple(case[key] for key in ("model_id", "experiment_id", "setting_index", "seed_id"))
        require(coordinate not in identities, "duplicate model/experiment/setting/seed")
        identities.add(coordinate)
        require(Path(entry["source_root"]).is_absolute(), "reference subroot must be absolute")
    check_hash(root / "source-manifest.json", plan["source_manifest_sha256"])
    manifest = read(root / "source-manifest.json")
    require(bool(manifest.get("files")), "empty source manifest")
    for name, expected in manifest["files"].items():
        check_hash(file_inside(root / "source", name), expected)
    preparation = read(root / "preparation.json")
    require(preparation.get("status") == "complete" and preparation.get("success") is True,
            "retrospective preparation incomplete")
    for key, expected in (("schema_version", 1), ("method", METHOD), ("plan_sha256", sha(root / "plan.json")),
                          ("plan_fingerprint", plan["plan_fingerprint"]), ("n_cases", plan["n_cases"]),
                          ("source_manifest_sha256", plan["source_manifest_sha256"])):
        require(preparation.get(key) == expected, f"preparation identity mismatch: {key}")
    prepared = {entry["case_id"]: entry for entry in preparation["cases"]}
    require(len(prepared) == len(preparation["cases"]) and set(prepared) == set(ids), "prepared case coverage mismatch")
    return plan, prepared


def validate_winner(winner, arm):
    if winner is None:
        return
    require(isinstance(winner, dict), "invalid winner record")
    require(type(winner.get("task_id")) is int and winner["task_id"] >= 0, "invalid selected task ID")
    require(type(winner.get("iteration")) is int and winner["iteration"] >= 0, "invalid selected iteration")
    require(winner.get("endpoint") == ("terminal" if arm == ARMS[2] else "selected"), "wrong endpoint for comparison arm")
    require(winner.get("fit_method") in ("sparse_smart_v2", "target_rrr"), "unknown fit method")
    require(isinstance(winner.get("task"), dict), "missing selected tuning configuration")
    for key in ("score", "rss", "model_dimension"):
        require(finite(winner["bic"].get(key)), f"invalid selected BIC {key}")
    require(winner["bic"]["rss"] >= 0 and winner["bic"]["model_dimension"] >= 0, "negative RSS/model dimension")
    for key in (*METRICS, "coefficient_frobenius_squared"):
        require(finite(winner["metrics"].get(key)) and winner["metrics"][key] >= 0, f"invalid evaluation metric: {key}")
    for key in ("result_sha256", "states_sha256"):
        require(isinstance(winner.get(key), str) and re.fullmatch("[0-9a-f]{64}", winner[key]), f"invalid winner provenance: {key}")


def load_case(root, plan, entry, prepared):
    cid = entry["case_id"]
    result_path, status_path = root / "cases" / f"{cid}.json", root / "cases" / f"{cid}.status.json"
    input_path = root / "inputs" / f"{cid}.json"
    check_hash(input_path, prepared["input_sha256"])
    result, status = read(result_path), read(status_path)
    check_hash(result_path, status.get("result_sha256"))
    for key, expected in (("schema_version", 1), ("method", METHOD), ("case_id", cid),
                          ("plan_fingerprint", plan["plan_fingerprint"]),
                          ("source_manifest_sha256", plan["source_manifest_sha256"]),
                          ("input_sha256", prepared["input_sha256"])):
        require(result.get(key) == expected and status.get(key) == expected, f"case result/status identity mismatch: {key}")
    require(result.get("case") == entry["case"], "case parameters differ from plan")
    require(result.get("reference_subroot") == entry["source_root"], "reference subroot identity mismatch")
    require(status.get("status") == "finished", "case scoring has not finished")
    require(type(result.get("success")) is bool and status.get("success") == result["success"], "case success/status mismatch")
    for key in ("complete_execution_coverage", "complete_tuning_coverage"):
        require(type(result.get(key)) is bool, f"missing case coverage flag: {key}")
    require(isinstance(result.get("errors"), list), "missing case error audit")
    require(isinstance(result.get("counts"), dict) and all(type(v) is int and v >= 0 for v in result["counts"].values()),
            "invalid task classification counts")
    require(type(result.get("planned_tasks")) is int and result["planned_tasks"] > 0, "invalid planned task count")
    require(type(result.get("eligible_tasks")) is int and 0 <= result["eligible_tasks"] <= result["planned_tasks"], "invalid eligible task count")
    require(sum(result["counts"].values()) == result["planned_tasks"], "task outcome count differs from plan")
    require(result["counts"].get("eligible", 0) == result["eligible_tasks"], "eligible task count differs")
    require(isinstance(result.get("candidate_inputs_fingerprint"), str) and
            re.fullmatch("[0-9a-f]{64}", result["candidate_inputs_fingerprint"]) and
            result["candidate_inputs_fingerprint"] == status.get("candidate_inputs_fingerprint"),
            "candidate input fingerprint/status mismatch")
    require(set(result["arms"]) == set(ARMS), "comparison arm coverage differs")
    for arm in ARMS:
        validate_winner(result["arms"][arm], arm)
    if result["success"]:
        require(result["complete_execution_coverage"] and not result["errors"] and
                all(result["arms"].values()), "successful case lacks complete execution or winners")
    return result, {"case_id": cid, "result_sha256": sha(result_path), "status_sha256": sha(status_path),
                    "input_sha256": prepared["input_sha256"], "candidate_inputs_fingerprint": result["candidate_inputs_fingerprint"]}


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "se": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None}


def xvalue(case):
    return case[("n_train", "rank", "source_rank", "sigma0")[case["experiment_id"]]]


def changed(winner, reference):
    if winner is None or reference is None:
        return None, None
    candidate = winner["task_id"] != reference["task_id"]
    # RRR selected and terminal endpoints are the same state at iteration zero.
    state = candidate or winner["iteration"] != reference["iteration"]
    return candidate, state


def build_tables(records):
    rows, curves, differences = [], [], []
    grouped = defaultdict(list)
    for record in records:
        case = record["case"]
        grouped[tuple(case[k] for k in ("model_id", "experiment_id", "setting_index"))].append(record)
        for arm in ARMS:
            winner = record["arms"][arm]
            candidate_changed, state_changed = changed(winner, record["arms"][ARMS[0]])
            row = dict(case_id=case["case_id"], model_id=case["model_id"], experiment_id=case["experiment_id"],
                       setting_index=case["setting_index"], seed_id=case["seed_id"], x=xvalue(case), arm=arm,
                       available=winner is not None, scoring_status=record["classification"],
                       complete_execution_coverage=record["complete_execution_coverage"],
                       complete_tuning_coverage=record["complete_tuning_coverage"],
                       candidate_changed_from_validation=candidate_changed, state_changed_from_validation=state_changed)
            if winner is not None:
                row.update(winner["metrics"])
                row.update(task_id=winner["task_id"], endpoint=winner["endpoint"], iteration=winner["iteration"],
                           fit_method=winner["fit_method"], bic=winner["bic"]["score"], rss=winner["bic"]["rss"],
                           model_dimension=winner["bic"]["model_dimension"], rank=case["rank"],
                           free_u=case["free_directions"][0], free_v=case["free_directions"][1],
                           termination_reason=winner.get("termination_reason"), optimization_converged=winner.get("optimization_converged"))
                row.update({k: winner["task"].get(k) for k in ("init_penalty", "penalty_u", "penalty_v")})
            rows.append(row)
    for key, members in sorted(grouped.items()):
        common = dict(model_id=key[0], experiment_id=key[1], setting_index=key[2], x=xvalue(members[0]["case"]),
                      requested_seeds=len(members), requested_seed_ids=sorted(r["case"]["seed_id"] for r in members),
                      complete_execution_coverage=all(r["complete_execution_coverage"] for r in members),
                      complete_tuning_coverage=all(r["complete_tuning_coverage"] for r in members))
        for arm in ARMS:
            available = [r for r in members if r["arms"][arm] is not None]
            winners = [r["arms"][arm] for r in available]
            curve = dict(common, arm=arm, available_seeds=len(available),
                         missing_seed_ids=sorted(r["case"]["seed_id"] for r in members if r["arms"][arm] is None),
                         rrr_selected=sum(w["fit_method"] == "target_rrr" for w in winners),
                         rrr_fraction=sum(w["fit_method"] == "target_rrr" for w in winners) / len(winners) if winners else None)
            for metric in METRICS:
                value = stats([w["metrics"][metric] for w in winners])
                curve.update({metric + "_mean": value["mean"], metric + "_se": value["se"]})
            for name, values in (("model_dimension", [w["bic"]["model_dimension"] for w in winners]),
                                 ("iteration", [w["iteration"] for w in winners])):
                value = stats(values)
                curve.update({name + "_mean": value["mean"], name + "_se": value["se"]})
            comparisons = [changed(r["arms"][arm], r["arms"][ARMS[0]]) for r in available if r["arms"][ARMS[0]] is not None]
            curve.update(paired_with_validation=len(comparisons), candidate_changed_from_validation=sum(c[0] for c in comparisons),
                         state_changed_from_validation=sum(c[1] for c in comparisons))
            curves.append(curve)
        for arm, reference in PAIRS:
            paired = [r for r in members if r["arms"][arm] is not None and r["arms"][reference] is not None]
            ids = sorted(r["case"]["seed_id"] for r in paired)
            for metric in METRICS:
                value = stats([r["arms"][arm]["metrics"][metric] - r["arms"][reference]["metrics"][metric] for r in paired])
                differences.append(dict(common, arm=arm, reference=reference, metric=metric, paired_seeds=value["n"],
                                        paired_seed_ids=ids, difference_mean=value["mean"], difference_se=value["se"],
                                        interpretation="negative means lower error for arm"))
    return rows, curves, differences


def plot_curves(output, curves):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import t
    colors = {ARMS[0]: "#64748B", ARMS[1]: "#007F87", ARMS[2]: "#B4512E"}
    xlabels = ("Training observations", "Fitted rank", "Free/source directions", "Source-noise standard deviation")
    outputs = []
    book = output / "comparison.pdf"
    temporary_book = output / f"comparison.{os.getpid()}.tmp.pdf"
    with PdfPages(temporary_book) as pdf:
        for metric, ylabel in (("coefficient_rmse", "Coefficient RMSE"), ("training_prediction_mse", "Prediction MSE against truth")):
            for model in sorted({r["model_id"] for r in curves}):
                fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
                fig.subplots_adjust(top=.82, bottom=.16, hspace=.38, wspace=.28)
                fig.suptitle(f"SparseSMART v2 retrospective selection — Model {('I', 'II', 'III')[model]}", fontsize=17, y=.97)
                handles = []
                for experiment, ax in enumerate(axes.flat):
                    subset = [r for r in curves if r["model_id"] == model and r["experiment_id"] == experiment]
                    ticks = sorted({r["x"] for r in subset})
                    available_counts = [r["available_seeds"] for r in subset]
                    for arm in ARMS:
                        selected = sorted((r for r in subset if r["arm"] == arm and r[metric + "_mean"] is not None), key=lambda r: r["x"])
                        xs = [ticks.index(r["x"]) if experiment == 3 else r["x"] for r in selected]
                        ys = [r[metric + "_mean"] for r in selected]
                        intervals = [t.ppf(.975, r["available_seeds"] - 1) * r[metric + "_se"] if r[metric + "_se"] is not None else 0 for r in selected]
                        line = ax.errorbar(xs, ys, yerr=intervals, color=colors[arm], marker="o", ms=4,
                                           lw=1.7, capsize=2, linestyle="--" if arm == ARMS[0] else "-", label=LABELS[arm])
                        if experiment == 0:
                            handles.append(line)
                    nlabel = "no data" if not available_counts else f"n={min(available_counts)}–{max(available_counts)}"
                    ax.set_title(f"Experiment {experiment + 1} ({nlabel} seeds)", loc="left", fontsize=12)
                    ax.set_xlabel(xlabels[experiment]); ax.set_ylabel(ylabel)
                    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.22)
                    if experiment == 3:
                        ax.set_xticks(range(len(ticks)), [f"{x:g}" for x in ticks])
                    else:
                        ax.set_xticks(ticks)
                    positive = [r[metric + "_mean"] for r in subset if r[metric + "_mean"] is not None]
                    if experiment == 1 and positive and min(positive) > 0:
                        ax.set_yscale("log")
                fig.legend(handles=handles, labels=[LABELS[arm] for arm in ARMS], loc="upper center", bbox_to_anchor=(.5,.925), frameon=False, ncol=1, fontsize=10)
                fig.text(.06, .055, "Means and 95% t intervals over available seeds; lower is better. Missing cases are explicit in coverage tables.\n"
                         "Both BIC arms reuse validation-stopped trajectories; they are not training-only BIC. No models were refitted.\n"
                         "Prediction MSE here is X(C − C*) error, not observed training residual RSS. Source-noise ticks are equally spaced.", fontsize=9)
                for suffix in ("png", "pdf"):
                    name = f"model_{model + 1}_{metric}.{suffix}"
                    temporary = output / f"{name}.{os.getpid()}.tmp"
                    fig.savefig(temporary, format=suffix, dpi=170, bbox_inches="tight")
                    temporary.replace(output / name); outputs.append(name)
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    temporary_book.replace(book); outputs.append(book.name)
    return outputs


def write_report(output, report):
    lines = ["# Retrospective SparseSMART v2 BIC comparison", "", LIMITATION, "",
             f"Coverage: {report['completed_cases']}/{report.get('planned_cases', 0)} cases completed without execution/audit errors; "
             f"{len(report['errors'])} recorded issues.",
             f"Models: {', '.join(str(v + 1) for v in report.get('models', []))}; experiments: "
             f"{', '.join(str(v + 1) for v in report.get('experiments', []))}; "
             f"{report.get('planned_settings', 0)} settings; seed IDs: {report.get('seed_ids', [])}.", "",
             "| Selection | Cases with result | RRR selected | Candidate changed from validation | State changed from validation |", 
             "|---|---:|---:|---:|---:|" ]
    for arm in ARMS:
        entry = report["selection_counts"][arm]
        lines.append(f"| {LABELS[arm]} | {entry['available_cases']} | {entry['rrr_selected']} | "
                     f"{entry['candidate_changed_from_validation']} | {entry['state_changed_from_validation']} |")
    lines += ["", "Tables retain every planned seed, including explicit unavailable records. Paired differences use only the same "
              "model/experiment/setting/seed with results in both compared arms; negative differences favor the first arm. "
              "Initializer exclusions are retained as scientific coverage information and do not alone cause execution failure.", "",
              "The BIC penalty uses the existing model-dimension approximation. It is not a proved adaptive effective degrees of freedom. "
              "Optimization convergence is distinct from budget completion and validation stopping.", "",
              "Files: `per-seed-results.csv`, `performance-curves.csv`, `paired-differences.csv`, `coverage.csv`, "
              "`summary.json`, and per-model coefficient/prediction figures. Input and output hashes are recorded in `summary.json`."]
    if report["errors"]:
        lines += ["", "## Issues", ""] + ["- " + json.dumps(error, sort_keys=True) for error in report["errors"][:30]]
        if len(report["errors"]) > 30:
            lines.append(f"- {len(report['errors']) - 30} additional issues are listed in summary.json.")
    path = output / "REPORT.md"
    temporary = output / f"REPORT.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n"); temporary.replace(path)
    return path.name


def summarize(root, output_root=None, *, make_plots=True):
    root = Path(root).resolve()
    output = Path(output_root).resolve() if output_root else root / "summary"
    require(output != root and output.is_relative_to(root), "summary output must be a subdirectory of this retrospective campaign")
    require(output.relative_to(root).parts[0] not in ("source", "cases", "inputs"), "summary cannot overwrite campaign input/result directories")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, method=METHOD, result_root=str(root), output_root=str(output),
                  checked_utc=datetime.now(timezone.utc).isoformat(), success=False, errors=[], case_results=[],
                  input_hashes=[], completed_cases=0, limitation=LIMITATION, no_refitting=True,
                  validation_used_for_original_trajectories=True, bic_uses_training_rss_only=True,
                  truth_used_for_selection=False, complete_execution_coverage=False, complete_tuning_coverage=False)
    total_counts, case_counts = Counter(), Counter()
    try:
        plan, prepared = load_plan(root)
        report.update(plan_fingerprint=plan["plan_fingerprint"], plan_sha256=sha(root / "plan.json"),
                      preparation_sha256=sha(root / "preparation.json"), source_manifest_sha256=plan["source_manifest_sha256"],
                      source_campaign_root=plan.get("source_campaign_root"), planned_cases=plan["n_cases"],
                      planned_settings=len({(e["case"]["model_id"], e["case"]["experiment_id"], e["case"]["setting_index"]) for e in plan["cases"]}),
                      models=sorted({e["case"]["model_id"] for e in plan["cases"]}),
                      experiments=sorted({e["case"]["experiment_id"] for e in plan["cases"]}),
                      seed_ids=sorted({e["case"]["seed_id"] for e in plan["cases"]}))
        for entry in plan["cases"]:
            record = dict(case=entry["case"], case_id=entry["case_id"], classification="missing",
                          complete_execution_coverage=False, complete_tuning_coverage=False,
                          counts={}, planned_tasks=None, eligible_tasks=0, arms={arm: None for arm in ARMS})
            try:
                result, hashes = load_case(root, plan, entry, prepared[entry["case_id"]])
                record.update({key: result[key] for key in ("arms", "counts", "planned_tasks", "eligible_tasks",
                                                          "complete_execution_coverage", "complete_tuning_coverage")})
                record["classification"] = "complete" if result["success"] else "incomplete"
                report["input_hashes"].append(hashes)
                total_counts.update(result["counts"])
                if not result["success"]:
                    report["errors"].append(dict(case_id=entry["case_id"], classification="incomplete", errors=result["errors"]))
            except FileNotFoundError as error:
                report["errors"].append(dict(case_id=entry["case_id"], classification="missing", error=str(error)))
            except (OSError, ValueError, KeyError, TypeError) as error:
                record["classification"] = "corrupt"
                report["errors"].append(dict(case_id=entry["case_id"], classification="corrupt", error=str(error)))
            case_counts[record["classification"]] += 1
            report["case_results"].append(record)
        report["completed_cases"] = case_counts["complete"]
        report["complete_execution_coverage"] = all(r["complete_execution_coverage"] for r in report["case_results"])
        report["complete_tuning_coverage"] = all(r["complete_tuning_coverage"] for r in report["case_results"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(dict(classification="campaign_provenance", error=str(error)))
    report.update(task_counts=dict(total_counts), case_counts=dict(case_counts))
    rows, curves, paired = build_tables(report["case_results"])
    output_files = []
    coverage = [{key: r[key] for key in ("case_id", "classification", "planned_tasks", "eligible_tasks", "counts",
                                        "complete_execution_coverage", "complete_tuning_coverage")} for r in report["case_results"]]
    for name, table in (("per-seed-results.csv", rows), ("performance-curves.csv", curves),
                        ("paired-differences.csv", paired), ("coverage.csv", coverage)):
        write_csv(output / name, table); output_files.append(name)
    report.update(setting_results=curves, paired_differences=paired, selection_counts={})
    for arm in ARMS:
        available = [r for r in report["case_results"] if r["arms"][arm] is not None]
        comparisons = [changed(r["arms"][arm], r["arms"][ARMS[0]]) for r in available if r["arms"][ARMS[0]] is not None]
        report["selection_counts"][arm] = dict(available_cases=len(available), rrr_selected=sum(r["arms"][arm]["fit_method"] == "target_rrr" for r in available),
             paired_with_validation=len(comparisons), candidate_changed_from_validation=sum(c[0] for c in comparisons),
             state_changed_from_validation=sum(c[1] for c in comparisons),
             model_dimension=stats([r["arms"][arm]["bic"]["model_dimension"] for r in available]))
    if make_plots and curves:
        try:
            output_files.extend(plot_curves(output, curves))
        except (ImportError, OSError, ValueError) as error:
            report["errors"].append(dict(classification="plotting", error=str(error)))
    report["success"] = not report["errors"] and report["complete_execution_coverage"]
    output_files.append(write_report(output, report))
    report["output_hashes"] = {name: sha(output / name) for name in output_files}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--result-root", dest="root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = summarize(args.root, args.output_root, make_plots=not args.no_plots)
    print(json.dumps({key: report.get(key) for key in ("success", "planned_cases", "completed_cases", "case_counts", "task_counts", "selection_counts")}, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
