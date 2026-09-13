#!/usr/bin/env python3
"""Summarize one frozen v2 campaign in place, without refitting or copying raw data.

Checks every planned task's provenance, artifact hashes and process/launcher exit
records. Selects successful candidates by validation MSE only. Initializer
exclusions and numerical stalls are scientific outcomes, not process crashes.
This is an integrity/coverage summary, not a numerical saved-state audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
SELECTION_RULE = "smallest selected validation MSE; exact ties use smallest frozen task ID"
STALLS = {"numerical_stagnation", "line_search_failed"}
SUCCESS_STATUSES = {"completed", "converged"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def relative(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "invalid relative artifact path")
    path = root / name
    require(".." not in Path(name).parts and path.resolve().is_relative_to(root.resolve()),
            f"unsafe artifact path: {name}")
    return path


def check_hash(path, expected):
    require(isinstance(expected, str) and sha(path) == expected, f"SHA256 mismatch: {path}")


def load_provenance(root):
    plan = read(root / "plan.json")
    require(isinstance(plan, dict), "invalid plan object")
    require(plan.get("root") == str(root), "plan root identity mismatch")
    require(digest({k: v for k, v in plan.items() if k != "plan_fingerprint"}) ==
            plan.get("plan_fingerprint"), "plan fingerprint mismatch")
    require(plan.get("schema_version") == 1 and plan.get("method") == "SparseSMARTv2Pilot",
            "unsupported plan schema or method")
    cases = {case["case_id"]: case for case in plan["cases"]}
    require(len(cases) == len(plan["cases"]) == plan["n_cases"] > 0, "invalid case counts")
    require(len(plan["tasks"]) == plan["n_tasks"] > 0, "invalid task count")
    require((root / "work-items.tsv").read_text().splitlines() ==
            [str(i) for i in range(plan["n_tasks"])], "work table differs from frozen task IDs")
    by_case = {cid: [] for cid in cases}
    for tid, task in enumerate(plan["tasks"]):
        require(type(task["task_id"]) is int and task["task_id"] == tid and task["case_id"] in cases,
                "invalid task identity")
        require(all(finite(task.get(k)) and task[k] >= 0 for k in
                    ("init_penalty", "penalty_u", "penalty_v")), "invalid task tuning parameters")
        by_case[task["case_id"]].append(task)
    require(all(by_case.values()), "planned case has no tasks")
    for members in by_case.values():
        keys = [(t["init_penalty"], t["penalty_u"], t["penalty_v"]) for t in members]
        require(len(keys) == len(set(keys)), "duplicate tuning candidate in plan")
    check_hash(root / "source-manifest.json", plan["source_manifest_sha256"])
    manifest = read(root / "source-manifest.json")
    require(manifest.get("schema_version") == 1 and isinstance(manifest.get("files"), dict)
            and bool(manifest["files"]), "invalid source manifest")
    source = next((p for p in (root / "source", root.parent / "source") if p.is_dir()), None)
    require(source is not None, "frozen source snapshot unavailable")
    for name, expected in manifest["files"].items():
        check_hash(relative(source, name), expected)
    prep = read(root / "preparation.json")
    require(prep.get("status") == "complete" and prep.get("success") is True,
            "preparation incomplete")
    require(prep.get("plan_sha256") == sha(root / "plan.json") and
            prep.get("source_manifest_sha256") == plan["source_manifest_sha256"] and
            prep.get("n_cases") == plan["n_cases"] and prep.get("n_tasks") == plan["n_tasks"],
            "preparation identity/count mismatch")
    prepared = {c["case_id"]: c for c in prep["cases"]}
    require(len(prepared) == len(prep["cases"]) and set(prepared) == set(cases),
            "preparation case coverage mismatch")
    if plan.get("reference_cases_sha256"):
        check_hash(root / "reference-cases.json", plan["reference_cases_sha256"])
    return plan, by_case, prepared


def case_metadata(root, plan, case, prepared):
    folder = relative(root / "cases", case["case_id"])
    check_hash(folder / "case.json", prepared["case_json_sha256"])
    meta = read(folder / "case.json")
    require(isinstance(meta, dict), "invalid case metadata object")
    require(meta.get("case") == case and meta.get("plan_fingerprint") == plan["plan_fingerprint"],
            "case metadata identity mismatch")
    require(meta.get("files") == prepared["files"] and meta.get("initializers") == prepared["initializers"],
            "case/preparation files or initializers differ")
    for name, expected in meta["files"].items():
        check_hash(relative(folder, name), expected)
    return meta


def exit_record(folder, prefix, slurm):
    value = (folder / f"{prefix}-exit-code.txt").read_text().strip()
    require(value.isdecimal(), f"invalid {prefix} exit code")
    with (folder / f"{prefix}-status.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(len(rows) == 1 and rows[0].get("exit_code") == value,
            f"invalid {prefix} exit status")
    row = rows[0]
    require(row.get("job_id") == str(slurm["job_id"]) and bool(row.get("finished_utc")),
            f"{prefix} job/time binding differs")
    if prefix == "process":
        require(row.get("step_id") == str(slurm["step_id"]), "process step binding differs")
    return int(value)


def expects_rrr(case, config, task):
    capacities = [(dimension - free) * case["rank"] for dimension, free in
                  zip((case["p"], case["q"]), case["free_directions"])]
    return (config.get("rrr_shortcut", False) and case["support_limits"] == capacities and
            all(penalty == 0 or capacity == 0 for penalty, capacity in
                zip((task["penalty_u"], task["penalty_v"]), capacities)))


def check_task(root, plan, task, meta):
    folder = root / "tasks" / f"{task['task_id']:05d}"
    result, status = read(folder / "result.json"), read(folder / "status.json")
    require(isinstance(result, dict) and isinstance(status, dict), "invalid result/status object")
    check_hash(folder / "result.json", status.get("result_sha256"))
    identity = dict(schema_version=plan["schema_version"], method=plan["method"], task=task,
                    case=meta["case"], plan_fingerprint=plan["plan_fingerprint"],
                    source_manifest_sha256=plan["source_manifest_sha256"])
    for key, expected in identity.items():
        require(result.get(key) == status.get(key) == expected, f"result/status identity differs: {key}")
    for key in ("success", "execution_success", "fit_status"):
        require(result.get(key) == status.get(key), f"result/status differs: {key}")
    require(type(result.get("success")) is bool and type(result.get("execution_success")) is bool,
            "invalid task success flags")
    require(status.get("status") == "finished", "task status not finished")
    require(isinstance(result.get("files"), dict), "missing artifact hash mapping")
    for name, expected in result["files"].items():
        check_hash(relative(folder, name), expected)
    codes = [exit_record(folder, prefix, result["slurm"]) for prefix in ("process", "launcher")]
    row = dict(task, classification="execution_failure", eligible=False,
               fit_status=result.get("fit_status"), fit_method=result.get("fit_method", "sparse_smart_v2"),
               termination_reason=result.get("termination_reason"),
               optimization_converged=result.get("optimization_converged") is True,
               elapsed_seconds=result.get("elapsed_seconds"), n_iter=result.get("n_iter"),
               selected_iteration=result.get("selected_iteration"), result_sha256=sha(folder / "result.json"),
               status_sha256=sha(folder / "status.json"), process_exit_code=codes[0], launcher_exit_code=codes[1])
    require(finite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0, "invalid task elapsed time")
    if any(codes) or not result["execution_success"] or result.get("status") != "complete":
        return row
    require(result.get("fingerprints") == meta["fingerprints"] and
            result.get("source_fingerprint") == meta["source_fingerprint"], "data/source fingerprints differ")
    initial = next(r for r in meta["initializers"] if r["init_penalty"] == task["init_penalty"])
    require(result.get("initialization_record") == initial, "initializer record differs")
    require(type(row["n_iter"]) is int and 0 <= row["n_iter"] <= plan["configuration"]["iterations"],
            "invalid iteration count")
    direct = expects_rrr(meta["case"], plan["configuration"], task)
    excluded = result.get("refinement_skipped") == "initializer_failed_strict_preflight"
    if excluded:
        require(not direct and not initial["eligible"] and not result["success"] and row["n_iter"] == 0
                and row["selected_iteration"] is None and result.get("avg_err") is None and not result["files"],
                "invalid initializer exclusion")
        require(result["fit_status"] == initial["status"] and result.get("metadata") == initial["metadata"],
                "excluded initializer status differs")
        row["classification"] = "initializer_exclusion"
        return row
    require(initial["eligible"] or direct, "refinement used an excluded initializer")
    if direct:
        require(result.get("fit_method") == "target_rrr", "planned RRR endpoint was not used")
    elif result.get("fit_method") == "target_rrr":
        raise ValueError("unexpected RRR endpoint under frozen plan")
    if result["success"]:
        require(row["fit_status"] in SUCCESS_STATUSES, "successful fit reports a failure status")
        require({"states.npz", "history.json.gz"}.issubset(result["files"]), "missing bound fit artifacts")
        selected = row["selected_iteration"]
        require(type(selected) is int and 0 <= selected <= row["n_iter"], "invalid selected iteration")
        for endpoint in ("selected", "terminal"):
            metrics = result.get(f"{endpoint}_metrics", {})
            require(isinstance(metrics, dict), f"missing {endpoint} metrics")
            require(all(finite(metrics.get(key)) and metrics[key] >= 0 for key in
                        ("validation_mse", "coefficient_rmse", "coefficient_frobenius_squared",
                         "training_prediction_mse")), f"invalid {endpoint} metrics")
        require(result.get("avg_err") == result["selected_metrics"]["coefficient_rmse"],
                "reported coefficient error differs")
        if direct:
            require(row["n_iter"] == selected == 0 and
                    isinstance(result.get("rrr_certificate"), dict) and
                    result["rrr_certificate"].get("certified") is True and
                    row["termination_reason"] == "target_rrr_closed_form" and row["optimization_converged"],
                    "invalid RRR certificate or iteration")
        row.update(classification="eligible", eligible=True,
                   selected_validation_mse=result["selected_metrics"]["validation_mse"],
                   coefficient_rmse=result["selected_metrics"]["coefficient_rmse"],
                   coefficient_frobenius_squared=result["selected_metrics"]["coefficient_frobenius_squared"],
                   training_prediction_mse=result["selected_metrics"]["training_prediction_mse"],
                   terminal_validation_mse=result["terminal_metrics"]["validation_mse"])
    else:
        require(result.get("avg_err") is None, "failed fit has a paper error")
        row["classification"] = "numerical_stagnation" if row["fit_status"] in STALLS else "other_scientific_failure"
    return row


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


def summarize(result_root, output_root):
    root, output = Path(result_root).resolve(), Path(output_root).resolve()
    require(output != root and not root.is_relative_to(output), "output must not be the results root or its ancestor")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, result_root=str(root), output_root=str(output),
                  checked_utc=datetime.now(timezone.utc).isoformat(), success=False,
                  selection_rule=SELECTION_RULE, truth_used_for_selection=False,
                  numerical_state_audit_performed=False, errors=[], case_results=[], setting_results=[])
    report.update(complete_execution_coverage=False, complete_tuning_coverage=False)
    counts = Counter()
    task_path = output / "task-outcomes.jsonl"
    temporary = task_path.with_suffix(".jsonl.tmp")
    try:
        plan, by_case, prepared = load_provenance(root)
        report.update(plan_fingerprint=plan["plan_fingerprint"], plan_sha256=sha(root / "plan.json"),
                      source_manifest_sha256=plan["source_manifest_sha256"],
                      seed_file_sha256=plan.get("seed_file_sha256"), configuration=plan["configuration"],
                      case_selection=plan.get("case_selection"), planned_cases=plan["n_cases"],
                      planned_tasks=plan["n_tasks"])
        with temporary.open("w") as stream:
            for case in plan["cases"]:
                cid, members = case["case_id"], by_case[case["case_id"]]
                case_counts, winner, elapsed = Counter(), None, 0.
                try:
                    meta = case_metadata(root, plan, case, prepared[cid])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    meta = None
                    report["errors"].append(dict(case_id=cid, error=f"case provenance: {error}"))
                for task in members:
                    try:
                        require(meta is not None, "case provenance invalid")
                        row = check_task(root, plan, task, meta)
                    except FileNotFoundError as error:
                        row = dict(task, classification="missing", eligible=False, error=str(error))
                    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                        row = dict(task, classification="corrupt", eligible=False, error=str(error))
                    classification = row["classification"]
                    counts[classification] += 1
                    case_counts[classification] += 1
                    elapsed += row.get("elapsed_seconds", 0.)
                    if classification in ("missing", "corrupt", "execution_failure"):
                        report["errors"].append(dict(task_id=task["task_id"], case_id=cid,
                                                     classification=classification, error=row.get("error")))
                    if row["eligible"] and (winner is None or
                            (row["selected_validation_mse"], row["task_id"]) <
                            (winner["selected_validation_mse"], winner["task_id"])):
                        winner = row
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                record = dict(case=case, planned_tasks=len(members), outcome_counts=dict(case_counts), winner=winner,
                              case_total_task_elapsed_seconds=elapsed,
                              complete_execution_coverage=not any(case_counts[k] for k in ("missing", "corrupt", "execution_failure")),
                              complete_tuning_coverage=case_counts["eligible"] == len(members))
                if winner is None:
                    report["errors"].append(dict(case_id=cid, error="no eligible validation-selected winner"))
                report["case_results"].append(record)
        temporary.replace(task_path)
        groups = defaultdict(list)
        for record in report["case_results"]:
            case = record["case"]
            groups[(case["model_id"], case["experiment_id"], case["setting_index"])].append(record)
        for key, records in sorted(groups.items()):
            winners = [r["winner"] for r in records if r["winner"] is not None]
            values = [w["coefficient_rmse"] for w in winners]
            setting = {k: v for k, v in records[0]["case"].items() if k not in ("case_id", "seed_id", "random_seed")}
            for record in records[1:]:
                require({k: v for k, v in record["case"].items() if k not in ("case_id", "seed_id", "random_seed")} == setting,
                        f"inconsistent metadata within setting {key}")
            report["setting_results"].append(dict(setting=setting, requested_seeds=len(records), available_seeds=len(values),
                seed_ids=[r["case"]["seed_id"] for r in records],
                available_seed_ids=[r["case"]["seed_id"] for r in records if r["winner"] is not None],
                coefficient_rmse_mean=statistics.mean(values) if values else None,
                coefficient_rmse_se=statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
                selected_validation_mse_mean=statistics.mean(w["selected_validation_mse"] for w in winners) if winners else None,
                complete_execution_coverage=all(r["complete_execution_coverage"] for r in records),
                complete_tuning_coverage=all(r["complete_tuning_coverage"] for r in records),
                coverage_status="complete" if len(values) == len(records) and all(r["complete_execution_coverage"] for r in records)
                                else "partial"))
        report.update(outcome_counts=dict(counts), cases_with_winner=sum(r["winner"] is not None for r in report["case_results"]),
                      complete_execution_coverage=not any(counts[k] for k in ("missing", "corrupt", "execution_failure")),
                      complete_tuning_coverage=counts["eligible"] == plan["n_tasks"],
                      success=not report["errors"], task_outcomes_sha256=sha(task_path))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(dict(error=f"campaign provenance/summary: {error}"))
    finally:
        temporary.unlink(missing_ok=True)
    write_csv(output / "case-results.csv", [dict(r["case"], **{k: v for k, v in r.items() if k != "case"})
                                           for r in report["case_results"]])
    write_csv(output / "setting-results.csv", [dict(r["setting"], **{k: v for k, v in r.items() if k != "setting"})
                                               for r in report["setting_results"]])
    report["output_hashes"] = {name: sha(output / name) for name in ("case-results.csv", "setting-results.csv")}
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.result_root, args.output_root)
    print(json.dumps({k: report.get(k) for k in ("success", "planned_tasks", "outcome_counts", "cases_with_winner",
                                               "complete_execution_coverage", "complete_tuning_coverage")}, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
