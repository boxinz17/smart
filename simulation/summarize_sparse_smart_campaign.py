"""Reduce published case audits without reading raw fits or model factors.

This module intentionally uses only the standard library. Scientific validation
belongs to the case aggregator; this reducer verifies its published receipts and
compact audit semantics, then computes statistics from individual saved seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from uuid import uuid4


METHOD = "SparseSMARTCampaignSummary"
INDEX_METHOD = "SparseSMARTCaseAggregation"
HEX = re.compile(r"[0-9a-f]{64}\Z")
CASE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _fingerprint(value, label):
    _require(isinstance(value, str) and HEX.fullmatch(value), f"Invalid {label}")
    return value


def _object_pairs(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _finite_tree(value):
    if isinstance(value, float):
        _require(math.isfinite(value), "Nonfinite number in compact record")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _read_json(path, expected_sha=None):
    _require(not path.is_symlink(), f"Symlink is not a published audit file: {path}")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None:
        _require(sha == expected_sha, f"SHA256 mismatch: {path.name}")
    value = json.loads(raw, object_pairs_hook=_object_pairs)
    _require(isinstance(value, dict), f"Expected JSON object: {path.name}")
    _finite_tree(value)
    return value, sha


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid {label}")
    return value


def _boolean(value, label):
    _require(type(value) is bool, f"Invalid {label}")
    return value


def _number(value, label):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             f"Invalid {label}")
    return value


def _configuration(value):
    _require(isinstance(value, dict), "Invalid runner configuration")
    result = dict(value)
    # Older continuous studies predate this optional schedule. Its absence
    # denotes exactly the empty list, not a different tuning protocol.
    result.setdefault("validation_iterations", [])
    budgets = result.get("iteration_budgets")
    _require(isinstance(budgets, list) and budgets, "Missing iteration budgets")
    for item in budgets:
        _integer(item, "iteration budget", 1)
    _require(budgets == sorted(set(budgets)), "Unordered or duplicate iteration budgets")
    for name in ("init_penalties", "penalties_u", "penalties_v"):
        grid = result.get(name)
        _require(isinstance(grid, list) and grid, f"Missing {name}")
        for item in grid:
            _number(item, name)
        _require(len(grid) == len(set(grid)), f"Duplicate {name}")
    _require(isinstance(result["validation_iterations"], list), "Invalid validation iterations")
    for item in result["validation_iterations"]:
        _integer(item, "validation iteration", 1)
    return result


def _validate_index(index):
    _require(type(index.get("schema_version")) is int and index["schema_version"] in (1, 2, 3)
             and index.get("method") == INDEX_METHOD, "Unsupported campaign index")
    fingerprint = _fingerprint(index.get("index_fingerprint"), "index fingerprint")
    _require(fingerprint == _digest({key: value for key, value in index.items()
                                    if key != "index_fingerprint"}), "Campaign index fingerprint mismatch")
    cases = index.get("cases")
    _require(isinstance(cases, list) and cases, "Campaign index has no cases")
    models = None
    if index["schema_version"] >= 2:
        _require(index.get("publication_mode") in ("full", "compact"), "Invalid publication mode")
        scope = index.get("scope")
        _require(isinstance(scope, dict) and set(scope) == {"models"}, "Invalid campaign model scope")
        models = scope["models"]
        _require(isinstance(models, list) and models and
                 all(type(model) is int and 0 <= model < 3 for model in models)
                 and models == sorted(set(models)), "Invalid campaign model scope")
    else:
        _require("publication_mode" not in index and "scope" not in index,
                 "Version 1 index cannot declare version 2 publication settings")
    keys, identities, suffix_identities, random_identities = set(), set(), set(), set()
    for entry in cases:
        _require(isinstance(entry, dict), "Invalid index case")
        key = entry.get("case_key")
        _require(isinstance(key, str) and CASE_KEY.fullmatch(key), "Invalid case key")
        _require(key not in keys, f"Duplicate case key: {key}")
        keys.add(key)
        for name in ("plan_fingerprint", "plan_sha256", "work_table_sha256", "seed_file_sha256"):
            _fingerprint(entry.get(name), name)
        _require(isinstance(entry.get("run_root"), str) and Path(entry["run_root"]).is_absolute(),
                 "Run root must be absolute")
        source = entry.get("source")
        _require(isinstance(source, dict) and isinstance(source.get("scheme"), str)
                 and isinstance(source.get("implementation"), dict), "Missing planned source")
        for name in ("true", "false"):
            _require(isinstance(source["implementation"].get(name), dict), "Missing source applicability class")
            _fingerprint(source["implementation"][name].get("fingerprint"), "planned implementation fingerprint")
        _configuration(entry.get("configuration"))
        cell = entry.get("cell")
        _require(isinstance(cell, dict), "Missing planned cell")
        for name in ("model", "experiment", "setting", "seed", "random_seed"):
            _integer(cell.get(name), f"cell {name}")
        _require(cell["model"] < 3 and cell["experiment"] < 4, "Invalid model or experiment")
        _require(models is None or cell["model"] in models, "Case outside campaign model scope")
        setting = cell.get("simulation_setting")
        _require(isinstance(setting, dict) and isinstance(setting.get("suffix"), str)
                 and setting["suffix"], "Invalid simulation setting")
        _require("inapplicability_reason" in cell and
                 (cell["inapplicability_reason"] is None or isinstance(cell["inapplicability_reason"], str)),
                 "Missing or invalid applicability declaration")
        identity = (cell["model"], cell["experiment"], cell["setting"], cell["seed"])
        suffix_identity = (cell["model"], cell["experiment"], setting["suffix"], cell["seed"])
        random_identity = (cell["model"], cell["experiment"], setting["suffix"], cell["random_seed"])
        _require(identity not in identities and suffix_identity not in suffix_identities
                 and random_identity not in random_identities,
                 f"Overlapping scientific case: {key}")
        identities.add(identity)
        suffix_identities.add(suffix_identity)
        random_identities.add(random_identity)
    if models is not None:
        _require(models == sorted({entry["cell"]["model"] for entry in cases}),
                 "Campaign scope declares models without planned cases")
    if index["schema_version"] == 3:
        _validate_policy(index)
    else:
        _require("allowed_missing_tasks" not in index, "Missing-task policy requires index schema version 3")


def _grid_size(configuration):
    return math.prod(len(configuration[name]) for name in ("init_penalties", "penalties_u", "penalties_v"))


def _policy_key(item):
    return item["run_root"], item["case_key"], item["task_id"]


def _validate_cancellation_reference(item):
    """Validate the indexed reference without reopening external evidence.

    The case audit verifies the referenced accounting and task environment;
    its input fingerprint and published receipt bind that verification. The
    campaign reducer consumes only the resulting metadata.
    """
    reference = item["cancellation_evidence"]
    _require(isinstance(reference, dict) and set(reference) == {
        "path", "sha256", "job_id", "step_id", "task_id"}, "Invalid cancellation evidence reference")
    _require(isinstance(reference["path"], str) and Path(reference["path"]).is_absolute()
             and isinstance(reference["sha256"], str) and HEX.fullmatch(reference["sha256"])
             and isinstance(reference["job_id"], str) and re.fullmatch(r"[1-9][0-9]*", reference["job_id"])
             and isinstance(reference["step_id"], str)
             and re.fullmatch(re.escape(reference["job_id"]) + r"\.[0-9]+", reference["step_id"])
             and reference["task_id"] == item["task_id"], "Invalid cancellation evidence identity")


def _validate_policy(index):
    policy = index.get("allowed_missing_tasks")
    _require(isinstance(policy, list) and policy, "Version 3 index requires an explicit nonempty missing-task policy")
    cases = {entry["case_key"]: entry for entry in index["cases"]}
    common = {"run_root", "case_key", "task_id", "grid_candidate_ids", "reason"}
    seen, grids = set(), defaultdict(set)
    for item in policy:
        _require(isinstance(item, dict) and set(item) in (
            common | {"launcher_exit_code"}, common | {"cancellation_evidence"}),
            "Invalid allowed-missing task declaration")
        key = item["case_key"]
        _require(isinstance(key, str) and key in cases, "Allowed-missing task references an unknown case")
        entry = cases[key]
        _require(item["run_root"] == entry["run_root"], "Allowed-missing task run root mismatch")
        _require(entry["cell"]["inapplicability_reason"] is None, "Inapplicable case cannot permit missing tuning tasks")
        ids = item["grid_candidate_ids"]
        _require(isinstance(ids, list) and ids and all(type(i) is int and 0 <= i < _grid_size(entry["configuration"])
                 for i in ids) and ids == sorted(set(ids)), "Invalid allowed-missing candidate IDs")
        _require(item["task_id"] == f"{entry['cell']['task_id']}_g{ids[0]}", "Allowed-missing task/grid identity mismatch")
        _require(isinstance(item["reason"], str) and item["reason"].strip(), "Missing policy reason")
        if "cancellation_evidence" in item:
            _validate_cancellation_reference(item)
        else:
            _require(type(item["launcher_exit_code"]) is int and 0 < item["launcher_exit_code"] <= 255,
                     "Invalid allowed launcher exit code")
        identity = _policy_key(item)
        _require(identity not in seen and not grids[key].intersection(ids), "Duplicate or overlapping missing-task policy")
        seen.add(identity)
        grids[key].update(ids)
    _require(policy == sorted(policy, key=_policy_key), "Missing-task policy is not canonically ordered")
    _require(all(len(grids[key]) < _grid_size(entry["configuration"]) for key, entry in cases.items()),
             "Missing-task policy cannot omit an entire tuning grid")


def _validate_missing_tuning(index, entry, audit):
    metadata = audit.get("missing_tuning")
    _require(isinstance(metadata, dict), "Version 3 audit omits missing-tuning metadata")
    allowed = [item for item in index["allowed_missing_tasks"] if item["case_key"] == entry["case_key"]]
    _require(_digest(metadata.get("allowed_tasks")) == _digest(allowed), "Audit allowed-missing policy differs from index")
    unavailable = metadata.get("unavailable_tasks")
    _require(isinstance(unavailable, list) and all(_digest(item) in {_digest(value) for value in allowed}
                                                 for item in unavailable),
             "Unavailable task is not explicitly allowed")
    _require(unavailable == sorted(unavailable, key=_policy_key)
             and len({_policy_key(item) for item in unavailable}) == len(unavailable),
             "Duplicate or unordered unavailable tasks")
    ids = sorted(candidate for item in unavailable for candidate in item["grid_candidate_ids"])
    planned = _grid_size(entry["configuration"])
    _require(_digest(metadata.get("unavailable_grid_candidate_ids")) == _digest(ids)
             and type(metadata.get("planned_candidate_count")) is int and metadata["planned_candidate_count"] == planned
             and type(metadata.get("available_candidate_count")) is int
             and metadata["available_candidate_count"] == planned-len(ids), "Missing-tuning grid/count mismatch")
    execution = audit["execution"]
    missing_ids = sorted(item["task_id"] for item in unavailable)
    _require(execution.get("allowed_missing_task_ids") == missing_ids, "Execution missing-task scope mismatch")
    declared_usable = _boolean(execution.get("available_tasks_success"), "available tasks success")
    outcomes = execution.get("task_outcomes")
    _require(isinstance(outcomes, list) and outcomes and all(isinstance(row, dict) for row in outcomes),
             "Policy audit requires individual execution outcomes")
    by_task = {}
    fit_success = launch_success = True
    remaining_clean = True
    archive_ids = []
    permitted = {item["task_id"]: item for item in unavailable}
    code_names = ("process-exit-code.txt", "exit-code.txt", "launcher-exit-code.txt")
    for outcome in outcomes:
        task_id = outcome.get("task_id")
        _require(isinstance(task_id, str) and task_id not in by_task, "Missing or duplicate execution task ID")
        by_task[task_id] = outcome
        codes = outcome.get("exit_codes")
        _require(isinstance(codes, dict) and all(name in codes for name in code_names), "Missing task exit markers")
        _require(all(codes[name] is None or (type(codes[name]) is int and 0 <= codes[name] <= 255)
                     for name in code_names), "Invalid task exit marker")
        process, worker, launcher = (codes[name] for name in code_names)
        archive = outcome.get("archive_status")
        archive_only = process == 0 and worker in (0, 74) and launcher == 74 and archive == "failed"
        if archive_only:
            archive_ids.append(task_id)
        else:
            fit_success = fit_success and process == 0
            launch_success = launch_success and worker == 0 and launcher == 0
        if task_id in permitted:
            if "cancellation_evidence" in permitted[task_id]:
                _require(process is None and worker is None and launcher is None
                         and archive in ("deferred", "unknown"),
                         "Unavailable task exit markers do not match its external cancellation policy")
            else:
                expected = permitted[task_id]["launcher_exit_code"]
                _require(process in (None, expected) and worker in (None, expected)
                         and launcher == expected and archive in ("deferred", "unknown"),
                         "Unavailable task exit markers do not match its explicit policy")
        else:
            remaining_clean = remaining_clean and (archive_only or (process == worker == launcher == 0))
    _require(set(permitted) <= set(by_task), "Allowed unavailable task omits execution outcome")
    _require(sorted(execution["archive_only_failures"]) == sorted(archive_ids), "Archive-only task classification mismatch")
    _require(execution["fit_success"] == fit_success and execution["launch_success"] == launch_success,
             "Policy must preserve truthful original execution flags")
    for issue in execution["issues"]:
        _require(issue.get("task_id") in by_task, "Execution issue references an unknown task")
        if issue["task_id"] not in permitted:
            remaining_clean = False
        else:
            _require(issue.get("kind") in ("fit_execution", "worker_or_launch_execution"),
                     "Missing-task policy cannot excuse an unrelated issue")
    if any(issue.get("kind") == "invalid_missing_task_permission" for issue in audit["scientific"]["issues"]):
        remaining_clean = False
    _require(declared_usable == remaining_clean, "Available-task success contradicts execution evidence")
    summary = audit.get("summary")
    if summary is not None:
        availability = ("unavailable_grid_candidate_ids", "planned_candidate_count", "available_candidate_count")
        if ids or any(key in summary for key in availability):
            for key in availability:
                _require(summary.get(key) == metadata[key] and type(summary.get(key)) is type(metadata[key]),
                         f"Summary {key} differs from missing-task audit")
        if ids:
            _require(not audit["scientific"]["budget_coverage_complete"], "Missing tuning tasks cannot certify full-grid coverage")
            for cap in summary["caps"]:
                _require(set(ids) <= set(cap["unresolved_candidate_ids"]) and not cap["coverage_complete"],
                         "Cap coverage hides an unavailable candidate")
                if cap["success"]:
                    _require(cap["winner_candidate_id"] % planned not in ids, "Selected winner is an unavailable candidate")
    return declared_usable


def _validate_summary(summary, entry, scientific):
    _require(isinstance(summary, dict), "Successful audit omits summary")
    cell, config = entry["cell"], _configuration(entry["configuration"])
    applicable = cell["inapplicability_reason"] is None
    expected = dict(model=f"model{cell['model']+1}", model_id=cell["model"],
                    experiment=f"exp{cell['experiment']+1}", setting=cell["simulation_setting"]["suffix"],
                    seed_id=cell["seed"], applicable=applicable, missing=False,
                    inapplicability_reason=cell["inapplicability_reason"])
    for name, value in expected.items():
        _require(summary.get(name) == value and type(summary.get(name)) is type(value),
                 f"Summary {name} differs from planned identity")
    _require(_configuration(summary.get("configuration")) == config, "Summary configuration mismatch")
    _fingerprint(summary.get("implementation_fingerprint"), "implementation fingerprint")
    _require(summary["implementation_fingerprint"] ==
             entry["source"]["implementation"][str(applicable).lower()]["fingerprint"],
             "Summary implementation differs from planned source")
    caps = summary.get("caps")
    _require(isinstance(caps, list), "Invalid summary caps")
    _require(summary.get("status") == scientific["status"], "Scientific/summary status mismatch")
    if not applicable:
        _require(summary["status"] == "inapplicable" and not caps,
                 "Inapplicable case contains fitting outcomes")
        return
    budgets = config["iteration_budgets"]
    grid_size = math.prod(len(config[name]) for name in ("init_penalties", "penalties_u", "penalties_v"))
    _require(all(isinstance(cap, dict) for cap in caps)
             and [cap.get("iteration_budget") for cap in caps] == budgets, "Summary budget scope mismatch")
    previous_success = False
    previous_complete = True
    for budget_index, cap in enumerate(caps):
        budget = _integer(cap["iteration_budget"], "cap budget", 1)
        success = _boolean(cap.get("success"), "cap success")
        complete = _boolean(cap.get("coverage_complete"), "cap coverage")
        _require(not previous_success or success, "Cumulative cap lost available winner")
        _require(previous_complete or not complete, "Later cap resolved an incomplete earlier grid")
        previous_success, previous_complete = success, complete
        unresolved = cap.get("unresolved_candidate_ids")
        _require(isinstance(unresolved, list), "Invalid unresolved candidate IDs")
        for candidate in unresolved:
            _require(_integer(candidate, "unresolved candidate") < grid_size,
                     "Unresolved candidate outside tuning grid")
        _require(unresolved == sorted(set(unresolved)), "Duplicate or unordered unresolved candidates")
        _require(complete == (not unresolved) and
                 _integer(cap.get("reached_candidates"), "reached candidates") == grid_size-len(unresolved),
                 "Cap coverage/count mismatch")
        _require(not complete or success, "Complete grid has no available winner")
        if success:
            for name in ("validation_mse", "coefficient_error"):
                _number(cap.get(name), name)
            selected = _integer(cap.get("selected_iteration"), "selected iteration")
            checkpoint = _integer(cap.get("selected_checkpoint"), "selected checkpoint")
            _require(selected <= checkpoint <= budget, "Selected iteration outside certified cap")
            _require(_integer(cap.get("winner_candidate_id"), "winner candidate") < grid_size*(budget_index+1),
                     "Winner candidate outside eligible tuning grid or cap")
            for name in ("optimizer_converged", "selected_converged", "selected_at_cap", "selected_near_cap"):
                _boolean(cap.get(name), name)
            _require(not cap["selected_converged"] or cap["optimizer_converged"],
                     "Selected convergence contradicts optimizer convergence")
            _require(cap["selected_at_cap"] == (selected == budget), "Selected-at-cap flag mismatch")
        else:
            for name in ("validation_mse", "coefficient_error", "selected_iteration", "selected_checkpoint",
                         "winner_candidate_id", "optimizer_converged", "selected_converged"):
                _require(cap.get(name) is None, f"Failed cap contains {name}")
    final = caps[-1]
    expected_status = ("complete" if final["coverage_complete"] else
                       "partial" if final["success"] else "all_candidates_failed")
    _require(summary["status"] == expected_status, "Scientific status contradicts final cap")
    _require(scientific["budget_coverage_complete"] == final["coverage_complete"],
             "Scientific budget coverage contradicts final cap")


def _published_audit(root, index, entry):
    case_root = root / "cases" / entry["case_key"]
    for directory in (root / "cases", case_root):
        _require(not directory.is_symlink(), "Published case directory is a symlink")
    current, _ = _read_json(case_root / "current.json")
    generation = _fingerprint(current.get("generation"), "generation")
    receipt_sha = _fingerprint(current.get("receipt_sha256"), "receipt SHA256")
    directory = case_root / "generations" / generation
    _require(not directory.parent.is_symlink() and not directory.is_symlink(),
             "Published generation directory is a symlink")
    receipt, _ = _read_json(directory / "receipt.json", receipt_sha)
    identity = dict(schema_version=index["schema_version"], case_key=entry["case_key"], generation=generation,
                    index_fingerprint=index["index_fingerprint"], plan_fingerprint=entry["plan_fingerprint"])
    mode = index.get("publication_mode", "full")
    if index["schema_version"] >= 2:
        identity["publication_mode"] = mode
    else:
        _require("publication_mode" not in receipt, "Version 1 receipt cannot declare a publication mode")
    for name, value in identity.items():
        _require(receipt.get(name) == value and type(receipt.get(name)) is type(value),
                 f"Receipt {name} mismatch")
    for name in ("input_fingerprint", "analysis_fingerprint"):
        _fingerprint(receipt.get(name), name)
    _require(generation == _digest(dict(index_fingerprint=index["index_fingerprint"],
                 case_key=entry["case_key"], input_fingerprint=receipt["input_fingerprint"],
                 analysis_fingerprint=receipt["analysis_fingerprint"])), "Generation identity mismatch")
    outputs = receipt.get("outputs")
    expected_outputs = {"audit", "merged"} | ({"selected_factors"} if index["schema_version"] >= 2 else set())
    _require(isinstance(outputs, dict) and set(outputs) == expected_outputs, "Invalid receipt output roster")
    descriptor = outputs.get("audit")
    _require(isinstance(descriptor, dict) and descriptor.get("path") == "audit.json",
             "Invalid audit output path")
    audit_sha = _fingerprint(descriptor.get("sha256"), "audit SHA256")
    merged = outputs.get("merged")
    if merged is not None:
        _require(isinstance(merged, dict) and merged.get("path") == "merged.json", "Invalid merged descriptor")
        _fingerprint(merged.get("sha256"), "merged SHA256")
    selected = outputs.get("selected_factors")
    if selected is not None:
        _require(isinstance(selected, dict) and selected.get("path") == "selected-factors.json",
                 "Invalid selected factors descriptor")
        _fingerprint(selected.get("sha256"), "selected factors SHA256")
    # Never read or stat either factor artifact: reduction inputs are compact.
    audit, _ = _read_json(directory / "audit.json", audit_sha)
    if index["schema_version"] == 1:
        _require("publication_mode" not in audit, "Version 1 audit cannot declare a publication mode")
    for name, value in identity.items():
        _require(audit.get(name) == value and type(audit.get(name)) is type(value),
                 f"Audit {name} mismatch")
    _require(audit.get("identity") == entry["cell"], "Audit identity differs from planned cell")
    _require(_configuration(audit.get("configuration")) == _configuration(entry["configuration"]),
             "Audit configuration differs from index")
    execution, scientific = audit.get("execution"), audit.get("scientific")
    _require(isinstance(execution, dict) and isinstance(scientific, dict), "Missing execution/scientific audit")
    for name in ("fit_success", "launch_success"):
        _boolean(execution.get(name), name)
    _require(isinstance(execution.get("archive_status"), str), "Missing archive status")
    archive_failures = execution.get("archive_only_failures")
    _require(isinstance(archive_failures, list) and all(isinstance(task, str) for task in archive_failures)
             and len(archive_failures) == len(set(archive_failures)), "Invalid archive-only task list")
    for envelope in (execution, scientific):
        _require(isinstance(envelope.get("issues"), list)
                 and all(isinstance(issue, dict) for issue in envelope["issues"]), "Invalid audit issues")
    _boolean(scientific.get("audit_passed"), "scientific audit flag")
    _boolean(scientific.get("budget_coverage_complete"), "scientific coverage flag")
    _require(isinstance(scientific.get("status"), str), "Missing scientific status")
    policy_usable = None
    if index["schema_version"] == 3:
        policy_usable = _validate_missing_tuning(index, entry, audit)
    else:
        _require("missing_tuning" not in audit, "Missing-tuning audit requires an explicit version 3 policy")
        _require("allowed_missing_task_ids" not in execution and "available_tasks_success" not in execution,
                 "Execution missing-task exceptions require an explicit version 3 policy")
        _require(not (audit.get("summary") or {}).get("unavailable_grid_candidate_ids"),
                 "Unavailable tuning candidates require an explicit version 3 policy")
    if scientific["audit_passed"]:
        _require(not scientific["issues"], "Successful scientific audit contains errors")
        _require((merged is not None and selected is None) if mode == "full" else
                 (merged is None and selected is not None), "Successful publication artifact/mode mismatch")
        _validate_summary(audit.get("summary"), entry, scientific)
    else:
        _require(audit.get("summary") is None and not scientific["budget_coverage_complete"],
                 "Failed scientific audit contains usable summary or complete coverage")
        _require(merged is None and selected is None, "Failed scientific audit advertises scientific artifacts")
    _require(audit.get("paper") is None or isinstance(audit["paper"], dict), "Invalid compact paper record")
    return audit, dict(generation=generation, receipt_sha256=receipt_sha, audit_sha256=audit_sha,
                       audit_path=(directory / "audit.json").relative_to(root).as_posix(),
                       receipt_path=(directory / "receipt.json").relative_to(root).as_posix(),
                       publication_mode=mode, factor_artifact=(dict(
                           path=(directory / descriptor["path"]).relative_to(root).as_posix(),
                           sha256=descriptor["sha256"]) if (descriptor := merged or selected) else None),
                       **(dict(policy_execution_usable=policy_usable) if index["schema_version"] == 3 else {}),
                       analysis_fingerprint=receipt["analysis_fingerprint"],
                       input_fingerprint=receipt["input_fingerprint"])


def _lean_summary(summary):
    if summary is None:
        return None
    keys = ("model", "model_id", "experiment", "setting", "seed_id", "status", "applicable",
            "inapplicability_reason", "missing", "trajectory_status_counts", "failed_trajectories",
            "verified_factor_states", "validation_audit", "configuration", "implementation_fingerprint",
            "unavailable_grid_candidate_ids", "planned_candidate_count", "available_candidate_count")
    value = {key: summary[key] for key in keys if key in summary}
    cap_keys = ("iteration_budget", "success", "coverage_complete", "reached_candidates",
                "validation_mse", "coefficient_error", "selected_iteration", "selected_checkpoint",
                "selected_near_cap", "selected_at_cap", "optimizer_converged", "selected_converged",
                "winner_candidate_id")
    value["caps"] = [{**{key: cap[key] for key in cap_keys if key in cap},
                      "unresolved_candidate_count": len(cap["unresolved_candidate_ids"])}
                     for cap in summary["caps"]]
    return value


def _lean_paper(audit, configuration):
    if audit["summary"] is None:
        return None
    grid_size = math.prod(len(configuration[key]) for key in ("init_penalties", "penalties_u", "penalties_v"))
    selected = []
    for cap in audit["summary"]["caps"]:
        choice = {key: cap[key] for key in ("iteration_budget", "success", "coverage_complete",
                   "selected_iteration", "selected_checkpoint", "validation_mse", "coefficient_error",
                   "optimizer_converged", "selected_converged")}
        choice.update(grid_candidate_id=None, params=None, winner_origin_budget=None)
        if cap["success"]:
            origin, grid_id = divmod(cap["winner_candidate_id"], grid_size)
            init_id, remainder = divmod(grid_id, len(configuration["penalties_u"])*len(configuration["penalties_v"]))
            u_id, v_id = divmod(remainder, len(configuration["penalties_v"]))
            choice.update(grid_candidate_id=grid_id, winner_origin_budget=configuration["iteration_budgets"][origin],
                          params=dict(init_penalty=configuration["init_penalties"][init_id],
                                      penalty_u=configuration["penalties_u"][u_id],
                                      penalty_v=configuration["penalties_v"][v_id]))
        selected.append(choice)
    paper = audit.get("paper") or {}
    return dict(selected=selected, **{key: paper[key] for key in
                ("fit_time_sec", "elapsed_time_sec", "time_scope") if key in paper})


def _lean_execution(execution):
    return {**{key: value for key, value in execution.items() if key != "task_outcomes"},
            "task_outcome_count": len(execution.get("task_outcomes", []))}


def _stats(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.fmean(values) if values else None,
                se=statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else None)


def _cap_statistics(rows):
    return dict(seed_ids=[row["seed_id"] for row, _ in rows], count=len(rows),
                **{name: _stats(cap[name] for _, cap in rows) for name in
                   ("validation_mse", "coefficient_error", "selected_iteration")},
                optimizer_converged=sum(cap["optimizer_converged"] for _, cap in rows),
                selected_converged=sum(cap["selected_converged"] for _, cap in rows))


def _group_summary(rows, *, policy=False):
    rows = sorted(rows, key=lambda row: row["seed_id"])
    first = rows[0]
    applicable = [row for row in rows if row["applicable"]]
    groups = dict(model=first["model"], experiment=first["experiment"], setting=first["setting"],
                  simulation_setting=first["simulation_setting"], configuration=first["configuration"],
                  configuration_fingerprint=first["configuration_fingerprint"],
                  expected=len(rows), expected_seed_ids=[row["seed_id"] for row in rows],
                  expected_applicable=len(applicable), expected_inapplicable=len(rows)-len(applicable),
                  recorded=sum(row["record_state"] == "published" for row in rows),
                  missing=sum(row["record_state"] == "missing" for row in rows),
                  invalid=sum(row["record_state"] == "invalid" for row in rows),
                  source_fingerprints=sorted({row["source_fingerprint"] for row in rows}),
                  analysis_fingerprints=sorted({row["provenance"]["analysis_fingerprint"]
                                               for row in rows if row["provenance"]}),
                  caps=[])
    implementations = {row["summary"]["implementation_fingerprint"] for row in rows if row["summary"]}
    # Per-case validation binds implementation to the source. A campaign group
    # nevertheless must not pool seeds produced by different implementations.
    groups["implementation_fingerprints"] = sorted(implementations)
    groups["implementation_consistent"] = len(implementations) <= 1
    if policy:
        groups["policy_missing_seed_ids"] = [row["seed_id"] for row in applicable if row["has_permitted_missing_tasks"]]
    for budget in first["configuration"]["iteration_budgets"]:
        candidates = [(row, next(cap for cap in row["summary"]["caps"] if cap["iteration_budget"] == budget))
                      for row in applicable if row["summary"]]
        usable = [(row, cap) for row, cap in candidates if row["execution_usable"] and cap["success"]
                  and groups["implementation_consistent"]]
        complete = [(row, cap) for row, cap in usable if cap["coverage_complete"]]
        incomplete = [(row, cap) for row, cap in usable if not cap["coverage_complete"]]
        groups["caps"].append(dict(iteration_budget=budget, expected=len(applicable),
            recorded=len(candidates), usable=len(complete), available_winners=len(usable),
            execution_excluded=sum(not row["execution_usable"] for row, _ in candidates),
            no_available_winner=sum(not cap["success"] for _, cap in candidates),
            complete_grid=_cap_statistics(complete), incomplete_grid_available=_cap_statistics(incomplete),
            primary_complete=len(complete) == len(applicable) and bool(applicable)))
        if policy:
            permitted = [(row, cap) for row, cap in candidates if row["policy_execution_usable"]
                         and cap["success"] and groups["implementation_consistent"]]
            sensitivity = [(row, cap) for row, cap in permitted if not row["has_permitted_missing_tasks"]]
            available_stats = _cap_statistics(permitted)
            available_stats.update(full_grid_coverage_count=sum(cap["coverage_complete"] for _, cap in permitted),
                available_grid_budget_complete_count=sum(cap["reached_candidates"] ==
                    row["missing_tuning"]["available_candidate_count"] for row, cap in permitted),
                missing_task_seed_ids=[row["seed_id"] for row, _ in permitted if row["has_permitted_missing_tasks"]])
            groups["caps"][-1].update(policy_available=available_stats,
                policy_sensitivity_excluding_missing=_cap_statistics(sensitivity),
                policy_sensitivity_expected=len(applicable)-len(groups["policy_missing_seed_ids"]),
                policy_available_complete=len(permitted) == len(applicable) and bool(applicable))
    return groups


def _atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_campaign(index_path, *, output_root=None, include_histories=False):
    """Return every planned case and separate complete/incomplete-grid statistics.

    Missing or invalid publications remain explicit report rows. An invalid
    index (including scientific overlap) raises ValueError before reduction.
    Output is written only when output_root is supplied. Detailed per-candidate
    histories stay in the referenced case audits unless include_histories=True.
    """
    index_path = Path(index_path).absolute()
    index, index_sha = _read_json(index_path)
    _validate_index(index)
    _boolean(include_histories, "include histories flag")
    policy = index["schema_version"] == 3
    rows, grouped = [], defaultdict(list)
    for entry in index["cases"]:
        cell = entry["cell"]
        config = _configuration(entry["configuration"])
        row = dict(case_key=entry["case_key"], run_root=entry["run_root"],
            model=f"model{cell['model']+1}", experiment=f"exp{cell['experiment']+1}",
            setting=cell["simulation_setting"]["suffix"], simulation_setting=cell["simulation_setting"],
            seed_id=cell["seed"], random_seed=cell["random_seed"],
            applicable=cell["inapplicability_reason"] is None,
            inapplicability_reason=cell["inapplicability_reason"], configuration=entry["configuration"],
            configuration_fingerprint=_digest(config), source_fingerprint=_digest(entry["source"]),
            plan_fingerprint=entry["plan_fingerprint"], seed_file_sha256=entry["seed_file_sha256"],
            record_state="missing", issues=[], execution=None, scientific=None, summary=None, paper=None,
            provenance=None, execution_usable=False)
        if policy:
            row.update(missing_tuning=None, policy_execution_usable=False, has_permitted_missing_tasks=False)
        try:
            audit, provenance = _published_audit(index_path.parent, index, entry)
            row.update(record_state="published", scientific=audit["scientific"], provenance=provenance,
                       execution=audit["execution"] if include_histories else _lean_execution(audit["execution"]),
                       summary=audit["summary"] if include_histories else _lean_summary(audit["summary"]),
                       paper=audit.get("paper") if include_histories else _lean_paper(audit, config),
                       execution_usable=(audit["execution"]["fit_success"] and audit["execution"]["launch_success"]
                                         and not audit["execution"]["issues"]))
            if policy:
                row.update(missing_tuning=audit["missing_tuning"],
                           policy_execution_usable=provenance["policy_execution_usable"],
                           has_permitted_missing_tasks=bool(audit["missing_tuning"]["unavailable_tasks"]))
        except FileNotFoundError as error:
            row["issues"].append(dict(error="missing_publication", detail=str(error)))
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
            row.update(record_state="invalid", issues=[dict(error="invalid_publication", detail=str(error))])
        rows.append(row)
        key = (row["model"], row["experiment"], _digest(row["simulation_setting"]), row["configuration_fingerprint"])
        grouped[key].append(row)
    groups = [_group_summary(grouped[key], policy=policy) for key in sorted(grouped)]
    counts = Counter(row["record_state"] for row in rows)
    report = dict(schema_version=2, method=METHOD, index_path=str(index_path), index_sha256=index_sha,
        input_schema_version=index["schema_version"], publication_mode=index.get("publication_mode", "full"),
        scope=index.get("scope", dict(models=sorted({entry["cell"]["model"] for entry in index["cases"]}))),
        include_histories=include_histories,
        index_fingerprint=index["index_fingerprint"], completed_utc=datetime.now(timezone.utc).isoformat(),
        expected_cases=len(rows), recorded_cases=counts["published"], missing_cases=counts["missing"],
        invalid_cases=counts["invalid"], expected_applicable=sum(row["applicable"] for row in rows),
        expected_inapplicable=sum(not row["applicable"] for row in rows),
        audited_inapplicable=sum(not row["applicable"] and row["summary"] is not None for row in rows),
        scientific_audit_failed=sum(row["scientific"] is not None and not row["scientific"]["audit_passed"] for row in rows),
        archive_only_failed_tasks=sum(len(row["execution"]["archive_only_failures"]) for row in rows if row["execution"]),
        implementation_inconsistent_groups=sum(not group["implementation_consistent"] for group in groups),
        publication_complete=counts["published"] == len(rows),
        primary_complete=(all(row["summary"] is not None and row["execution_usable"] for row in rows)
                          and all(group["implementation_consistent"] and all(cap["primary_complete"]
                              for cap in group["caps"]) for group in groups if group["expected_applicable"])),
        statistical_unit="Individual saved seeds within one model, experiment, setting and runner configuration",
        interpretation="Complete-grid statistics are primary; incomplete-grid available winners are descriptive. "
                       "Budget completion and optimizer convergence are reported separately. Validation MSE uses "
                       "the reused tuning sample and is not an independent test error.",
        cases=rows, groups=groups)
    if policy:
        report.update(schema_version=3, allowed_missing_tasks=index["allowed_missing_tasks"],
            allowed_missing_task_count=len(index["allowed_missing_tasks"]),
            permitted_missing_task_count=sum(len(row["missing_tuning"]["unavailable_tasks"])
                                             for row in rows if row["missing_tuning"]),
            permitted_missing_case_count=sum(row["has_permitted_missing_tasks"] for row in rows),
            allowed_missing_policy_complete=(all(row["summary"] is not None and row["policy_execution_usable"]
                for row in rows) and all(group["implementation_consistent"] and all(cap["policy_available_complete"]
                for cap in group["caps"]) for group in groups if group["expected_applicable"])),
            allowed_missing_policy_interpretation="Policy-available statistics include audited winners from explicitly "
                "permitted missing-task cases and existing valid partial-budget trajectories. The sensitivity excludes "
                "cases with actual permitted missing tasks. Policy completion means a permitted winner exists for every "
                "expected case at each budget; it does not mean full-grid budget coverage or optimizer convergence.")
    if output_root is not None:
        _atomic_json(report, Path(output_root).absolute() / "campaign-summary.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Prepared campaign-index.json")
    parser.add_argument("--output-root", type=Path, help="Write campaign-summary.json atomically here")
    parser.add_argument("--include-histories", action="store_true",
                        help="Also copy detailed scalar histories from case audits into the campaign report")
    args = parser.parse_args(argv)
    try:
        report = summarize_campaign(args.index, output_root=args.output_root, include_histories=args.include_histories)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Campaign summary failed: {error}\n")
    if args.output_root is not None:
        printed = {key: report[key] for key in (
            "schema_version", "method", "index_fingerprint", "completed_utc", "expected_cases",
            "publication_mode", "scope", "include_histories",
            "recorded_cases", "missing_cases", "invalid_cases", "expected_applicable",
            "expected_inapplicable", "audited_inapplicable", "scientific_audit_failed",
            "archive_only_failed_tasks", "implementation_inconsistent_groups", "publication_complete",
            "primary_complete")}
        printed["output_path"] = str(args.output_root.absolute() / "campaign-summary.json")
        if "allowed_missing_policy_complete" in report:
            printed.update({key: report[key] for key in ("allowed_missing_policy_complete", "allowed_missing_task_count",
                           "permitted_missing_task_count", "permitted_missing_case_count")})
    else:
        printed = report
    print(json.dumps(printed, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("allowed_missing_policy_complete", report["primary_complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
