"""Independent artifact audit and paired Monte Carlo summaries.

Production array processing requires Slurm and a /scratch2 root. Tiny local
fixtures may explicitly opt in. Neither missing tasks nor failed methods are
silently removed: every planned method receives a per-replication status and
all summaries retain planned, successful, failed, missing and invalid counts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


BASE_METHODS = ("target_rrr", "oracle_subspace_rrr", "source_subspace_rrr",
                "target_ridge_rrr", "source_subspace_ridge_rrr", "ridge_to_source",
                "source_target_mixture", "nuclear_contrast")
METRICS = ("coefficient_rmse", "coefficient_squared_error",
           "population_prediction_excess", "validation_mse", "training_mse",
           "test_mse", "numerical_rank", "total_seconds")
PAIRED_METRICS = ("coefficient_rmse", "population_prediction_excess", "total_seconds")
REFERENCES = ("target_rrr", "target_ridge_rrr")
PARK_EXTERNAL = "park_two_stage_nr_external_validation"
Z975 = 1.959963984540054


def _fingerprint(arrays):
    """Independent reproduction of the runner's typed-array SHA-256 contract."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        descriptor = json.dumps([key, list(value.shape), value.dtype.str],
                                separators=(",", ":"), sort_keys=True)
        digest.update(descriptor.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".audit-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _moments(values):
    values = np.asarray(values, dtype=float)
    count = len(values)
    if count == 0:
        return dict(n=0, mean=None, mcse=None, ci95_low=None, ci95_high=None)
    mean = float(np.mean(values))
    mcse = float(np.std(values, ddof=1) / np.sqrt(count)) if count > 1 else None
    return dict(n=count, mean=mean, mcse=mcse,
                ci95_low=None if mcse is None else mean - Z975 * mcse,
                ci95_high=None if mcse is None else mean + Z975 * mcse)


def _wilson(harms, count):
    if count == 0:
        return None, None
    frequency = harms / count
    denominator = 1 + Z975**2 / count
    center = (frequency + Z975**2 / (2 * count)) / denominator
    half = Z975 * math.sqrt(frequency * (1 - frequency) / count +
                            Z975**2 / (4 * count**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _expected_methods(plan, task):
    if "expected_methods" in task:
        return list(task["expected_methods"])
    if "expected_methods" in plan:
        declared = plan["expected_methods"]
        if isinstance(declared, dict):
            result = list(declared.get("benchmarks", BASE_METHODS))
            for variant in task.get("variants", ["full_caps"]):
                result.extend(declared.get(variant, []))
            return result
        return list(declared)
    result = list(BASE_METHODS)
    for variant in task.get("variants", ["full_caps"]):
        suffix = "" if variant == "full_caps" else "_" + variant
        result.extend(name + suffix for name in ("v2", "v2_transfer_only", "initializer_only"))
    return result


def _expected_candidates(config, variant):
    frozen = config.get("expected_v2_candidates", {})
    if variant in frozen:
        return int(frozen[variant])
    if variant in ("active_caps", "cap_control"):
        if "init_penalties" not in config:
            return None
        return len(config["init_penalties"]) * 4 * (3 if variant == "active_caps" else 1)
    required = ("init_penalties", "penalties_u", "penalties_v")
    if not all(key in config for key in required):
        return None
    pairs = sum(not (left == right == 0) for left in config["penalties_u"]
                for right in config["penalties_v"])
    return 1 + len(config["init_penalties"]) * pairs


def _declared_specs(case, config, variant):
    """Rebuild the frozen schema-1 grid without importing the fitting runner."""
    rank = int(case.get("fitted_rank", case["target_rank"]))
    source_rank = int(case["source_rank"])
    free = rank if variant in ("active_caps", "cap_control") else max(rank, min(source_rank, 10))
    limits = [(case["p"]-free)*rank, (case["q"]-free)*rank]
    def spec(initial, left, right, caps):
        return dict(rank=rank, source_rank=source_rank, free_directions=[free, free],
                    init_penalty=initial, penalty=[left, right], support_limits=caps)
    if variant in ("active_caps", "cap_control"):
        caps = [rank, 2*rank, 4*rank] if variant == "active_caps" else [max(limits)]
        return [spec(initial, penalty, penalty, [min(cap, limit) for limit in limits])
                for initial, penalty, cap in product(config["init_penalties"], [0., .0025, .01, .04], caps)]
    return [spec(0., 0., 0., limits)] + [spec(initial, left, right, limits)
        for initial, left, right in product(config["init_penalties"], config["penalties_u"], config["penalties_v"])
        if not left == right == 0]


def _specs_digest(specs):
    return hashlib.sha256(json.dumps(specs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _benchmark_grid(method, case, config):
    rank = int(case.get("fitted_rank", case["target_rank"]))
    source_rank = int(case["source_rank"])
    if method in ("target_rrr", "oracle_subspace_rrr", "source_subspace_rrr"):
        return [dict(method=method, rank=rank, source_rank=source_rank)]
    if method in ("target_ridge_rrr", "source_subspace_ridge_rrr", "ridge_to_source"):
        return [dict(method=method, rank=rank, source_rank=source_rank, ridge=value)
                for value in config["benchmark_ridges"] if method != "ridge_to_source" or value > 0]
    if method == "source_target_mixture":
        return [dict(method=method, rank=rank, ridge=ridge, alpha=alpha)
                for ridge, alpha in product(config["benchmark_ridges"], config["mixture_weights"])]
    if method == "nuclear_contrast":
        return [dict(method=method, rank=rank, nuclear_penalty=value) for value in config["nuclear_penalties"]]
    return None


def _park_external_checks(record, config, data, case):
    """Audit the published-method wrapper's externally selected candidate grid.

    The R bridge records SSE/(2*n_validation). The runner's validation_score
    and paper metrics are MSE = 2*bridge_loss/q. Eligibility refers only to the
    authors' squared-update stopping rule, not a global optimum certificate.
    """
    findings = []
    def add(code, message, severity="error"):
        findings.append((code, message, severity))
    diagnostics = record.get("diagnostics", {})
    options = diagnostics.get("options", {})
    candidate_records = record.get("candidates", diagnostics.get("candidate_results", []))
    grid_w = config.get("park_lambda_w", config.get("park_lambdas"))
    grid_delta = config.get("park_lambda_delta", config.get("park_lambdas"))
    if grid_w is None or grid_delta is None:
        add("park_missing_declared_grid", "The plan does not declare Park tuning penalties")
        return findings, None
    if options.get("lambda_w") != grid_w or options.get("lambda_delta") != grid_delta:
        add("park_options_grid_mismatch", "Bridge lambda options differ from frozen plan")
    expected_convergence = config.get("park_require_convergence", True)
    if options.get("require_convergence") is not expected_convergence:
        add("park_convergence_policy_mismatch", "Bridge stopping-rule eligibility differs from frozen policy")
    if options.get("mode") != "external_validation" or diagnostics.get("refit") is not False:
        add("park_validation_protocol_mismatch", "External target holdout and no refit were not recorded")
    if diagnostics.get("tuning") != "revision_independent_target_holdout_pair_selection":
        add("park_tuning_scope_mismatch", "Published-method record does not use the declared external holdout")
    if "source_X" not in data or "source_Y" not in data:
        add("park_raw_source_missing", "Published raw-source method has no available source sample")
    else:
        dimensions = dict(n_train=len(data["X"]), n_source=len(data["source_X"]),
                          n_predictors=case["p"], n_responses=case["q"],
                          n_validation=len(data["X_validation"]))
        if data["source_X"].shape != (dimensions["n_source"], case["p"]) or \
                data["source_Y"].shape != (dimensions["n_source"], case["q"]):
            add("park_raw_source_shape_mismatch", "Raw source arrays have incompatible dimensions")
        for name, expected in dimensions.items():
            if diagnostics.get(name) != expected:
                add("park_input_dimension_mismatch", f"{name}: recorded={diagnostics.get(name)}, expected={expected}")
    success = record.get("success") is True
    if success and "intercept" not in record:
        add("park_missing_intercept", "Published-method success requires its fitted intercept")
    if not candidate_records and not success:
        add("park_failure_without_candidate_trace", "Whole-method failure is retained; no completed candidate trace is available", "warning")
        return findings, None
    if "candidate_results" in diagnostics and candidate_records != diagnostics["candidate_results"]:
        add("park_candidate_record_mismatch", "Top-level candidates disagree with bridge diagnostics")
    expected_pairs = list(product(grid_w, grid_delta))
    actual_pairs = [(candidate.get("lambda_w"), candidate.get("lambda_delta")) for candidate in candidate_records]
    expected_indices = list(range(len(expected_pairs)))
    if actual_pairs != expected_pairs or [candidate.get("candidate_index") for candidate in candidate_records] != expected_indices:
        severity = "error" if success else "warning"
        add("park_candidate_grid_mismatch", "Candidate trace omits, reorders, duplicates or changes declared lambda pairs", severity)
    admm = diagnostics.get("admm_fits", [])
    for candidate in candidate_records:
        index, status = candidate.get("candidate_index"), candidate.get("status")
        if status not in ("ok", "failed", "uncertified"):
            add("park_missing_candidate_outcome", f"Candidate {index} has no recognized explicit outcome")
            continue
        if status == "failed":
            if not candidate.get("error"):
                add("park_unexplained_candidate_failure", f"Candidate {index} failed without error text")
            continue
        loss = candidate.get("validation_loss")
        if not _close(loss, loss) or loss < 0:
            add("park_nonfinite_candidate_score", f"Candidate {index} has invalid validation loss")
        declared_convergence = []
        for stage in ("pooled", "correction"):
            flag = candidate.get(stage + "_stopping_criterion_met")
            position = candidate.get(stage + "_admm_index")
            if not isinstance(flag, bool) or not isinstance(position, int) or not 1 <= position <= len(admm):
                add("park_admm_reference_mismatch", f"Candidate {index}: missing/invalid {stage} ADMM reference")
                continue
            fit = admm[position - 1]
            update, tolerance = fit.get("last_max_squared_update"), fit.get("tolerance")
            computed = _close(update, update) and _close(tolerance, tolerance) and update <= tolerance
            if flag != computed or fit.get("stopping_criterion_met") is not computed:
                add("park_stopping_flag_mismatch", f"Candidate {index}: {stage} stopping flag does not match saved residual/tolerance")
            declared_convergence.append(flag)
        if len(declared_convergence) == 2:
            expected_status = "ok" if not expected_convergence or all(declared_convergence) else "uncertified"
            if status != expected_status:
                add("park_candidate_eligibility_mismatch", f"Candidate {index}: status inconsistent with stopping policy")
    eligible = [candidate for candidate in candidate_records if candidate.get("status") == "ok" and
                _close(candidate.get("validation_loss"), candidate.get("validation_loss"))]
    if not success:
        if eligible:
            add("park_failure_despite_eligible_candidates", "Method marked failed with eligible completed candidates")
        return findings, None
    if diagnostics.get("preprocessing_scope") != "target_training_and_raw_source_only_validation_rows_excluded":
        add("park_preprocessing_scope_mismatch", "Record does not assert training-only preprocessing")
    if diagnostics.get("candidate_count") != len(expected_pairs):
        add("park_candidate_count_mismatch", "Bridge candidate count differs from complete declared grid")
    selected_index = record.get("selected_index")
    chosen = next((candidate for candidate in eligible if candidate.get("candidate_index") == selected_index), None)
    if chosen is None:
        add("park_ineligible_selected_candidate", "Selected index does not identify an eligible candidate")
        return findings, None
    best = min(eligible, key=lambda candidate: (candidate["validation_loss"], candidate["candidate_index"]))
    if chosen["candidate_index"] != best["candidate_index"]:
        add("park_nonminimal_validation_selection", "Winner violates minimum eligible loss and first-pair tie rule")
    mse = 2 * chosen["validation_loss"] / case["q"]
    if not _close(record.get("validation_score"), mse):
        add("park_validation_score_scale_mismatch", "Runner score must convert bridge SSE/(2*n) to response-averaged MSE")
    if diagnostics.get("selected_index") != selected_index or \
            not _close(diagnostics.get("validation_loss"), chosen["validation_loss"]):
        add("park_selected_diagnostics_mismatch", "Runner selection differs from bridge selection")
    if not _close(diagnostics.get("selected_lambda_w"), chosen["lambda_w"]) or \
            not _close(diagnostics.get("selected_lambda_delta"), chosen["lambda_delta"]):
        add("park_selected_penalty_mismatch", "Selected penalty diagnostics differ from chosen grid pair")
    return findings, mse


def _recompute(coefficient, data, truth, intercept=None):
    error = coefficient - truth["C_star"]
    intercept = np.zeros(coefficient.shape[1]) if intercept is None else np.asarray(intercept)
    metrics = dict(coefficient_rmse=float(np.sqrt(np.mean(error**2))),
        coefficient_squared_error=float(np.sum(error**2)),
        population_prediction_excess=float((np.sum(error * (truth["Sigma_x"] @ error)) + np.sum(intercept**2)) /
                                           coefficient.shape[1]),
        validation_mse=float(np.mean((data["Y_validation"] - data["X_validation"] @ coefficient - intercept)**2)),
        training_mse=float(np.mean((data["Y"] - data["X"] @ coefficient - intercept)**2)),
        numerical_rank=int(np.linalg.matrix_rank(coefficient)))
    if "X_test" in truth and len(truth["X_test"]):
        metrics["test_mse"] = float(np.mean((truth["Y_test"] - truth["X_test"] @ coefficient - intercept)**2))
    return metrics


def _close(first, second):
    try:
        return bool(np.isfinite(first) and np.isfinite(second) and
                    np.isclose(first, second, rtol=2e-8, atol=2e-10))
    except (ValueError, TypeError):
        return False


def _summaries(rows):
    groups = defaultdict(list)
    indexed = {}
    for row in rows:
        groups[row["case_id"], row["method"]].append(row)
        indexed[row["task_index"], row["method"]] = row
    aggregate, paired = [], []
    for (case_id, method), members in sorted(groups.items()):
        base = {key: members[0][key] for key in
                ("case_id", "family", "level", "n_train", "p", "q", "method", "variant")}
        counts = dict(n_planned=len(members),
            n_success=sum(r["status"] == "success" for r in members),
            n_failure=sum(r["status"] == "failed" for r in members),
            n_missing=sum(r["status"] in ("missing_task", "incomplete_task", "missing_method") for r in members),
            n_invalid=sum(r["status"] == "audit_invalid" for r in members))
        for metric in METRICS:
            values = [r[metric] for r in members if r["status"] == "success" and
                      r.get(metric) is not None]
            aggregate.append(dict(base, metric=metric, **counts, **_moments(values)))
        for reference in REFERENCES:
            if method == reference:
                continue
            for metric in PAIRED_METRICS:
                differences, reference_success = [], 0
                for row in members:
                    other = indexed.get((row["task_index"], reference))
                    if other and other["status"] == "success":
                        reference_success += 1
                    if row["status"] == "success" and other and other["status"] == "success" and \
                            row.get(metric) is not None and other.get(metric) is not None:
                        differences.append(row[metric] - other[metric])
                count = len(differences)
                harms = [value for value in differences if value > 1e-12]
                low, high = _wilson(len(harms), count)
                paired.append(dict(base, reference=reference, metric=metric,
                    n_planned=len(members), n_method_success=counts["n_success"],
                    n_reference_success=reference_success, n_paired=count,
                    n_unavailable_pairs=len(members)-count, **_moments(differences),
                    harm_count=len(harms), harm_frequency=len(harms)/count if count else None,
                    harm_wilson95_low=low, harm_wilson95_high=high,
                    mean_positive_excess=float(np.mean(np.maximum(differences, 0))) if count else None,
                    mean_harm_conditional=float(np.mean(harms)) if harms else None,
                    maximum_harm=max(harms) if harms else (0.0 if count else None)))
    return aggregate, paired


def _load_plan(root, allow_tiny_local=False):
    """Enforce the production guard against the entire original plan."""
    root = Path(root).resolve()
    plan_bytes = (root / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    production = bool(os.environ.get("SLURM_JOB_ID")) and str(root).startswith("/scratch2/")
    if not production:
        tiny = (allow_tiny_local and len(plan.get("tasks", [])) <= 8 and
                all(max(c.get("p", 10**9), c.get("q", 10**9)) <= 32 and
                    max(c.get("n_train", 0), c.get("n_validation", 0), c.get("n_test", 0)) <= 64
                    for c in plan.get("cases", [])))
        if not tiny:
            raise RuntimeError("Production auditing requires Slurm and /scratch2; local opt-in is limited to tiny fixtures")
    return root, plan, hashlib.sha256(plan_bytes).hexdigest()


def audit(root, *, write=True, allow_tiny_local=False, task_indices=None):
    """Audit a frozen plan, optionally returning an explicitly partial shard.

    The local exception is restricted to <=8 total planned tasks and dimensions
    <=32, regardless of shard size. Subsets require write=False and cannot claim
    a passing campaign audit. They retain original indices and plan hashes.
    """
    root, plan, plan_hash = _load_plan(root, allow_tiny_local)
    if task_indices is not None:
        if write:
            raise ValueError("Task-subset audits require write=False")
        task_indices = list(task_indices)
        if (any(type(index) is not int or not 0 <= index < len(plan["tasks"]) for index in task_indices)
                or len(set(task_indices)) != len(task_indices)):
            raise ValueError("Task subset must contain unique valid original indices")
        task_indices = sorted(task_indices)
    selected_indices = range(len(plan["tasks"])) if task_indices is None else task_indices
    issues, rows, task_reports = [], [], []
    def issue(code, message, task_index=None, method=None, severity="error"):
        issues.append(dict(code=code, message=message, task_index=task_index,
                           method=method, severity=severity))
    cases, tasks = plan["cases"], plan["tasks"]
    if len(cases) != plan.get("n_cases", len(cases)) or len(tasks) != plan.get("n_tasks", len(tasks)):
        issue("plan_count_mismatch", "Declared plan counts differ from case/task lists")
    if len({case["case_id"] for case in cases}) != len(cases):
        issue("duplicate_case_id", "Case IDs are not unique")
    identities = [(task["case_index"], task["seed"], tuple(task.get("variants", ["full_caps"]))) for task in tasks]
    if len(set(identities)) != len(identities):
        issue("duplicate_task", "Plan repeats an identical case/seed/variant task")
    for index, case in enumerate(cases):
        if not any(task["case_index"] == index for task in tasks):
            issue("unplanned_case", f"Case {case['case_id']} has no planned task")
    planned_dirs = {f"task-{i:06d}" for i in range(len(tasks))}
    if (root / "tasks").exists():
        for directory in (root / "tasks").glob("task-*"):
            if directory.name not in planned_dirs:
                issue("unplanned_task_artifact", str(directory.relative_to(root)))
    if "expected_methods" not in plan and not all("expected_methods" in task for task in tasks):
        issue("inferred_method_plan", "Expected methods inferred from schema-1 baseline/variant contract", severity="warning")
    pairing_reference = {}
    for index in selected_indices:
        task = tasks[index]
        if not 0 <= task["case_index"] < len(cases):
            issue("bad_case_index", "Task references a nonexistent case", index)
            continue
        case = cases[task["case_index"]]
        expected = _expected_methods(plan, task)
        directory = root / "tasks" / f"task-{index:06d}"
        report = dict(task_index=index, case_id=case["case_id"], seed=task["seed"],
                      state="missing", expected_methods=expected)
        task_reports.append(report)
        base = dict(task_index=index, case_id=case["case_id"], family=case.get("family", ""),
                    level=case.get("level", ""), seed=task["seed"], n_train=case["n_train"],
                    p=case["p"], q=case["q"], source_rank=case.get("source_rank"),
                    target_rank=case.get("fitted_rank", case.get("target_rank")))
        result_path = directory / "result.json"
        if not result_path.exists():
            incomplete = (directory / "partial.json").exists()
            report["state"] = "incomplete" if incomplete else "missing"
            issue("incomplete_task" if incomplete else "missing_task", "No completed result.json", index)
            rows.extend(dict(base, method=method, variant=_method_variant(method),
                             status="incomplete_task" if incomplete else "missing_task") for method in expected)
            continue
        start_issues = len(issues)
        try:
            result = _read_json(result_path)
        except Exception as exc:
            issue("unreadable_result", str(exc), index)
            report["state"] = "invalid"
            rows.extend(dict(base, method=method, variant=_method_variant(method),
                             status="audit_invalid") for method in expected)
            continue
        report["state"] = "complete" if result.get("complete") is True else "incomplete"
        if result.get("complete") is not True:
            issue("false_complete", "result.json exists without complete=true", index)
        if result.get("plan_sha256") != plan_hash:
            issue("plan_fingerprint_mismatch", "Result was not generated under these exact plan bytes", index)
        if result.get("task") != task or result.get("case") != case:
            issue("task_identity_mismatch", "Result task/case differs from the frozen plan", index)
        metadata = result.get("metadata", {})
        if metadata.get("seed") != task["seed"] or metadata.get("case_id") != case["case_id"]:
            issue("metadata_identity_mismatch", "Generator metadata seed or case ID mismatch", index)
        data = truth = coefficients = None
        try:
            data, truth = _arrays(directory / "data.npz"), _arrays(directory / "truth.npz")
            coefficients = _arrays(directory / "coefficients.npz")
            if _fingerprint(data) != result.get("fit_data_fingerprint"):
                issue("fit_data_fingerprint_mismatch", "Saved data hash differs from the fitting record", index)
            if _fingerprint(truth) != result.get("truth_fingerprint"):
                issue("truth_fingerprint_mismatch", "Saved truth hash differs from the evaluation record", index)
            for kind, archive in (("data", data), ("truth", truth)):
                if not all(np.isfinite(value).all() for value in archive.values()):
                    issue("nonfinite_data", f"{kind}.npz contains nonfinite values", index)
            forbidden = {"C_star", "U0", "V0", "Sigma_x", "clean_source", "X_test", "Y_test"}
            if forbidden & data.keys():
                issue("truth_in_fit_archive", "Truth/test objects were included in data.npz", index)
            shapes = {"X": (case["n_train"], case["p"]), "Y": (case["n_train"], case["q"]),
                      "X_validation": (case["n_validation"], case["p"]),
                      "Y_validation": (case["n_validation"], case["q"]), "C0": (case["p"], case["q"])}
            for name, shape in shapes.items():
                if name not in data or data[name].shape != shape:
                    issue("data_shape_mismatch", f"{name} does not have planned shape {shape}", index)
            if truth["C_star"].shape != (case["p"], case["q"]) or truth["Sigma_x"].shape != (case["p"], case["p"]):
                issue("truth_shape_mismatch", "Truth coefficient/covariance dimensions mismatch", index)
            # Same seed and design law imply identical covariates/noise across stress cases.
            pairing_key = (task["seed"], case["p"], case["q"], case.get("rho", .5), case.get("noise_sd", .5))
            current = dict(X=data["X"], X_validation=data["X_validation"],
                           E=data["Y"] - data["X"] @ truth["C_star"],
                           E_validation=data["Y_validation"] - data["X_validation"] @ truth["C_star"])
            if pairing_key in pairing_reference:
                previous = pairing_reference[pairing_key]
                for name, value in current.items():
                    count = min(len(value), len(previous[name]))
                    matched = (np.array_equal(value[:count], previous[name][:count]) if name.startswith("X")
                               else np.allclose(value[:count], previous[name][:count], rtol=2e-11, atol=2e-11))
                    if not matched:
                        issue("paired_data_mismatch", f"Common-seed {name} differs across matched design settings", index)
            else:
                pairing_reference[pairing_key] = current
        except Exception as exc:
            issue("unreadable_or_invalid_arrays", str(exc), index)
        task_invalid = any(entry["severity"] == "error" for entry in issues[start_issues:])
        methods = result.get("methods", {})
        for reference in REFERENCES:
            if reference not in methods:
                issue("missing_reference", f"Missing paired comparator {reference}", index, reference)
        unexpected = sorted(set(methods) - set(expected))
        if unexpected:
            issue("unplanned_methods", ", ".join(unexpected), index)
        candidate_archives, initializer_archives = {}, {}
        for variant in task.get("variants", ["full_caps"]):
            candidate_start = len(issues)
            try:
                archive = _read_json(directory / variant / "candidates.json")
                candidates = archive["candidates"]
                specs = _declared_specs(case, plan["configuration"], variant)
                if archive.get("specs_sha256") != _specs_digest(specs):
                    issue("candidate_specs_fingerprint_mismatch", f"{variant}: frozen grid hash mismatch", index)
                if [candidate.get("spec") for candidate in candidates] != specs:
                    issue("candidate_specs_mismatch", f"{variant}: recorded specs differ from declared grid", index)
                if archive.get("complete") is not True:
                    issue("incomplete_candidates", f"{variant} candidate list is not complete", index)
                if archive.get("variant") != variant:
                    issue("candidate_variant_mismatch", variant, index)
                count = _expected_candidates(plan.get("configuration", {}), variant)
                declared_count = archive.get("expected_n_candidates", count)
                if declared_count is None:
                    issue("unknown_candidate_count", f"Cannot establish expected count for {variant}", index)
                elif len(candidates) != declared_count or (count is not None and declared_count != count):
                    issue("candidate_count_mismatch", f"{variant}: observed={len(candidates)}, declared={declared_count}, expected={count}", index)
                if [candidate.get("index") for candidate in candidates] != list(range(len(candidates))):
                    issue("candidate_index_mismatch", f"{variant}: missing, reordered or repeated candidate indices", index)
                eligible = [candidate for candidate in candidates if candidate.get("eligible") is True]
                if archive.get("n_eligible") != len(eligible):
                    issue("eligible_count_mismatch", variant, index)
                for candidate in candidates:
                    if not isinstance(candidate.get("eligible"), bool):
                        issue("missing_candidate_outcome", f"{variant} index {candidate.get('index')}", index)
                    if candidate.get("eligible") and (candidate.get("exception") is not None or
                        not _close(candidate.get("validation_mse"), candidate.get("validation_mse"))):
                        issue("invalid_eligible_candidate", f"{variant} index {candidate.get('index')}", index)
                    if candidate.get("eligible"):
                        try:
                            C = _arrays(directory / variant / f"candidate-{candidate['index']:04d}.npz")["coefficient"]
                            if C.shape != (case["p"], case["q"]) or not np.isfinite(C).all():
                                raise ValueError("Invalid shape/nonfinite candidate coefficient")
                            score = float(np.mean((data["Y_validation"] - data["X_validation"] @ C)**2))
                            if not _close(score, candidate["validation_mse"]):
                                issue("candidate_validation_mismatch", f"{variant} index {candidate['index']}", index)
                        except Exception as exc:
                            issue("invalid_candidate_coefficient", f"{variant} index {candidate['index']}: {exc}", index)
                actual_files = {path.name for path in (directory / variant).glob("candidate-*.npz")}
                expected_files = {f"candidate-{candidate['index']:04d}.npz" for candidate in eligible}
                if actual_files != expected_files:
                    issue("candidate_coefficient_inventory_mismatch", variant, index)
                bad = any(entry["severity"] == "error" for entry in issues[candidate_start:])
                candidate_archives[variant] = (candidates, bad)
            except Exception as exc:
                issue("missing_or_invalid_candidates", f"{variant}: {exc}", index)
                candidate_archives[variant] = ([], True)
            initializer_start = len(issues)
            try:
                archive = _read_json(directory / variant / "initializers.json")
                initializers = archive["candidates"]
                expected_initializers, seen = [], set()
                for spec in _declared_specs(case, plan["configuration"], variant):
                    penalty = spec["init_penalty"]
                    if penalty in plan["configuration"]["init_penalties"] and penalty not in seen:
                        seen.add(penalty)
                        expected_initializers.append(spec)
                if archive.get("complete") is not True:
                    issue("incomplete_initializers", variant, index)
                if [candidate.get("index") for candidate in initializers] != list(range(len(initializers))) or \
                        [candidate.get("spec") for candidate in initializers] != expected_initializers:
                    issue("initializer_grid_mismatch", variant, index)
                for candidate in initializers:
                    if not isinstance(candidate.get("eligible"), bool):
                        issue("missing_initializer_outcome", variant, index)
                    if candidate.get("eligible"):
                        C = _arrays(directory / variant / f"initializer-{candidate['index']:04d}.npz")["coefficient"]
                        if C.shape != (case["p"], case["q"]) or not np.isfinite(C).all():
                            issue("invalid_initializer_coefficient", variant, index)
                        score = float(np.mean((data["Y_validation"] - data["X_validation"] @ C)**2))
                        if not _close(score, candidate.get("validation_mse")):
                            issue("initializer_validation_mismatch", variant, index)
                actual_files = {path.name for path in (directory / variant).glob("initializer-*.npz")}
                expected_files = {f"initializer-{candidate['index']:04d}.npz" for candidate in initializers if candidate.get("eligible") is True}
                if actual_files != expected_files:
                    issue("initializer_coefficient_inventory_mismatch", variant, index)
                initializer_archives[variant] = (initializers, any(entry["severity"] == "error" for entry in issues[initializer_start:]))
            except Exception as exc:
                issue("missing_or_invalid_initializers", f"{variant}: {exc}", index)
                initializer_archives[variant] = ([], True)
        for method in expected + unexpected:
            row = dict(base, method=method, variant=_method_variant(method),
                       fit_data_fingerprint=result.get("fit_data_fingerprint"),
                       truth_fingerprint=result.get("truth_fingerprint"))
            if method not in methods:
                issue("missing_method", "No explicit success or failure record", index, method)
                rows.append(dict(row, status="missing_method"))
                continue
            recorded = methods[method]
            row.update(total_seconds=recorded.get("total_seconds"),
                n_candidates=recorded.get("n_candidates"), n_eligible=recorded.get("n_eligible"),
                selected_iteration=recorded.get("selected_iteration"),
                candidate_index=recorded.get("candidate_index", recorded.get("selected_index")), selected_method=recorded.get("selected_method"),
                error=recorded.get("error", ""))
            method_start = len(issues)
            park_validation_mse = None
            if method == PARK_EXTERNAL:
                try:
                    findings, park_validation_mse = _park_external_checks(recorded, plan.get("configuration", {}), data or {}, case)
                    for code, message, severity in findings:
                        issue(code, message, index, method, severity)
                except Exception as exc:
                    issue("invalid_park_external_audit_record", str(exc), index, method)
            benchmark_candidates = None
            if method in BASE_METHODS and "benchmark_ridges" in plan.get("configuration", {}):
                legacy_failure = recorded.get("success") is False and not recorded.get("audit")
                if legacy_failure:
                    issue("benchmark_failure_without_candidate_trace",
                          "Whole-method failure is retained; individual failed tuning candidates cannot be audited from this legacy record",
                          index, method, severity="warning")
                benchmark_candidates = recorded.get("audit", [])
                expected_grid = _benchmark_grid(method, case, plan["configuration"])
                if not legacy_failure and expected_grid is not None and [candidate.get("candidate") for candidate in benchmark_candidates] != expected_grid:
                    issue("benchmark_candidate_grid_mismatch", "Incomplete or changed declared comparator candidate library", index, method)
                if [candidate.get("candidate_index") for candidate in benchmark_candidates] != list(range(len(benchmark_candidates))):
                    issue("benchmark_candidate_index_mismatch", "Comparator candidate indices are not complete and sequential", index, method)
                if any(candidate.get("status") not in ("ok", "failed", "uncertified") for candidate in benchmark_candidates):
                    issue("missing_benchmark_candidate_outcome", "Comparator candidate has no explicit recognized outcome", index, method)
            if recorded.get("success") is not True:
                if recorded.get("success") is not False:
                    issue("missing_success_flag", "Outcome has no explicit boolean success flag", index, method)
                if not recorded.get("error"):
                    issue("unexplained_failure", "Failed method has no recorded error", index, method)
                invalid_failure = task_invalid or any(entry["severity"] == "error" for entry in issues[method_start:])
                if method.startswith("v2"):
                    candidate_list, bad = candidate_archives.get(row["variant"], ([], True))
                    eligible = [candidate for candidate in candidate_list if candidate.get("eligible") is True and
                        (not method.startswith("v2_transfer_only") or candidate.get("method") != "target_rrr")]
                    if eligible:
                        issue("failure_despite_eligible_candidates", "Method marked failed despite an eligible recorded candidate", index, method)
                        invalid_failure = True
                    invalid_failure = invalid_failure or bad
                elif method.startswith("initializer_only"):
                    candidate_list, bad = initializer_archives.get(row["variant"], ([], True))
                    if any(candidate.get("eligible") is True for candidate in candidate_list):
                        issue("failure_despite_eligible_initializers", "Initializer marked failed despite an eligible recorded candidate", index, method)
                        invalid_failure = True
                    invalid_failure = invalid_failure or bad
                elif benchmark_candidates is not None and any(candidate.get("status") == "ok" for candidate in benchmark_candidates):
                    issue("failure_despite_eligible_benchmark", "Comparator marked failed despite a successful candidate", index, method)
                    invalid_failure = True
                rows.append(dict(row, status="audit_invalid" if invalid_failure else "failed"))
                continue
            try:
                coefficient = coefficients[method]
                if coefficient.shape != (case["p"], case["q"]) or not np.isfinite(coefficient).all():
                    raise ValueError("Coefficient has an incorrect shape or nonfinite entries")
                intercept = None
                if "intercept" in recorded:
                    intercept = coefficients[method+"__intercept"]
                    if intercept.shape != (case["q"],) or not np.isfinite(intercept).all():
                        raise ValueError("Intercept has incorrect shape or nonfinite entries")
                    if not np.array_equal(intercept, np.asarray(recorded["intercept"])):
                        issue("intercept_record_mismatch", "Saved intercept differs from method record", index, method)
                recomputed = _recompute(coefficient, data, truth, intercept)
                row.update(recomputed)
                if method == PARK_EXTERNAL and not _close(recomputed["validation_mse"], park_validation_mse):
                    issue("park_winning_score_mismatch", "Saved published-method coefficient/intercept does not reproduce the selected external validation score", index, method)
                for metric, value in recomputed.items():
                    if not _close(value, recorded.get("metrics", {}).get(metric)):
                        issue("metric_mismatch", f"{metric}: recomputed={value}, recorded={recorded.get('metrics', {}).get(metric)}", index, method)
                if row["total_seconds"] is None or not np.isfinite(row["total_seconds"]) or row["total_seconds"] < 0:
                    issue("invalid_runtime", "Missing/nonfinite/negative complete tuning runtime", index, method)
                if method.startswith("v2"):
                    variant = row["variant"]
                    candidates, bad = candidate_archives.get(variant, ([], True))
                    if bad:
                        issue("invalid_candidate_archive", variant, index, method)
                    eligible = [candidate for candidate in candidates if candidate.get("eligible") is True and
                                (not method.startswith("v2_transfer_only") or candidate.get("method") != "target_rrr")]
                    chosen = next((candidate for candidate in eligible if candidate.get("index") == recorded.get("candidate_index")), None)
                    if chosen is None:
                        issue("ineligible_selected_candidate", "Winner index is absent from its eligible library", index, method)
                    else:
                        best = min(candidate["validation_mse"] for candidate in eligible)
                        if not _close(recomputed["validation_mse"], chosen["validation_mse"]):
                            issue("winning_score_mismatch", "Saved coefficient does not reproduce winning candidate score", index, method)
                        if not _close(chosen["validation_mse"], best):
                            issue("nonminimal_validation_selection", f"Winner={chosen['validation_mse']}, library minimum={best}", index, method)
                        if recorded.get("selected_method") != chosen.get("method") or recorded.get("parameters") != chosen.get("spec"):
                            issue("winning_candidate_identity_mismatch", "Saved winner method/spec differ from candidate record", index, method)
                        if recorded.get("selected_iteration") != chosen.get("selected_iteration"):
                            issue("winning_iteration_mismatch", "Saved winner iteration differs from candidate record", index, method)
                    if recorded.get("n_candidates") != len(candidates) or recorded.get("n_eligible") != sum(c.get("eligible") is True for c in candidates):
                        issue("winner_candidate_count_mismatch", "Winner summary does not match complete candidate archive", index, method)
                elif method.startswith("initializer_only"):
                    candidates, bad = initializer_archives.get(row["variant"], ([], True))
                    if bad:
                        issue("invalid_initializer_archive", row["variant"], index, method)
                    eligible = [candidate for candidate in candidates if candidate.get("eligible") is True]
                    chosen = next((candidate for candidate in eligible if candidate.get("index") == recorded.get("candidate_index")), None)
                    if chosen is None:
                        issue("ineligible_selected_initializer", "Initializer winner is absent from eligible library", index, method)
                    else:
                        if not _close(chosen["validation_mse"], min(candidate["validation_mse"] for candidate in eligible)):
                            issue("nonminimal_initializer_selection", "Initializer does not minimize validation loss over its independent library", index, method)
                        if not _close(recomputed["validation_mse"], chosen["validation_mse"]):
                            issue("winning_initializer_score_mismatch", "Saved initializer does not reproduce its candidate score", index, method)
                        if recorded.get("parameters") != chosen.get("spec") or recorded.get("selected_iteration") != 0:
                            issue("winning_initializer_identity_mismatch", "Saved initializer spec/iteration mismatch", index, method)
                elif benchmark_candidates is not None:
                    eligible = [candidate for candidate in benchmark_candidates if candidate.get("status") == "ok"]
                    if "selected_index" in recorded:
                        chosen = next((candidate for candidate in eligible if candidate["candidate_index"] == recorded["selected_index"]), None)
                    else:
                        # The first frozen pilot did not persist this redundant index.
                        # Its saved coefficient must still attain the minimum score.
                        chosen = min(eligible, key=lambda candidate: candidate["validation_loss"]) if eligible else None
                    if chosen is None:
                        issue("ineligible_selected_benchmark", "Comparator winner index is not eligible", index, method)
                    else:
                        best = min(candidate["validation_loss"] for candidate in eligible)
                        if not _close(chosen["validation_loss"], best):
                            issue("nonminimal_benchmark_selection", "Comparator winner does not minimize validation loss", index, method)
                        # Comparator objective averages SSE/(2*n), while the paper metric is SSE/(n*q).
                        if not _close(recomputed["validation_mse"], 2*chosen["validation_loss"]/case["q"]):
                            issue("winning_benchmark_score_mismatch", "Comparator coefficient does not reproduce selected validation loss", index, method)
            except Exception as exc:
                issue("invalid_coefficient_or_metrics", str(exc), index, method)
            invalid = task_invalid or method in unexpected or any(entry["severity"] == "error" for entry in issues[method_start:])
            rows.append(dict(row, status="audit_invalid" if invalid else "success"))
        if coefficients is not None:
            successful = {name for name, record in methods.items() if record.get("success") is True}
            successful.update(name+"__intercept" for name, record in methods.items()
                              if record.get("success") is True and "intercept" in record)
            if set(coefficients) != successful:
                issue("coefficient_inventory_mismatch", "Coefficient archive keys differ from successful method names", index)
        report["validated_successes"] = sum(row["task_index"] == index and row["status"] == "success" for row in rows)
        if any(entry["severity"] == "error" for entry in issues[start_issues:]):
            report["state"] = "invalid" if result.get("complete") else "incomplete"
    return _assemble(root, plan, plan_hash, issues, task_reports, rows,
                     write=write, task_indices=task_indices)


def _assemble(root, plan, plan_hash, issues, task_reports, rows, *, write=True, task_indices=None):
    """Build final summaries; called for a full serial audit or verified merge."""
    cases, tasks = plan["cases"], plan["tasks"]
    subset = task_indices is not None
    if subset and write:
        raise ValueError("Task-subset audit artifacts must not overwrite campaign outputs")
    aggregate, paired = ([], []) if subset else _summaries(rows)
    errors = [entry for entry in issues if entry["severity"] == "error"]
    all_complete = not subset and len(task_reports) == len(tasks) and all(report["state"] == "complete" for report in task_reports)
    passed = not subset and not errors and all_complete
    audit_report = dict(schema=1, root=str(root), plan_sha256=plan_hash,
        audit_passed=passed, all_planned_tasks_complete=all_complete,
        n_planned_tasks=len(tasks), n_validated_complete_tasks=sum(report["state"] == "complete" for report in task_reports),
        n_errors=len(errors), n_warnings=len(issues)-len(errors), issues=issues, tasks=task_reports,
        audit_scope=["plan and task identity", "typed-array fingerprints", "common random numbers across stress settings",
                     "finite selected coefficients", "independently recomputed selected metrics",
                     "complete declared v2 and custom comparator candidate inventories",
                     "all eligible v2 and initializer validation scores recomputed from coefficient snapshots",
                     "minimum eligible v2, initializer and custom comparator validation losses",
                     "published competitor intercept included in prediction risk",
                     "Park external-validation grid, stopping eligibility, minimum loss, selected coefficient score"],
        limitations=["Benchmark selection uses each comparator's persisted tuning diagnostics; per-candidate benchmark coefficients are not saved.",
                     "A legacy benchmark record without selected_index is verified by its coefficient attaining the minimum eligible validation loss.",
                     "Published Park internal cross-validation is not reconstructed; the external-validation variant's full candidate trace and winner score are checked, but unselected Park coefficients are not saved.",
                     "Recorded solver success is not a theorem or global optimality certificate.",
                     "Successful-only Monte Carlo means describe available fits; failures and unavailable pairs retain explicit denominators."])
    summary = dict(schema=1, root=str(root), plan_sha256=plan_hash,
        audit_passed=passed, all_planned_tasks_complete=all_complete,
        n_cases=len(cases), n_planned_tasks=len(tasks),
        n_method_rows=len(rows), status_counts={status:sum(row["status"] == status for row in rows)
            for status in ("success", "failed", "missing_task", "incomplete_task", "missing_method", "audit_invalid")},
        metric_conventions=dict(population_prediction_excess="[tr((C_hat-C_star)' Sigma_x (C_hat-C_star))+||intercept||^2]/q; zero-mean design and response noise excluded",
            coefficient_rmse="Frobenius norm divided by sqrt(p*q)",
            paired_difference="method minus reference; positive means harm for loss and additional time for runtime",
            mean_ci="pointwise normal Monte Carlo 95% interval; unavailable for fewer than 2 successful replications",
            harm_ci="pointwise Wilson 95% interval; harm means paired difference > 1e-12",
            runtime="recorded candidate-library call time; separately recorded source preparation is added by evidence_report for the two observed-source-subspace baselines. Shared v2 library elapsed time is repeated across its reported variants, not additive"),
        artifacts=dict(audit="audit.json", replications="per_replication.csv", aggregate="aggregate.csv", paired="paired.csv"),
        aggregation=aggregate, paired_comparisons=paired)
    if subset:
        audit_report.update(coverage_scope="task_subset", audited_task_indices=task_indices,
                            n_audited_tasks=len(task_reports), subset_valid=not errors and
                            len(task_reports) == len(task_indices) and
                            all(report["state"] == "complete" for report in task_reports))
        summary.update(coverage_scope="task_subset", audited_task_indices=task_indices)
    if write:
        _atomic_json(root / "audit.json", audit_report)
        _atomic_json(root / "summary.json", summary)
        row_fields = ["task_index", "case_id", "family", "level", "seed", "n_train", "p", "q",
                      "source_rank", "target_rank", "method", "variant", "status", *METRICS,
                      "n_candidates", "n_eligible", "selected_iteration", "candidate_index", "selected_method",
                      "error", "fit_data_fingerprint", "truth_fingerprint"]
        _csv(root / "per_replication.csv", rows, row_fields)
        aggregate_fields = ["case_id", "family", "level", "n_train", "p", "q", "method", "variant", "metric",
                            "n_planned", "n_success", "n_failure", "n_missing", "n_invalid", "n", "mean", "mcse", "ci95_low", "ci95_high"]
        _csv(root / "aggregate.csv", aggregate, aggregate_fields)
        paired_fields = ["case_id", "family", "level", "n_train", "p", "q", "method", "variant", "reference", "metric",
                         "n_planned", "n_method_success", "n_reference_success", "n_paired", "n_unavailable_pairs",
                         "n", "mean", "mcse", "ci95_low", "ci95_high", "harm_count", "harm_frequency",
                         "harm_wilson95_low", "harm_wilson95_high", "mean_positive_excess", "mean_harm_conditional", "maximum_harm"]
        _csv(root / "paired.csv", paired, paired_fields)
    return dict(audit=audit_report, summary=summary, per_replication=rows,
                aggregate=aggregate, paired=paired)


def _method_variant(method):
    for variant in ("active_caps", "cap_control"):
        if method.endswith("_" + variant):
            return variant
    if method.startswith("v2") or method.startswith("initializer_only"):
        return "full_caps"
    return "benchmark"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--allow-tiny-local-tests", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, allow_tiny_local=args.allow_tiny_local_tests)
    report = result["audit"]
    print(json.dumps({key: report[key] for key in ("audit_passed", "all_planned_tasks_complete",
                     "n_planned_tasks", "n_validated_complete_tasks", "n_errors", "n_warnings")}, sort_keys=True))
    raise SystemExit(0 if report["audit_passed"] else 2)


if __name__ == "__main__":
    main()
