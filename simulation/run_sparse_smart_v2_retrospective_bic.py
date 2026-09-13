#!/usr/bin/env python3
"""Rescore two retained endpoints of completed validation-stopped v2 fits.

This program never fits, initializes, generates data, or replays a trajectory.
Both BIC arms inherit the original validation-based stopping. The selected arm
also inherits validation-based within-trajectory selection. They are therefore
retrospective hybrids, not training-only BIC experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
for directory in (CODE / "simulation", CODE / "sparse-smart/src", CODE / "sparse-smart-v2/src"):
    sys.path.insert(0, str(directory))

import numpy as np

import audit_sparse_smart_v2_campaign as audit
import summarize_sparse_smart_v2_campaign as legacy
from sparse_smart_v2.selection import bic_score

METHOD = "SparseSMARTv2RetrospectiveBIC"
ARMS = ("original_validation", "bic_validation_selected", "bic_validation_terminal")
ISSUES = {"missing", "corrupt", "execution_failure", "numerical_stagnation", "other_scientific_failure"}
LIMITATION = ("Both BIC arms inherit validation-based stopping; the selected-state arm also "
              "inherits validation-based iteration selection. Neither arm is training-only BIC.")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_plan(path):
    path = Path(path).resolve()
    plan = legacy.read(path)
    legacy.require(plan.get("schema_version") == 1 and plan.get("method") == METHOD,
                   "unsupported retrospective plan")
    legacy.require(plan.get("root") == str(path.parent), "retrospective root identity mismatch")
    legacy.require(plan.get("plan_fingerprint") == legacy.digest(
        {k: v for k, v in plan.items() if k != "plan_fingerprint"}), "retrospective plan fingerprint mismatch")
    cases = plan["cases"]
    ids = [row["case_id"] for row in cases]
    legacy.require(len(ids) == len(set(ids)) == plan["n_cases"] > 0, "invalid retrospective case coverage")
    checked_sources = set()
    for row in cases:
        legacy.require(re.fullmatch(r"[A-Za-z0-9_-]+", row["case_id"]) is not None,
                       "unsafe case ID")
        legacy.require(row["case"]["case_id"] == row["case_id"], "case ID differs from case metadata")
        if row["source_root"] in checked_sources:
            continue
        source = Path(row["source_root"])
        legacy.require(source.is_absolute() and str(source.resolve()) == str(source), "noncanonical source root")
        if plan.get("source_campaign_root"):
            legacy.require(source.is_relative_to(Path(plan["source_campaign_root"])), "case source outside campaign")
        legacy.require(not path.parent.is_relative_to(source) and not source.is_relative_to(path.parent),
                       "retrospective outputs must be separate from reference outputs")
        checked_sources.add(row["source_root"])
    return plan


def prepare(plan_path):
    """Verify full provenance once, then write small immutable per-case inputs."""
    path = Path(plan_path).resolve()
    root, plan = path.parent, load_plan(path)
    marker = root / "preparation.json"
    if marker.exists():
        previous = legacy.read(marker)
        legacy.require(previous.get("success") is True and previous.get("status") == "complete"
                       and previous.get("plan_sha256") == legacy.sha(path),
                       "existing preparation belongs to a different or incomplete plan")
        for row in previous["cases"]:
            legacy.check_hash(root / "inputs" / f"{row['case_id']}.json", row["input_sha256"])
        return previous
    legacy.check_hash(root / "source-manifest.json", plan["source_manifest_sha256"])
    manifest = legacy.read(root / "source-manifest.json")
    legacy.require(manifest.get("schema_version") == 1 and bool(manifest.get("files")),
                   "invalid retrospective source manifest")
    for name, expected in manifest["files"].items():
        legacy.check_hash(legacy.relative(root / "source", name), expected)
    cached, rows = {}, []
    for case_row in plan["cases"]:
        old_root = Path(case_row["source_root"])
        if str(old_root) not in cached:
            old_plan, by_case, prepared_cases = legacy.load_provenance(old_root)
            cached[str(old_root)] = (old_plan, by_case, prepared_cases, legacy.sha(old_root / "plan.json"),
                                     legacy.sha(old_root / "preparation.json"))
        old_plan, by_case, prepared_cases, plan_sha, prep_sha = cached[str(old_root)]
        cid = case_row["case_id"]
        original_case = next(row for row in old_plan["cases"] if row["case_id"] == cid)
        legacy.require(case_row["case"] == original_case, "retrospective and reference cases differ")
        for key, actual in (("reference_plan_sha256", plan_sha), ("reference_preparation_sha256", prep_sha),
                            ("reference_source_manifest_sha256", old_plan["source_manifest_sha256"])):
            if key in case_row:
                legacy.require(case_row[key] == actual, f"frozen reference identity differs: {key}")
        payload = dict(schema_version=1, method=METHOD, case_id=cid, case=original_case,
                       retrospective_plan_fingerprint=plan["plan_fingerprint"], source_root=str(old_root),
                       reference_plan_sha256=plan_sha, reference_preparation_sha256=prep_sha,
                       reference_source_manifest_sha256=old_plan["source_manifest_sha256"],
                       plan={key: old_plan[key] for key in ("schema_version", "method", "configuration",
                                                           "plan_fingerprint", "source_manifest_sha256")},
                       prepared_case=prepared_cases[cid], tasks=by_case[cid])
        target = root / "inputs" / f"{cid}.json"
        if target.exists():
            legacy.require(legacy.read(target) == payload, "existing prepared case input differs")
        else:
            write_json(target, payload)
        rows.append(dict(case_id=cid, input_sha256=legacy.sha(target)))
    report = dict(schema_version=1, method=METHOD, status="complete", success=True,
                  prepared_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=legacy.sha(path),
                  plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"],
                  n_cases=len(rows), reference_roots=len(cached), cases=rows)
    write_json(marker, report)
    return report


def load_input(plan_path, case_id):
    path = Path(plan_path).resolve()
    root, plan = path.parent, load_plan(path)
    row = next(row for row in plan["cases"] if row["case_id"] == case_id)
    preparation = legacy.read(root / "preparation.json")
    legacy.require(preparation.get("success") is True and preparation.get("status") == "complete"
                   and preparation.get("plan_sha256") == legacy.sha(path)
                   and preparation.get("plan_fingerprint") == plan["plan_fingerprint"]
                   and preparation.get("source_manifest_sha256") == plan["source_manifest_sha256"],
                   "retrospective preparation identity mismatch")
    records = {record["case_id"]: record for record in preparation["cases"]}
    legacy.require(len(records) == len(preparation["cases"]) == plan["n_cases"]
                   and set(records) == {item["case_id"] for item in plan["cases"]},
                   "retrospective preparation coverage mismatch")
    input_path = root / "inputs" / f"{case_id}.json"
    expected = records[case_id]["input_sha256"]
    legacy.check_hash(input_path, expected)
    payload = legacy.read(input_path)
    legacy.require(payload.get("retrospective_plan_fingerprint") == plan["plan_fingerprint"]
                   and payload.get("case") == row["case"] and payload.get("case_id") == case_id
                   and payload.get("source_root") == row["source_root"], "prepared case identity differs")
    old_root = Path(payload["source_root"])
    for name, key in (("plan.json", "reference_plan_sha256"),
                      ("preparation.json", "reference_preparation_sha256"),
                      ("source-manifest.json", "reference_source_manifest_sha256")):
        legacy.check_hash(old_root / name, payload[key])
    return plan, payload, expected


def endpoint_state(arrays, endpoint, case, source, configuration, *, direct):
    """Reconstruct exactly the endpoint's own chart; never use terminal aliases."""
    if direct:
        legacy.require(int(np.asarray(arrays["states_schema_version"]).item()) == 3, "unsupported RRR state schema")
        coefficient = np.asarray(arrays[f"{endpoint}_coefficient"])
        legacy.require(coefficient.shape == (case["p"], case["q"]) and np.isfinite(coefficient).all(),
                       "invalid RRR coefficient")
        P, d, Q = (arrays[f"{endpoint}_{label}"] for label in ("P", "d", "Q"))
        legacy.require(d.ndim == 1 and len(d) <= case["rank"]
                       and P.shape == (case["p"], len(d)) and Q.shape == (case["q"], len(d))
                       and np.all(d >= 0) and np.all(np.diff(d) <= 0), "invalid RRR factor dimensions or spectrum")
        audit.close(P.T @ P, np.eye(len(d)), f"{endpoint} RRR left orthogonality")
        audit.close(Q.T @ Q, np.eye(len(d)), f"{endpoint} RRR right orthogonality")
        audit.close((P * d) @ Q.T, coefficient, f"{endpoint} RRR stored factor reconstruction")
        return coefficient, dict(direct_rrr=True)
    legacy.require(int(np.asarray(arrays["states_schema_version"]).item()) == 2, "unsupported chart state schema")
    chart = audit.saved_chart(arrays, endpoint, case)
    state = arrays[f"{endpoint}_state"]
    legacy.require(state.ndim == 1 and np.isfinite(state).all(), "invalid retained chart state")
    domain = {key: configuration["margins"][key] for key in ("d_lower", "d_upper", "gap", "anchor_min")}
    legacy.require(chart.domain_reason(state, **domain) is None, "retained endpoint violates its fixed domain")
    P, d, Q = chart.reconstruct(state)
    for label, value in (("P", P), ("d", d), ("Q", Q)):
        audit.close(value, arrays[f"{endpoint}_{label}"], f"{endpoint} stored {label}")
    for label, factor in (("u", P), ("v", Q)):
        audit.close(factor.T @ factor, np.eye(case["rank"]), f"{endpoint} {label} orthogonality")
    physical_rows = [arrays[f"free_rows_{side}"] for side in ("u", "v")]
    for rows, count, dimension in zip(physical_rows, case["free_directions"], (case["p"], case["q"])):
        legacy.require(rows.ndim == 1 and rows.dtype.kind in "iu" and len(rows) == count
                       and len(np.unique(rows)) == count and np.all((rows >= 0) & (rows < dimension)),
                       "invalid fixed free-row identities")
    free = audit.fixed_free_rows(chart, physical_rows)
    weighted_u, weighted_v = chart.unpack(state)[-2:]
    for side, block, factor, cap in zip(("u", "v"), (weighted_u, weighted_v), (P, Q), case["support_limits"]):
        mask = getattr(free, f"penalized_{side}")
        legacy.require(np.count_nonzero(block[mask]) <= cap, "retained endpoint exceeds hard support cap")
        outside = np.setdiff1d(np.arange(len(factor)), getattr(free, f"rows_{side}"))
        audit.close(block[mask], (factor[outside] * d).ravel(), f"{endpoint} physical masked coordinates {side}")
    coefficient = ((source["left"] @ P) * d) @ (source["right"] @ Q).T
    return coefficient, dict(free_directions=case["free_directions"], weighted_u=weighted_u,
                             weighted_v=weighted_v, penalized_u=free.penalized_u,
                             penalized_v=free.penalized_v)


def score_endpoints(folder, result, data, source, case, configuration, *, design_rank=None):
    direct = result.get("fit_method") == "target_rrr"
    with np.load(folder / "states.npz", allow_pickle=False) as archive:
        # Deliberately skip checkpoint arrays: these two comparisons do not select
        # over checkpoints or reconstruct unsaved accepted iterations.
        arrays = {name: archive[name] for name in archive.files
                  if name.startswith(("selected_", "terminal_", "free_rows_")) or name == "states_schema_version"}
    for name, array in arrays.items():
        legacy.require(array.dtype.kind in "biuf" and np.isfinite(array).all(), f"invalid saved endpoint array: {name}")
    endpoints = {}
    for endpoint in ("selected", "terminal"):
        coefficient, score_args = endpoint_state(arrays, endpoint, case, source, configuration, direct=direct)
        if direct and design_rank is not None:
            score_args["design_rank"] = design_rank
        # BIC receives observed training data only; truth and validation enter
        # the reporting audit below, after scoring, never the BIC comparison.
        score = bic_score(data["X"], data["Y"], coefficient, rank=case["rank"], **score_args).as_dict()
        metrics = audit._metrics(coefficient, data)
        for key, value in metrics.items():
            audit.close(value, result[f"{endpoint}_metrics"][key], f"{endpoint} saved {key}")
        # Preserve the literal historical validation comparison (including exact
        # ties), after independently verifying all its reported metric values.
        metrics = {key: result[f"{endpoint}_metrics"][key] for key in metrics}
        endpoints[endpoint] = dict(task_id=result["task"]["task_id"], task=result["task"], endpoint=endpoint,
                                   iteration=result["selected_iteration"] if endpoint == "selected" else result["n_iter"],
                                   fit_method=result.get("fit_method", "sparse_smart_v2"),
                                   termination_reason=result.get("termination_reason"),
                                   optimization_converged=result.get("optimization_converged") is True,
                                   bic=score, metrics=metrics, states_sha256=result["files"]["states.npz"],
                                   result_sha256=legacy.sha(folder / "result.json"))
    if direct:
        audit.close(arrays["selected_coefficient"], arrays["terminal_coefficient"], "RRR endpoints differ")
    return endpoints


def select_arms(candidates):
    """Exact score ties use frozen task ID; no truth-based tie-breaking."""
    if not candidates:
        return dict.fromkeys(ARMS)
    selected = [row["selected"] for row in candidates]
    terminal = [row["terminal"] for row in candidates]
    return dict(original_validation=min(selected, key=lambda row: (row["metrics"]["validation_mse"], row["task_id"])),
                bic_validation_selected=min(selected, key=lambda row: (row["bic"]["score"], row["task_id"])),
                bic_validation_terminal=min(terminal, key=lambda row: (row["bic"]["score"], row["task_id"])))


def task_bindings(folder, row):
    """Hashes are read after the legacy checker has verified the artifacts."""
    names = ["result.json", "status.json", "process-exit-code.txt", "process-status.tsv",
             "launcher-exit-code.txt", "launcher-status.tsv"]
    binding = {name: legacy.sha(folder / name) for name in names}
    # check_task already streamed and verified these potentially large files.
    # Reuse the verified expected hashes instead of reading their bytes twice.
    binding.update(legacy.read(folder / "result.json")["files"])
    return binding


def run_case(plan_path, case_id):
    root = Path(plan_path).resolve().parent
    (root / "cases").mkdir(parents=True, exist_ok=True)
    # An accidental duplicate worker must not race an existing worker's output.
    with (root / "cases" / f"{case_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_case_locked(plan_path, case_id)


def _run_case_locked(plan_path, case_id):
    started = time.monotonic()
    root = Path(plan_path).resolve().parent
    plan, payload, input_sha = load_input(plan_path, case_id)
    output, status_path = root / "cases" / f"{case_id}.json", root / "cases" / f"{case_id}.status.json"
    identity = dict(schema_version=1, method=METHOD, case_id=case_id, plan_fingerprint=plan["plan_fingerprint"],
                    source_manifest_sha256=plan["source_manifest_sha256"], input_sha256=input_sha)
    report = dict(identity, case=payload["case"], reference_subroot=payload["source_root"],
                  reference_plan_sha256=payload["reference_plan_sha256"],
                  reference_preparation_sha256=payload["reference_preparation_sha256"],
                  reference_source_manifest_sha256=payload["reference_source_manifest_sha256"],
                  started_utc=datetime.now(timezone.utc).isoformat(), success=False, errors=[], counts={},
                  planned_tasks=len(payload["tasks"]), eligible_tasks=0, candidates=[],
                  arms=dict.fromkeys(ARMS), limitation=LIMITATION, fitting_performed=False,
                  bic_uses_validation_or_truth=False, validation_used_for_original_stopping=True,
                  selection_rule="minimum score; exact ties use smallest frozen task ID", support_tolerance=0.0,
                  complete_execution_coverage=False, complete_tuning_coverage=False)
    old_root, old_plan, case = Path(payload["source_root"]), payload["plan"], payload["case"]
    counts, checked, bindings = Counter(), [], []
    try:
        data, source, meta = audit.load_case(old_root, old_root, case, payload["prepared_case"], old_plan)
        # The explicit RRR endpoint appears twice. Reuse its certified design
        # dimension rather than decomposing the same training matrix again.
        singular_values = np.linalg.svd(data["X"], compute_uv=False)
        rank_threshold = np.finfo(float).eps * max(data["X"].shape) * singular_values[0]
        design_rank = int(np.count_nonzero(singular_values > rank_threshold))
        report["input_fingerprints"] = meta["fingerprints"]
        report["source_fingerprint"] = meta["source_fingerprint"]
        for task in payload["tasks"]:
            folder = old_root / "tasks" / f"{task['task_id']:05d}"
            try:
                row = legacy.check_task(old_root, old_plan, task, meta)
                binding = dict(task_id=task["task_id"], files=task_bindings(folder, row))
                checked.append((task, row, folder))
                bindings.append(binding)
            except FileNotFoundError as error:
                row = dict(task, classification="missing", eligible=False, error=str(error))
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                row = dict(task, classification="corrupt", eligible=False, error=str(error))
            counts[row["classification"]] += 1
            if row["classification"] in ISSUES:
                report["errors"].append(dict(task_id=task["task_id"], classification=row["classification"],
                                             error=row.get("error", row.get("fit_status"))))
        candidate_fingerprint = legacy.digest(bindings)
        report["candidate_inputs_fingerprint"] = candidate_fingerprint
        if output.exists() or status_path.exists():
            legacy.require(output.exists() and status_path.exists(), "incomplete existing retrospective output")
            status, previous = legacy.read(status_path), legacy.read(output)
            legacy.check_hash(output, status.get("result_sha256"))
            for key, value in identity.items():
                legacy.require(status.get(key) == previous.get(key) == value, f"restart identity mismatch: {key}")
            legacy.require(status.get("candidate_inputs_fingerprint") == previous.get("candidate_inputs_fingerprint")
                           == candidate_fingerprint, "reference candidate inputs changed since previous scoring")
            if previous.get("success") is True and not report["errors"]:
                return previous
        for index, (task, row, folder) in enumerate(checked):
            if not row["eligible"]:
                continue
            try:
                candidate = score_endpoints(folder, legacy.read(folder / "result.json"), data, source, case,
                                            old_plan["configuration"], design_rank=design_rank)
                report["candidates"].append(candidate)
            except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
                counts["eligible"] -= 1
                counts["corrupt"] += 1
                report["errors"].append(dict(task_id=task["task_id"], classification="corrupt", error=str(error)))
            if (index + 1) % 25 == 0:
                print(json.dumps(dict(case_id=case_id, checked=index + 1, planned=len(payload["tasks"]))), flush=True)
        report["arms"] = select_arms(report["candidates"])
        report["eligible_tasks"] = len(report["candidates"])
        report["complete_execution_coverage"] = not any(counts[key] for key in ("missing", "corrupt", "execution_failure"))
        report["complete_tuning_coverage"] = len(report["candidates"]) == len(payload["tasks"])
        report["success"] = bool(report["candidates"]) and not report["errors"]
    except (OSError, ValueError, KeyError, TypeError, StopIteration, np.linalg.LinAlgError) as error:
        if output.exists() or status_path.exists():
            # A mismatch must never overwrite the previously bound result.
            raise
        report["errors"].append(dict(classification="case_provenance", error=str(error)))
    report["counts"] = dict(counts)
    report["elapsed_seconds"] = time.monotonic() - started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, report)
    write_json(status_path, dict(identity, status="finished", success=report["success"],
                                 candidate_inputs_fingerprint=report.get("candidate_inputs_fingerprint"),
                                 result_sha256=legacy.sha(output)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--prepare", action="store_true")
    operation.add_argument("--case-id")
    args = parser.parse_args(argv)
    result = prepare(args.plan) if args.prepare else run_case(args.plan, args.case_id)
    print(json.dumps({key: result.get(key) for key in ("success", "case_id", "n_cases", "counts", "elapsed_seconds", "errors")},
                     sort_keys=True), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
