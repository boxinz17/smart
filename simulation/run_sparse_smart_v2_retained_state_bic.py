#!/usr/bin/env python3
"""Rescore every retained state of completed validation-stopped v2 fits.

This program never fits, initializes, generates data, or replays a trajectory.
All arms inherit validation-driven stopping. The all-retained arm additionally
searches validation-filtered incumbents; these are retrospective hybrids.
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
import run_sparse_smart_v2_retrospective_bic as historical
from sparse_smart_v2_retained_state_scoring import score_retained_states, state_key, RetainedStateError

METHOD = "SparseSMARTv2RetainedStateBIC"
ARMS = ("original_validation", "bic_validation_selected", "bic_validation_terminal",
        "bic_saved_current", "bic_all_retained")
HISTORICAL_ARMS = ARMS[:3]
ISSUES = {"missing", "corrupt", "execution_failure", "numerical_stagnation", "other_scientific_failure"}
LIMITATION = ("All five arms inherit validation-driven trajectory stopping. The all-retained-state "
              "arm additionally searches validation-filtered incumbent states. Neither new BIC arm "
              "establishes fully training-only tuning. No refitting or trajectory replay is performed.")


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
    legacy.require(plan.get("selection_arms") == list(ARMS), "retained-state arm declaration mismatch")
    previous = Path(plan["previous_retrospective_root"])
    legacy.require(previous.is_absolute() and previous.resolve() == previous and previous != path.parent
                   and not path.parent.is_relative_to(previous) and not previous.is_relative_to(path.parent),
                   "previous retrospective outputs must remain separate")
    legacy.check_hash(previous / "plan.json", plan["previous_plan_sha256"])
    legacy.check_hash(previous / "source-manifest.json", plan["previous_source_manifest_sha256"])
    previous_plan = historical.load_plan(previous / "plan.json")
    legacy.require(plan["cases"] == previous_plan["cases"]
                   and plan["source_campaign_root"] == previous_plan["source_campaign_root"],
                   "original case/rank/free-count/candidate identities changed")
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
    previous_root = Path(plan["previous_retrospective_root"])
    previous_plan = historical.load_plan(previous_root / "plan.json")
    previous_manifest = legacy.read(previous_root / "source-manifest.json")
    for name, expected in previous_manifest["files"].items():
        legacy.check_hash(legacy.relative(previous_root / "source", name), expected)
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
        previous_status_path = previous_root / "cases" / f"{cid}.status.json"
        previous_status = legacy.read(previous_status_path)
        legacy.require(previous_status.get("status") == "finished" and previous_status.get("success") is True
                       and previous_status.get("plan_fingerprint") == previous_plan["plan_fingerprint"]
                       and previous_status.get("case_id") == cid
                       and previous_status.get("source_manifest_sha256") == plan["previous_source_manifest_sha256"],
                       "previous endpoint comparison is incomplete or has different provenance")
        payload = dict(schema_version=1, method=METHOD, case_id=cid, case=original_case,
                       retrospective_plan_fingerprint=plan["plan_fingerprint"], source_root=str(old_root),
                       previous_retrospective_root=str(previous_root),
                       previous_case_sha256=previous_status["result_sha256"],
                       previous_status_sha256=legacy.sha(previous_status_path),
                       previous_plan_fingerprint=previous_plan["plan_fingerprint"],
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


def select_arms(candidates):
    """Keep historical endpoint ties; new arms compare training BIC then iteration."""
    if not candidates:
        return dict.fromkeys(ARMS)
    chosen = historical.select_arms(candidates)
    current, all_states = [], []
    for candidate in candidates:
        allowed = set(candidate["current_state_ids"])
        all_states.extend(candidate["states"])
        current.extend(row for row in candidate["states"] if row["state_id"] in allowed)
    legacy.require(current and all_states, "empty retained-state arm")
    # The original library has one RRR task per case. Its selected, terminal,
    # and checkpoint copies are collapsed by the per-trajectory inventory.
    # Ordinary configurations are never merged because their coefficients match.
    chosen["bic_saved_current"] = min(current, key=state_key)
    chosen["bic_all_retained"] = min(all_states, key=state_key)
    return chosen


def monotonicity(arms):
    checks = {}
    for left, right in ((ARMS[4], ARMS[1]), (ARMS[4], ARMS[2]),
                        (ARMS[4], ARMS[3]), (ARMS[3], ARMS[2])):
        a, b = arms[left]["bic"]["score"], arms[right]["bic"]["score"]
        checks[f"{left}<={right}"] = dict(left=a, right=b, passed=a <= b)
    legacy.require(all(c["passed"] for c in checks.values()),
                   "retained-state BIC minimum exceeds an included endpoint minimum")
    return dict(passed=True, identical_eligible_candidate_coverage=True, checks=checks)


def previous_result(payload, case_id):
    root = Path(payload["previous_retrospective_root"])
    path, status_path = root / "cases" / f"{case_id}.json", root / "cases" / f"{case_id}.status.json"
    legacy.check_hash(path, payload["previous_case_sha256"])
    legacy.check_hash(status_path, payload["previous_status_sha256"])
    result, status = legacy.read(path), legacy.read(status_path)
    legacy.require(result.get("success") is True and status.get("success") is True
                   and status.get("status") == "finished"
                   and status.get("result_sha256") == payload["previous_case_sha256"]
                   and result.get("case") == payload["case"]
                   and result.get("plan_fingerprint") == payload["previous_plan_fingerprint"]
                   and status.get("plan_fingerprint") == payload["previous_plan_fingerprint"]
                   and result.get("reference_subroot") == payload["source_root"],
                   "previous endpoint comparison identity mismatch")
    return result


def inventory_summary(candidates, *, complete):
    states = [(c["selected"]["task_id"], c["states"]) for c in candidates]
    by_origin = Counter()
    for _, rows in states:
        for row in rows:
            by_origin.update({origin["kind"] for origin in row["origins"]})
    counts = {}
    for target, source in (("expected_checkpoint_count", "expected_checkpoint_iterations"),
                           ("available_checkpoint_count", "available_checkpoint_iterations"),
                           ("available_incumbent_checkpoint_count", "available_incumbent_checkpoint_iterations"),
                           ("unreached_checkpoint_count", "unreached_checkpoint_iterations"),
                           ("validation_evaluation_count", "validation_evaluation_iterations"),
                           ("unsaved_validation_evaluation_count", "unsaved_validation_iterations")):
        counts[target] = sum(len(c["inventory"][source]) for c in candidates)
    for key in ("raw_saved_state_count", "distinct_current_state_count", "distinct_all_retained_state_count"):
        counts[key] = sum(c["inventory"][key] for c in candidates)
    return dict(eligible_task_ids=sorted(t for t, _ in states), **counts,
                states_total=sum(len(rows) for _, rows in states), states_by_origin=dict(by_origin),
                candidate_states_fingerprint=legacy.digest(states),
                current_coverage_complete=complete and all(c["inventory"]["current_coverage_complete"] for c in candidates),
                all_retained_coverage_complete=complete and all(c["inventory"]["all_retained_coverage_complete"] for c in candidates),
                chart_changing_candidates=sum(c.get("chart_event_count", 0) > 0 for c in candidates),
                chart_events=sum(c.get("chart_event_count", 0) for c in candidates),
                rrr_candidates=sum(c["selected"]["fit_method"] == "target_rrr" for c in candidates))


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
    started, cpu_started = time.monotonic(), time.process_time()
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
                  selection_rule="Historical endpoints: BIC then frozen task ID; new arms: BIC, actual iteration, frozen task ID, stable state ID",
                  support_tolerance=0.0, complete_execution_coverage=False, complete_tuning_coverage=False,
                  inventory={}, historical_comparison=dict(passed=False), monotonicity=dict(passed=False))
    old_root, old_plan, case = Path(payload["source_root"]), payload["plan"], payload["case"]
    counts, checked, bindings = Counter(), [], []
    try:
        previous = previous_result(payload, case_id)
        data, source, meta = audit.load_case(old_root, old_root, case, payload["prepared_case"], old_plan)
        singular_values = np.linalg.svd(data["X"], compute_uv=False)
        rank_threshold = np.finfo(float).eps * max(data["X"].shape) * singular_values[0]
        design_rank = int(np.count_nonzero(singular_values > rank_threshold))
        report["input_fingerprints"] = meta["fingerprints"]
        report["source_fingerprint"] = meta["source_fingerprint"]
        for task in payload["tasks"]:
            folder = old_root / "tasks" / f"{task['task_id']:05d}"
            try:
                row = legacy.check_task(old_root, old_plan, task, meta)
                bindings.append(dict(task_id=task["task_id"], files=task_bindings(folder, row)))
                checked.append((task, row, folder))
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
        if not report["errors"]:
            legacy.require(candidate_fingerprint == previous["candidate_inputs_fingerprint"],
                           "original artifacts differ from the existing retrospective comparison")
        if output.exists() or status_path.exists():
            legacy.require(output.exists() and status_path.exists(), "incomplete existing retained-state output")
            status, saved = legacy.read(status_path), legacy.read(output)
            legacy.check_hash(output, status.get("result_sha256"))
            for key, value in identity.items():
                legacy.require(status.get(key) == saved.get(key) == value, f"restart identity mismatch: {key}")
            legacy.require(status.get("candidate_inputs_fingerprint") == saved.get("candidate_inputs_fingerprint")
                           == candidate_fingerprint, "reference inputs changed since previous scoring")
            return saved
        old_candidates = {c["selected"]["task_id"]: c for c in previous["candidates"]}
        report["original_eligible_task_ids"] = sorted(old_candidates)
        for index, (task, row, folder) in enumerate(checked):
            if not row["eligible"]:
                continue
            try:
                result = legacy.read(folder / "result.json")
                scored = score_retained_states(folder, result, data, source, case,
                                               old_plan["configuration"], design_rank=design_rank)
                candidate = dict(scored["endpoints"], states=scored["all_states"],
                                 current_state_ids=[s["state_id"] for s in scored["current_states"]],
                                 inventory=scored["inventory"], chart_event_count=len(result.get("anchor_switches", [])))
                # Exactly reproduce historical scoring/metrics and tie behavior.
                for endpoint in ("selected", "terminal"):
                    legacy.require(candidate[endpoint] == old_candidates[task["task_id"]][endpoint],
                                   f"{endpoint} endpoint differs from historical retrospective record")
                report["candidates"].append(candidate)
            except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
                counts["eligible"] -= 1
                counts["corrupt"] += 1
                failure = dict(task_id=task["task_id"], classification="corrupt", error=str(error))
                if isinstance(error, RetainedStateError):
                    failure["inventory"] = error.inventory
                report["errors"].append(failure)
            print(json.dumps(dict(case_id=case_id, checked=index + 1, planned=len(payload["tasks"]),
                                  scored=len(report["candidates"]), errors=len(report["errors"])), sort_keys=True), flush=True)
        report["eligible_tasks"] = len(report["candidates"])
        report["complete_execution_coverage"] = not any(counts[k] for k in ("missing", "corrupt", "execution_failure"))
        report["complete_tuning_coverage"] = len(report["candidates"]) == len(payload["tasks"])
        report["inventory"] = inventory_summary(report["candidates"], complete=not report["errors"])
        if report["candidates"] and not report["errors"]:
            legacy.require(report["inventory"]["eligible_task_ids"] == report["original_eligible_task_ids"],
                           "originally eligible candidate coverage differs")
            selected = select_arms(report["candidates"])
            legacy.require(all(selected[arm] == previous["arms"][arm] for arm in HISTORICAL_ARMS),
                           "historical comparison changed despite identical eligible coverage")
            report["historical_comparison"] = dict(passed=True, previous_case_sha256=payload["previous_case_sha256"],
                                                    previous_status_sha256=payload["previous_status_sha256"],
                                                    eligible_tasks=len(old_candidates), arms=list(HISTORICAL_ARMS))
            report["monotonicity"] = monotonicity(selected)
            report["arms"] = selected
            report["success"] = True
    except (OSError, ValueError, KeyError, TypeError, StopIteration, np.linalg.LinAlgError) as error:
        if output.exists() or status_path.exists():
            raise
        report["errors"].append(dict(classification="case_provenance", error=str(error)))
    # An incomplete library is diagnostic only: never publish its partial minima.
    if not report["success"]:
        report["arms"] = dict.fromkeys(ARMS)
        report["inventory"] = inventory_summary(report["candidates"], complete=False)
    report["classification"] = "complete" if report["success"] else "incomplete"
    report["counts"] = dict(counts)
    report["elapsed_seconds"] = time.monotonic() - started
    report["process_cpu_seconds"] = time.process_time() - cpu_started
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
