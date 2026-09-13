"""Audit a frozen rank study in read-only processes, then publish one full audit.

Usage: python -m reviewer_revision_20260913.parallel_rank_audit ROOT --workers 8
Production execution requires Slurm under /scratch2. Every seed's conditions
stay in one worker; no plans or simulation artifacts are rewritten. Worker
exceptions or incomplete/duplicate coverage abort publication. Saved-artifact
audit failures instead remain in the complete aggregate and failure counts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

# Each process gets one numerical thread. This must precede NumPy imports.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reviewer_revision_20260913 import rank_audit


def partition_by_seed(plan, workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    by_seed = defaultdict(list)
    for index, task in enumerate(plan["tasks"]):
        by_seed[task["seed"]].append(index)
    if not by_seed:
        raise ValueError("Cannot audit an empty rank plan")
    groups = [[] for _ in range(min(workers, len(by_seed)))]
    for indices in by_seed.values():
        group = min(range(len(groups)), key=lambda index: (len(groups[index]), index))
        groups[group].extend(indices)
    return [sorted(indices) for indices in groups]


def _worker(root, indices, plan_hash, allow_tiny_local):
    if hashlib.sha256((Path(root) / "rank-plan.json").read_bytes()).hexdigest() != plan_hash:
        raise RuntimeError("Original rank plan changed before worker execution")
    result = rank_audit.audit(root, write=False, allow_tiny_local=allow_tiny_local,
                             task_indices=indices)
    return dict(task_indices=indices, plan_sha256=plan_hash, result=result)


def merge_results(root, shards, *, plan_sha256, write=True, allow_tiny_local=False):
    """Fail closed unless workers cover every original task and method exactly once."""
    root, plan, digest = rank_audit._load_plan(root, allow_tiny_local=allow_tiny_local)
    if digest != plan_sha256:
        raise RuntimeError("Original rank plan changed during the parallel audit")
    wanted = set(range(len(plan["tasks"])))
    coverage, seed_workers = Counter(), defaultdict(set)
    tasks, rows, issues, global_seen = [], [], [], set()
    for worker_index, shard in enumerate(shards):
        if shard.get("plan_sha256") != digest:
            raise ValueError("Worker plan hash differs from original rank plan")
        indices = rank_audit._task_indices(plan, shard["task_indices"])
        result = shard["result"]
        audit = result["audit"]
        if audit.get("plan_sha256") != digest:
            raise ValueError("Worker audit plan hash differs from original rank plan")
        if (audit.get("coverage_scope") != "task_subset"
                or audit.get("audited_task_indices") != indices
                or audit.get("n_audited_tasks") != len(indices)
                or audit.get("audit_passed") is not False
                or audit.get("all_planned_tasks_complete") is not False):
            raise ValueError("Worker did not return a read-only subset audit")
        reports = audit["tasks"]
        if Counter(report.get("task_index") for report in reports) != Counter(indices):
            raise ValueError("Worker task-report coverage is missing or duplicated")
        for report in reports:
            case = plan["cases"][plan["tasks"][report["task_index"]]["case_index"]]
            if report.get("case_id") != case["case_id"]:
                raise ValueError("Worker task-report identity differs from original plan")
        expected_rows = Counter((index, method) for index in indices
                                for method in plan["tasks"][index]["expected_methods"])
        actual_rows = Counter((row.get("task_index"), row.get("method"))
                              for row in result["per_replication"])
        if actual_rows != expected_rows or any(count != 1 for count in actual_rows.values()):
            raise ValueError("Worker method-outcome coverage is missing or duplicated")
        for row in result["per_replication"]:
            task = plan["tasks"][row["task_index"]]
            case = plan["cases"][task["case_index"]]
            if row.get("seed") != task["seed"] or row.get("case_id") != case["case_id"]:
                raise ValueError("Worker method-outcome identity differs from original plan")
        for item in audit["issues"]:
            index = item.get("task_index")
            if index is None:
                key = json.dumps(item, sort_keys=True)
                if key in global_seen:
                    continue
                global_seen.add(key)
            elif index not in indices:
                raise ValueError("Worker issue refers to an unaudited task")
            issues.append(item)
        coverage.update(indices)
        for index in indices:
            seed_workers[plan["tasks"][index]["seed"]].add(worker_index)
        tasks.extend(reports)
        rows.extend(result["per_replication"])
    if set(coverage) != wanted or any(count != 1 for count in coverage.values()):
        raise ValueError("Workers must cover every original rank task exactly once")
    if any(len(workers) != 1 for workers in seed_workers.values()):
        raise ValueError("All conditions for a seed must remain in one worker")
    # Restore the exact serial order, including deterministic floating summation.
    tasks.sort(key=lambda row: row["task_index"])
    rows.sort(key=lambda row: (row["task_index"],
        plan["tasks"][row["task_index"]]["expected_methods"].index(row["method"])))
    issues.sort(key=lambda row: -1 if row.get("task_index") is None else row["task_index"])
    if hashlib.sha256((root / "rank-plan.json").read_bytes()).hexdigest() != digest:
        raise RuntimeError("Original rank plan changed before publication")
    return rank_audit._assemble(root, plan, digest, issues, tasks, rows, write=write)


def audit(root, *, workers=8, write=True, allow_tiny_local=False):
    root, plan, digest = rank_audit._load_plan(root, allow_tiny_local=allow_tiny_local)
    groups = partition_by_seed(plan, workers)
    shards = []
    with ProcessPoolExecutor(max_workers=len(groups),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_worker, str(root), indices, digest, allow_tiny_local)
                   for indices in groups]
        try:
            for future in as_completed(futures):
                shards.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return merge_results(root, shards, plan_sha256=digest, write=write,
                         allow_tiny_local=allow_tiny_local)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, workers=args.workers, write=not args.no_write)
    print(json.dumps(dict(audit_passed=result["audit"]["audit_passed"],
                         status_counts=result["summary"]["status_counts"]), sort_keys=True))
    raise SystemExit(0 if result["audit"]["audit_passed"] else 1)


if __name__ == "__main__":
    main()
