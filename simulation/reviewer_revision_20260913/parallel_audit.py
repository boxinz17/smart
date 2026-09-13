"""Audit an unchanged study plan with seed-preserving process partitions.

Usage: python parallel_audit.py ROOT --workers 8
Production array work requires Slurm and /scratch2, as in audit.py. Workers
write no artifacts. Only the parent writes the ordinary audit outputs after
checking exact-once original-task coverage and unchanged plan identity.
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

# Each process uses one BLAS thread. Set before importing NumPy through audit.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

try:
    from . import audit as common
except ImportError:
    import audit as common


def seed_partitions(plan, workers):
    """Balance whole seed groups; indices retain their original plan meaning."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    seeds = defaultdict(list)
    for index, task in enumerate(plan["tasks"]):
        seeds[task["seed"]].append(index)
    groups = list(seeds.values())
    partitions = [[] for _ in range(min(workers, len(groups)))]
    # Stable largest-first allocation balances differing task counts per seed.
    for indices in sorted(groups, key=lambda group: (-len(group), group[0])):
        destination = min(range(len(partitions)), key=lambda slot: (len(partitions[slot]), slot))
        partitions[destination].extend(indices)
    return [sorted(indices) for indices in partitions]


def _worker(root, indices, allow_tiny_local):
    result = common.audit(root, write=False, allow_tiny_local=allow_tiny_local,
                          task_indices=indices)
    # Do not serialize redundant aggregation tables through the process pipe.
    return {"audit": result["audit"], "per_replication": result["per_replication"]}


def merge_worker_results(root, results, *, write=True):
    """Reject malformed shards; then rebuild the ordinary complete-plan schema.

    Explicit missing/invalid artifacts remain covered tasks and produce a
    nonpassing final audit. Missing/duplicate shard coverage raises ValueError
    before any existing campaign audit output can be overwritten.
    """
    root = Path(root).resolve()
    plan_bytes = (root / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    digest = hashlib.sha256(plan_bytes).hexdigest()
    expected_indices = set(range(len(plan["tasks"])))
    coverage, tasks, rows, globals_seen = [], [], [], set()
    global_issues, local_issues, seed_owners = [], [], {}
    for shard_index, result in enumerate(results):
        report = result["audit"]
        if report.get("plan_sha256") != digest or Path(report.get("root", "")).resolve() != root:
            raise ValueError("Worker plan hash or root identity does not match the original plan")
        if report.get("coverage_scope") != "task_subset" or report.get("audit_passed") is not False:
            raise ValueError("Worker must return explicitly partial, nonpassing campaign coverage")
        indices = report.get("audited_task_indices")
        if not isinstance(indices, list) or any(type(i) is not int or i not in expected_indices for i in indices):
            raise ValueError("Worker has invalid original task indices")
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate task coverage inside a worker")
        reports = report.get("tasks", [])
        reported_indices = [item.get("task_index") for item in reports]
        if Counter(reported_indices) != Counter(indices):
            raise ValueError("Worker task reports do not exactly cover its declared indices")
        if report.get("n_planned_tasks") != len(plan["tasks"]) or report.get("n_audited_tasks") != len(indices):
            raise ValueError("Worker task counts do not refer to the original plan")
        for index in indices:
            seed = plan["tasks"][index]["seed"]
            previous = seed_owners.setdefault(seed, shard_index)
            if previous != shard_index:
                raise ValueError("A seed is split across workers; cross-case pairing was not checked together")
        indexed_reports = {item["task_index"]: item for item in reports}
        for index in indices:
            task = plan["tasks"][index]
            if not 0 <= task["case_index"] < len(plan["cases"]):
                raise ValueError("Original plan has an invalid case index")
            case = plan["cases"][task["case_index"]]
            actual = indexed_reports[index]
            if (actual.get("case_id") != case["case_id"] or actual.get("seed") != task["seed"] or
                    actual.get("expected_methods") != common._expected_methods(plan, task)):
                raise ValueError("Worker task-report identity differs from the original plan")
        worker_rows = result.get("per_replication", [])
        row_counts = Counter((row.get("task_index"), row.get("method")) for row in worker_rows)
        if any(index not in indices or count != 1 for (index, _), count in row_counts.items()):
            raise ValueError("Duplicate or undeclared task/method row coverage")
        worker_issues = report.get("issues", [])
        for index in indices:
            expected = common._expected_methods(plan, plan["tasks"][index])
            if any(row_counts[index, method] != 1 for method in expected):
                raise ValueError("Worker omitted a planned method row")
            extras = [method for (task_index, method) in row_counts if task_index == index and method not in expected]
            if extras and not any(item.get("task_index") == index and item.get("code") == "unplanned_methods"
                                  and item.get("severity") == "error" for item in worker_issues):
                raise ValueError("Undeclared method rows lack an explicit audit error")
        for row in worker_rows:
            task = plan["tasks"][row["task_index"]]
            case = plan["cases"][task["case_index"]]
            if row.get("case_id") != case["case_id"] or row.get("seed") != task["seed"]:
                raise ValueError("Worker row identity differs from the original plan")
        for item in worker_issues:
            index = item.get("task_index")
            if index is None:
                key = json.dumps(item, sort_keys=True, separators=(",", ":"))
                if key not in globals_seen:
                    globals_seen.add(key)
                    global_issues.append(item)
            elif index not in indices:
                raise ValueError("Worker issue refers to a task outside its partition")
            else:
                local_issues.append(item)
        coverage.extend(indices)
        tasks.extend(reports)
        rows.extend(worker_rows)
    counts = Counter(coverage)
    if set(counts) != expected_indices or any(count != 1 for count in counts.values()):
        raise ValueError("Worker partitions require exact-once coverage of every original planned task")
    # Sorting restores serial task/method order and preserves each task's issue order.
    method_order = {index: {method: order for order, method in enumerate(common._expected_methods(plan, task))}
                    for index, task in enumerate(plan["tasks"])}
    rows.sort(key=lambda row: (row["task_index"], method_order[row["task_index"]].get(row["method"], 10**9), row["method"]))
    tasks.sort(key=lambda item: item["task_index"])
    local_issues.sort(key=lambda item: item["task_index"])
    return common._assemble(root, plan, digest, global_issues+local_issues, tasks, rows, write=write)


def parallel_audit(root, *, workers=8, write=True, allow_tiny_local=False, progress=False):
    root, plan, digest = common._load_plan(root, allow_tiny_local)
    partitions = seed_partitions(plan, workers)
    results = [None] * len(partitions)
    if partitions:
        with ProcessPoolExecutor(max_workers=len(partitions), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_worker, str(root), indices, allow_tiny_local): slot
                       for slot, indices in enumerate(partitions)}
            for future in as_completed(futures):
                slot = futures[future]
                results[slot] = future.result()
                if progress:
                    print(json.dumps(dict(completed_partitions=sum(result is not None for result in results),
                        n_partitions=len(partitions), audited_tasks=len(partitions[slot])), sort_keys=True), flush=True)
    if hashlib.sha256((root / "plan.json").read_bytes()).hexdigest() != digest:
        raise ValueError("Original plan changed while workers were auditing")
    return merge_worker_results(root, results, write=write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-tiny-local-tests", action="store_true")
    args = parser.parse_args()
    result = parallel_audit(args.root, workers=args.workers, allow_tiny_local=args.allow_tiny_local_tests, progress=True)
    report = result["audit"]
    print(json.dumps({key: report[key] for key in ("audit_passed", "all_planned_tasks_complete", "n_planned_tasks",
        "n_validated_complete_tasks", "n_errors", "n_warnings")}, sort_keys=True))
    raise SystemExit(0 if report["audit_passed"] else 2)


if __name__ == "__main__":
    main()
