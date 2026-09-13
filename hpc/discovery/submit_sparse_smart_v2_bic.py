#!/usr/bin/env python3
"""Submit frozen BIC groups to independent GNU Parallel pools, without arrays.

No scientific fitting happens here. A durable intent ledger precedes each
sbatch call; ambiguous submissions block retries pending reconciliation.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

_spec = importlib.util.spec_from_file_location("_bic_submission_core",
    Path(__file__).with_name("submit_sparse_smart_v2_campaign.py"))
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
require, read, sha, canonical = core.require, core.read, core.sha, core.canonical
atomic, immutable, now = core.atomic, core.immutable, core.now
RUN_PREFIX = Path("/scratch2")


def load_inputs(root):
    plan, manifest = read(root / "plan.json"), read(root / "source-manifest.json")
    expected = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "plan_fingerprint"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(plan.get("schema_version") == 1 and plan.get("method") == "SparseSMARTv2BIC", "Unsupported BIC plan")
    require(plan.get("plan_fingerprint") == expected and Path(plan["root"]).resolve() == root,
            "Frozen plan identity mismatch")
    require(plan["source_manifest_sha256"] == sha(root / "source-manifest.json"), "Source manifest identity mismatch")
    require(plan["n_groups"] == len(plan["groups"]) and plan["n_tasks"] == len(plan["tasks"]) and plan["tasks"],
            "Plan counts invalid")
    require((root / "work-items.tsv").read_text().splitlines() == [str(i) for i in range(plan["n_tasks"])],
            "Full work items must be consecutive canonical task IDs")
    source = (root / "source").resolve()
    require(manifest.get("schema_version") == 1 and Path(manifest["source_root"]).resolve() == source,
            "Manifest identifies a different source tree")
    require(isinstance(manifest.get("files"), dict) and manifest["files"], "Empty source manifest")
    for name, expected in manifest["files"].items():
        part, path = Path(name), (source / name).resolve()
        require(not part.is_absolute() and ".." not in part.parts and path.is_relative_to(source)
                and sha(path) == expected, f"Frozen source changed: {name}")
    for name in ("sparse_smart_v2_bic_prepare.sbatch", "sparse_smart_v2_bic_pool.sbatch",
                 "sparse_smart_v2_bic_worker.sh", "sparse_smart_v2_bic_summary.sbatch"):
        require((source / "hpc/discovery" / name).is_file(), f"Missing campaign script: {name}")
    return plan


def chunk_plan(root, plan, maximum):
    groups = OrderedDict()
    identifiers = {group["group_id"] for group in plan["groups"]}
    require(len(identifiers) == plan["n_groups"], "Duplicate group IDs")
    for tid, task in enumerate(plan["tasks"]):
        require(task["task_id"] == tid and task["group_id"] in identifiers, "Invalid task identity")
        groups.setdefault(task["group_id"], []).append(tid)
    require(set(groups) == identifiers, "Every group must have planned tasks")
    chunks, current, members = [], [], []
    def flush():
        if not current:
            return
        index = len(chunks)
        require(index <= 9999, "Too many chunks")
        name = f"campaign-chunk-{index:04d}.tsv"
        immutable(root / name, "".join(f"{tid}\n" for tid in current).encode())
        chunks.append(dict(chunk_id=index, group_ids=list(members), task_ids=list(current),
                           n_tasks=len(current), task_file=name, task_file_sha256=sha(root / name)))
        current.clear(); members.clear()
    for gid, ids in groups.items():
        require(len(ids) <= maximum, f"Group {gid} exceeds maximum tasks per chunk")
        if len(current) + len(ids) > maximum:
            flush()
        current.extend(ids); members.append(gid)
    flush()
    result = dict(schema_version=1, root=str(root), plan_sha256=sha(root / "plan.json"),
                  source_manifest_sha256=sha(root / "source-manifest.json"),
                  n_groups=plan["n_groups"], n_tasks=plan["n_tasks"],
                  max_tasks_per_chunk=maximum, chunks=chunks)
    immutable(root / "campaign/chunks.json", canonical(result))
    return result


def ready(root, plan):
    path = root / "preparation.json"
    if not path.exists():
        return False
    marker = read(path)
    require(marker.get("status") == "complete" and marker.get("success") is True and
            marker.get("plan_sha256") == sha(root / "plan.json") and
            marker.get("source_manifest_sha256") == plan["source_manifest_sha256"] and
            marker.get("n_groups") == plan["n_groups"] and marker.get("n_tasks") == plan["n_tasks"],
            "Preparation marker differs from frozen inputs or is incomplete")
    return True


def completed_task(root, plan, tid):
    folder = root / "tasks" / f"{tid:06d}"
    if not (folder / "result.json").exists():
        return False
    result = read(folder / "result.json")
    identity = dict(task=plan["tasks"][tid], group_id=plan["tasks"][tid]["group_id"],
                    plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"])
    require(all(result.get(k) == v for k, v in identity.items()), f"Task {tid} has different provenance")
    if result.get("execution_success") is not True or result.get("status") != "complete":
        return False
    marker = read(folder / "status.json")
    require(marker.get("status") == "finished" and marker.get("result_sha256") == sha(folder / "result.json")
            and all(marker.get(k) == v for k, v in identity.items()), f"Task {tid} has invalid status")
    for name in ("process-exit-code.txt", "launcher-exit-code.txt"):
        require((folder / name).read_text().strip() == "0", f"Task {tid} has incomplete process exit records")
    require(isinstance(result.get("files"), dict), f"Task {tid} lacks artifact hashes")
    for name, expected in result["files"].items():
        part, path = Path(name), (folder / name).resolve()
        require(not part.is_absolute() and ".." not in part.parts and path.is_relative_to(folder.resolve())
                and sha(path) == expected, f"Task {tid} artifact changed: {name}")
    return True


def command(root, args, tag, kind):
    directory = root / "campaign/attempts" / tag
    cmd = ["sbatch", "--parsable", "--job-name=sv2bic-" + hashlib.sha256(str(root).encode()).hexdigest()[:8] + "-" + tag,
           "--account=" + args.account, "--partition=" + args.partition, "--cpus-per-task=1", "--export=ALL",
           "--chdir=" + str(root), "--output=" + str(directory / "slurm-%j.out"), "--error=" + str(directory / "slurm-%j.err")]
    if kind == "pool":
        cmd += ["--ntasks=" + str(args.workers), "--mem-per-cpu=" + args.mem, "--time=" + args.time]
    else:
        cmd += ["--ntasks=1", "--mem=" + getattr(args, kind + "_mem"), "--time=" + getattr(args, kind + "_time")]
    return cmd


def summary_submission(root, args, identity, pools):
    previous = core.latest_attempt(root, "summary")
    if previous is not None:
        require(all(previous.get(k) == v for k, v in identity.items()) and previous.get("kind") == "summary",
                "Prior summary belongs to different frozen inputs")
        require(previous.get("tag") == f"summary-attempt-{previous['attempt']:04d}", "Invalid summary attempt tag")
        state = core.existing_state(previous, args)
        if state in ("active", "not_queried"):
            return dict(kind="summary", action="active_or_unqueried_skip", job_id=previous["job_id"])
        require(args.resume, "Previous summary ended; inspect its output and use --resume to repeat it")
    number = core.attempt_number(previous)
    tag = f"summary-attempt-{number:04d}"
    cmd = command(root, args, tag, "summary")
    if pools:
        cmd.append("--dependency=afterany:" + ":".join(dict.fromkeys(pools)))
    cmd += [str(root / "source/hpc/discovery/sparse_smart_v2_bic_summary.sbatch"), str(root), tag]
    return core.submit(root, args, dict(identity, kind="summary", tag=tag, attempt=number, command=cmd))


def orchestrate(root, args):
    plan = load_inputs(root)
    chunks = chunk_plan(root, plan, args.max_tasks_per_chunk)
    selected = list(range(len(chunks["chunks"]))) if args.chunks is None else args.chunks
    requested_tasks = None
    if args.task_ids is not None:
        require(args.chunks is None, "Use either --chunks or --task-ids, not both")
        require(args.no_summary, "A task subset requires --no-summary")
        require(len(args.task_ids) == len(set(args.task_ids)) and
                all(0 <= tid < plan["n_tasks"] for tid in args.task_ids), "Invalid or duplicate task IDs")
        requested_tasks = set(args.task_ids)
        selected = [chunk["chunk_id"] for chunk in chunks["chunks"] if requested_tasks.intersection(chunk["task_ids"])]
    require(len(selected) == len(set(selected)) and all(0 <= x < len(chunks["chunks"]) for x in selected), "Invalid chunk IDs")
    require(args.no_summary or args.prepare_only or set(selected) == set(range(len(chunks["chunks"]))),
            "Partial chunk submission requires --no-summary; submit the full summary after all chunks are launched")
    identity = dict(plan_sha256=sha(root / "plan.json"), source_manifest_sha256=sha(root / "source-manifest.json"),
                    chunk_plan_sha256=sha(root / "campaign/chunks.json"))
    output = dict(schema_version=1, run_dir=str(root), dry_run=args.dry_run, workers=args.workers,
                  n_groups=plan["n_groups"], n_tasks=plan["n_tasks"], chunks=len(chunks["chunks"]),
                  requested_task_ids=None if requested_tasks is None else sorted(requested_tasks),
                  actions=[], recorded_utc=now())
    dependency = None
    if ready(root, plan):
        output["actions"].append(dict(kind="prepare", action="already_ready"))
    else:
        previous = core.latest_attempt(root, "prepare")
        core.validate_attempt(root, previous, identity, kind="prepare")
        state = core.existing_state(previous, args)
        if state in ("active", "not_queried"):
            dependency = previous["job_id"]
            output["actions"].append(dict(kind="prepare", action="reuse_active_or_unqueried", job_id=dependency))
        else:
            require(previous is None or args.resume, "Previous preparation ended without readiness; inspect logs and use --resume")
            number = core.attempt_number(previous); tag = f"prepare-attempt-{number:04d}"
            cmd = command(root, args, tag, "prepare")
            cmd += [str(root / "source/hpc/discovery/sparse_smart_v2_bic_prepare.sbatch"), str(root), tag]
            record = core.submit(root, args, dict(identity, kind="prepare", tag=tag, attempt=number, command=cmd))
            output["actions"].append(record)
            dependency = record.get("job_id", "PREPARE_JOB_ID")
    pools = []
    if not args.prepare_only:
        for index in sorted(selected):
            chunk = chunks["chunks"][index]
            previous = core.latest_attempt(root, f"chunk-{index:04d}")
            core.validate_attempt(root, previous, identity, kind="pool", chunk=chunk)
            state = core.existing_state(previous, args)
            requested = [tid for tid in chunk["task_ids"] if requested_tasks is None or tid in requested_tasks]
            if state in ("active", "not_queried"):
                require(set(requested).issubset(previous["task_ids"]),
                        f"Chunk {index} has an active/unqueried subset; wait for it to end before launching its remaining tasks")
                pools.append(previous["job_id"])
                output["actions"].append(dict(kind="pool", chunk_id=index, action="active_or_unqueried_skip", job_id=previous["job_id"]))
                continue
            pending = [tid for tid in requested if not completed_task(root, plan, tid)]
            if not pending:
                output["actions"].append(dict(kind="pool", chunk_id=index, action="already_finished", n_tasks=0))
                continue
            require(previous is None or args.resume, f"Chunk {index} has unfinished tasks; inspect logs and use --resume")
            number = core.attempt_number(previous); tag = f"chunk-{index:04d}-attempt-{number:04d}"
            name = "campaign-" + tag + ".tsv"
            immutable(root / name, "".join(f"{tid}\n" for tid in pending).encode())
            cmd = command(root, args, tag, "pool")
            if dependency:
                cmd.append("--dependency=afterok:" + dependency)
            cmd += [str(root / "source/hpc/discovery/sparse_smart_v2_bic_pool.sbatch"), str(root), str(args.workers), name, tag]
            record = core.submit(root, args, dict(identity, kind="pool", tag=tag, chunk_id=index, attempt=number,
                task_file=name, task_file_sha256=sha(root / name), task_ids=pending, n_tasks=len(pending),
                completed_tasks_skipped=len(requested) - len(pending), command=cmd))
            pools.append(record.get("job_id", f"POOL_{index:04d}_JOB_ID"))
            output["actions"].append(record)
        if not args.no_summary:
            output["actions"].append(summary_submission(root, args, identity, pools))
    atomic(root / ("campaign-submission-plan.json" if args.dry_run else "campaign-submission-latest.json"), output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--max-tasks-per-chunk", type=int, default=500)
    parser.add_argument("--chunks", type=int, nargs="+")
    parser.add_argument("--task-ids", type=int, nargs="+",
                        help="Run an exact canary subset from the full immutable plan; requires --no-summary")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--prepare-time", default="06:00:00")
    parser.add_argument("--prepare-mem", default="16G")
    parser.add_argument("--summary-time", default="12:00:00")
    parser.add_argument("--summary-mem", default="16G")
    parser.add_argument("--account", default="mkolar_1314")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        require(args.run_dir.is_absolute() and args.run_dir.is_dir(), "An existing absolute --run-dir is required")
        root = args.run_dir.resolve()
        require(args.dry_run or root.is_relative_to(RUN_PREFIX), "Actual submission requires a resolved /scratch2 root")
        require(args.workers > 0 and args.max_tasks_per_chunk > 0, "Workers and chunk size must be positive")
        for value in (args.time, args.prepare_time, args.summary_time):
            require(re.fullmatch(r"([0-9]+-)?[0-9]+(:[0-9]{2}){0,2}", value), "Invalid Slurm time")
        for value in (args.mem, args.prepare_mem, args.summary_mem):
            require(re.fullmatch(r"[1-9][0-9]*[KMGT]?", value), "Invalid Slurm memory")
        for value in (args.account, args.partition):
            require(re.fullmatch(r"[A-Za-z0-9_.-]+", value), "Invalid account/partition")
        (root / "campaign/attempts").mkdir(parents=True, exist_ok=True)
        with (root / "campaign/.submission.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output = orchestrate(root, args)
        print(json.dumps(output, indent=2))
        for action in output["actions"]:
            if "command" in action:
                print(shlex.join(action["command"]), file=sys.stderr)
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
