#!/usr/bin/env python3
"""Durable canary-gated serial Slurm allocations for checkpoint BIC; no arrays."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


HERE = Path(__file__).resolve().parent
core = module("_checkpoint_submission_core", HERE / "submit_sparse_smart_v2_campaign.py")
bic = module("_checkpoint_bic_pool", HERE / "submit_sparse_smart_v2_bic.py")


@contextmanager
def clean_scheduler_environment():
    prefixes = ("SLURM_", "SBATCH_", "SRUN_")
    saved = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    for key in saved:
        del os.environ[key]
    try:
        yield
    finally:
        for key in tuple(os.environ):
            if key.startswith(prefixes):
                del os.environ[key]
        os.environ.update(saved)


@contextmanager
def submission_lock(root):
    (root / "campaign/attempts").mkdir(parents=True, exist_ok=True)
    with (root / "campaign/.submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def settings(root):
    value = core.read(root / "campaign.json")
    core.require(value.get("remote_root") == str(root) and value.get("accepted_update_cap") == 200 and
                 value.get("initializer_pair") == [.003, .1] and value.get("tied_penalties") == [.001, .0025, .01],
                 "campaign root or frozen scientific settings differ")
    core.require(value.get("seed_ids") == list(range(100)) and value.get("models") == [0, 1, 2],
                 "campaign must cover all models and 100 seeds")
    for name in ("workers", "max_tasks_per_chunk"):
        core.require(type(value.get(name)) is int and value[name] > 0, f"invalid {name}")
    core.require(value["max_tasks_per_chunk"] <= 1000, "chunk size exceeds authorized maximum")
    for name in ("pool_time", "stage_time"):
        core.require(re.fullmatch(r"([0-9]+-)?[0-9]+(:[0-9]{2}){0,2}", value[name]), "invalid Slurm time")
    for name in ("mem_per_cpu", "stage_mem"):
        core.require(re.fullmatch(r"[1-9][0-9]*[KMGT]?", value[name]), "invalid memory")
    core.require(core.sha(root / "source-manifest.json") == value["source_manifest_sha256"], "source manifest changed")
    if value.get("comparison_baselines_sha256"):
        core.require(core.sha(root / "comparison-baselines.json") == value["comparison_baselines_sha256"],
                     "comparison baselines changed")
    return value


def _identity(root):
    return dict(campaign_sha256=core.sha(root / "campaign.json"),
                source_manifest_sha256=core.sha(root / "source-manifest.json"))


def _submit_once(root, record, *, dry_run=False):
    path = root / "campaign/attempts" / record["tag"] / "submission.json"
    if path.exists():
        previous = core.read(path)
        core.require(all(previous.get(k) == v for k, v in record.items()), "existing submission intent differs")
        core.require(previous.get("status") == "submitted" and re.fullmatch(r"[0-9]+", previous.get("job_id") or ""),
                     "uncertain scheduler submission; reconcile its durable record before retrying")
        return previous
    with clean_scheduler_environment():
        return core.submit(root, SimpleNamespace(dry_run=dry_run), record)


def submit_stage(root, stage, dependencies=(), *, dry_run=False):
    core.require(stage in ("prepare", "canary-audit", "summary"), "invalid stage")
    config, tag = settings(root), stage + "-attempt-0001"
    directory = root / "campaign/attempts" / tag
    cmd = ["sbatch", "--parsable", "--job-name=sv2checkpoint-" + stage,
           "--account=" + config["account"], "--partition=" + config["partition"],
           "--ntasks=1", "--cpus-per-task=1", "--mem=" + config["stage_mem"],
           "--time=" + config["stage_time"], "--export=ALL", "--chdir=" + str(root),
           "--output=" + str(directory / "slurm-%j.out"), "--error=" + str(directory / "slurm-%j.err")]
    if dependencies:
        cmd += ["--dependency=afterany:" + ":".join(dict.fromkeys(str(x) for x in dependencies))]
    cmd += [str(root / "source/hpc/discovery/sparse_smart_v2_checkpoint_bic_stage.sbatch"), str(root), stage]
    return _submit_once(root, dict(_identity(root), kind=stage, tag=tag, attempt=1, command=cmd), dry_run=dry_run)


def freeze_chunks(root, plan, maximum):
    """Put canary in chunk zero, then group-align the remaining production tasks."""
    canary = plan["canary_task_ids"]
    core.require(canary and len(canary) == len(set(canary)) and len(canary) <= maximum and
                 all(type(t) is int and 0 <= t < plan["n_tasks"] for t in canary), "invalid frozen canary task IDs")
    groups = {g["group_id"]: [] for g in plan["groups"]}
    chosen = set(canary)
    for tid, task in enumerate(plan["tasks"]):
        core.require(task["task_id"] == tid and task["group_id"] in groups, "invalid task identity")
        if tid not in chosen:
            groups[task["group_id"]].append(tid)
    chunks = []
    def append(ids, phase):
        if not ids:
            return
        index, name = len(chunks), f"campaign-chunk-{len(chunks):04d}.tsv"
        core.immutable(root / name, "".join(f"{tid}\n" for tid in ids).encode())
        chunks.append(dict(chunk_id=index, phase=phase, task_ids=list(ids), n_tasks=len(ids),
            group_ids=list(dict.fromkeys(plan["tasks"][tid]["group_id"] for tid in ids)),
            task_file=name, task_file_sha256=core.sha(root / name)))
    append(sorted(canary), "canary")
    current = []
    for ids in groups.values():
        core.require(len(ids) <= maximum, "one group exceeds chunk limit")
        if len(current) + len(ids) > maximum:
            append(current, "production")
            current = []
        current.extend(ids)
    append(current, "production")
    core.require(sorted(t for c in chunks for t in c["task_ids"]) == list(range(plan["n_tasks"])),
                 "canary/production partition does not cover the plan exactly once")
    record = dict(schema_version=1, root=str(root), plan_sha256=core.sha(root / "plan.json"),
        source_manifest_sha256=core.sha(root / "source-manifest.json"), n_groups=plan["n_groups"],
        n_tasks=plan["n_tasks"], max_tasks_per_chunk=maximum, chunks=chunks)
    core.immutable(root / "campaign/chunks.json", core.canonical(record))
    return record


def _pool(root, config, chunk, dependency, *, dry_run=False):
    tag = f"chunk-{chunk['chunk_id']:04d}-attempt-0001"
    args = SimpleNamespace(workers=config["workers"], mem=config["mem_per_cpu"], time=config["pool_time"],
                           account=config["account"], partition=config["partition"])
    cmd = bic.command(root, args, tag, "pool")
    if dependency:
        # A previous failed fit must release the allocation before another
        # independent pool starts. Final coverage determines campaign success.
        cmd += ["--dependency=afterany:" + str(dependency)]
    cmd += [str(root / "source/hpc/discovery/sparse_smart_v2_bic_pool.sbatch"), str(root),
            str(config["workers"]), chunk["task_file"], tag]
    record = dict(_identity(root), plan_sha256=core.sha(root / "plan.json"),
        chunk_plan_sha256=core.sha(root / "campaign/chunks.json"), kind="pool", tag=tag,
        attempt=1, chunk_id=chunk["chunk_id"], task_file=chunk["task_file"],
        task_file_sha256=chunk["task_file_sha256"], task_ids=chunk["task_ids"],
        n_tasks=chunk["n_tasks"], phase=chunk["phase"], command=cmd)
    return _submit_once(root, record, dry_run=dry_run)


def launch_phase(root, phase, *, dry_run=False):
    core.require(phase in ("canary", "production"), "invalid phase")
    with submission_lock(root):
        config, plan = settings(root), bic.load_inputs(root)
        core.require(bic.ready(root, plan), "preparation must complete before any fits are submitted")
        core.require(plan["configuration"]["selection_rule"] == "bic_checkpoint" and
                     plan["configuration"]["iterations"] == 200, "unexpected fitting protocol")
        chunks = freeze_chunks(root, plan, config["max_tasks_per_chunk"])
        if phase == "production":
            gate = core.read(root / "canary-audit.json")
            core.require(gate.get("success") is True and gate.get("plan_fingerprint") == plan["plan_fingerprint"] and
                         gate.get("task_ids") == plan["canary_task_ids"] and
                         gate.get("source_manifest_sha256") == plan["source_manifest_sha256"],
                         "canary numerical audit has not passed for this exact plan")
            core.require(all(bic.completed_task(root, plan, tid) for tid in plan["canary_task_ids"]),
                         "canary execution or artifacts are incomplete")
        selected = [c for c in chunks["chunks"] if c["phase"] == phase]
        core.require(bool(selected), "no tasks for requested phase")
        actions, dependency = [], None
        for chunk in selected:
            record = _pool(root, config, chunk, dependency, dry_run=dry_run)
            actions.append(record)
            dependency = record.get("job_id", f"POOL_{chunk['chunk_id']:04d}_JOB_ID")
        followup = submit_stage(root, "canary-audit" if phase == "canary" else "summary",
                                [dependency], dry_run=dry_run)
        result = dict(phase=phase, actions=actions, followup=followup, dry_run=dry_run)
        core.atomic(root / (phase + ("-dry-run.json" if dry_run else "-launch.json")), result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    core.require(args.root.is_absolute() and root.is_relative_to(Path("/scratch2")) and root != Path("/scratch2"),
                 "campaign root must be below scratch2")
    with submission_lock(root):
        result = submit_stage(root, "prepare", dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
