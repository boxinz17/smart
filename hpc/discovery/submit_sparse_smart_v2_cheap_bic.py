#!/usr/bin/env python3
"""Submit only the bounded, two-phase T=200 pilot; never touch older jobs."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

HERE = Path(__file__).resolve().parent
core = module("_cheap_submission_core", HERE / "submit_sparse_smart_v2_campaign.py")
bic = module("_cheap_bic_launcher", HERE / "submit_sparse_smart_v2_bic.py")


@contextmanager
def clean_scheduler_environment():
    """Submit independent allocations without inheriting a parent job's defaults."""
    prefixes = ("SLURM_", "SBATCH_", "SRUN_")
    inherited = {name: value for name, value in os.environ.items() if name.startswith(prefixes)}
    for name in inherited:
        del os.environ[name]
    try:
        yield
    finally:
        for name in tuple(os.environ):
            if name.startswith(prefixes):
                del os.environ[name]
        os.environ.update(inherited)


def settings(root):
    result = core.read(root / "pilot.json")
    core.require(result["accepted_update_cap"] == 200 and type(result["workers"]) is int
                 and result["workers"] == 100, "pilot requires 200 updates and exactly 100 workers")
    core.require(core.sha(root / "source-manifest.json") == result["source_manifest_sha256"], "pilot source changed")
    return result


def submit_stage(root, stage, dependencies=(), *, dry_run=False):
    config = settings(root)
    tag = stage + "-attempt-0001"
    path = root / "campaign/attempts" / tag / "submission.json"
    identity = dict(pilot_sha256=core.sha(root / "pilot.json"),
                    source_manifest_sha256=core.sha(root / "source-manifest.json"))
    if path.exists():
        previous = core.read(path)
        core.require(all(previous.get(k) == v for k, v in identity.items()), "stage submission inputs differ")
        core.require(previous.get("status") == "submitted" and previous.get("job_id"),
                     "uncertain stage submission; reconcile ledger before retrying")
        return previous
    directory = path.parent
    cmd = ["sbatch", "--parsable", "--job-name=sv2cheap-" + stage,
           "--account=" + config["account"], "--partition=" + config["partition"],
           "--ntasks=1", "--cpus-per-task=1", "--mem=16G", "--time=02:00:00", "--export=ALL",
           "--chdir=" + str(root), "--output=" + str(directory / "slurm-%j.out"),
           "--error=" + str(directory / "slurm-%j.err")]
    if dependencies:
        cmd.append("--dependency=afterany:" + ":".join(dict.fromkeys(dependencies)))
    cmd += [str(root / "source/hpc/discovery/sparse_smart_v2_cheap_bic_stage.sbatch"), str(root), stage]
    with clean_scheduler_environment():
        return core.submit(root, SimpleNamespace(dry_run=dry_run),
                           dict(identity, kind=stage, tag=tag, attempt=1, command=cmd,
                                scheduler_environment_policy="clear inherited SLURM_*, SBATCH_*, SRUN_*; use explicit allocation requests"))


def launch_phase(root, phase, *, dry_run=False):
    config = settings(root)
    phase_root = root / phase
    plan = core.read(phase_root / "plan.json")
    core.require(plan["n_groups"] == 165 and plan["n_tasks"] == 990 and plan["configuration"]["iterations"] == 200,
                 "phase exceeds authorized pilot scope")
    if phase == "assessment":
        core.require((phase_root / "assessment-gate.json").exists(), "development pair is not frozen")
    args = SimpleNamespace(workers=config["workers"], max_tasks_per_chunk=990, chunks=None, task_ids=None,
            time=config["pool_time"], mem="4G", account=config["account"], partition=config["partition"],
            prepare_only=False, no_summary=True, resume=False, dry_run=dry_run,
            prepare_time="02:00:00", prepare_mem="16G", summary_time="02:00:00", summary_mem="16G")
    (phase_root / "campaign/attempts").mkdir(parents=True, exist_ok=True)
    with (phase_root / "campaign/.submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with clean_scheduler_environment():
            result = bic.orchestrate(phase_root, args)
    pool_actions = [a for a in result["actions"] if a.get("kind") == "pool"]
    pools = [str(a["job_id"]) for a in pool_actions if a.get("job_id")]
    if not pools and not dry_run:
        # The existing orchestrator verifies terminal scheduler state, task
        # provenance, exit records and artifacts before returning this action.
        # Recover a missing follow-up only; failed/uncertain pools or stages
        # still require manual reconciliation and are never retried here.
        core.require(bool(pool_actions) and all(a.get("action") == "already_finished"
                     and a.get("n_tasks") == 0 for a in pool_actions), "no fit pool was submitted or completed")
        for action in pool_actions:
            core.require(core.latest_attempt(phase_root, f"chunk-{action['chunk_id']:04d}") is not None,
                         "completed pool has no submission ledger; reconcile before launching follow-up")
    next_stage = "develop" if phase == "development" else "assess"
    gate = submit_stage(root, next_stage, pools, dry_run=dry_run)
    output = dict(phase=phase, fits=result, followup=gate)
    core.atomic(root / f"{phase}-launch.json", output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    core.require(args.root.is_absolute() and root.is_relative_to(Path("/scratch2")), "pilot root must be on scratch2")
    (root / "campaign/attempts").mkdir(parents=True, exist_ok=True)
    with (root / "campaign/.submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = submit_stage(root, "prepare", dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
