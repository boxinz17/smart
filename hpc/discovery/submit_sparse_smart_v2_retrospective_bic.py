#!/usr/bin/env python3
"""Submit two retrospective BIC comparisons in one shared read-only pass.

Preparation and a 12-case preflight precede a GNU Parallel allocation. A
separate summary job reports partial coverage even if the allocation fails.
Submission intents are durable; rerunning never duplicates an uncertain job.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("_retro_submission_core",
    Path(__file__).with_name("submit_sparse_smart_v2_campaign.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def validate(root):
    plan = core.read(root / "plan.json")
    core.require(plan.get("method") == "SparseSMARTv2RetrospectiveBIC" and
                 plan.get("schema_version") == 1 and plan.get("root") == str(root), "Invalid retrospective plan")
    digest = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "plan_fingerprint"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    core.require(plan.get("plan_fingerprint") == digest, "Plan fingerprint mismatch")
    core.require(plan.get("n_cases") == len(plan["cases"]) > 0, "Invalid case count")
    ids = [c["case_id"] for c in plan["cases"]]
    core.require(len(ids) == len(set(ids)), "Duplicate cases")
    core.require((root / "work-items.tsv").read_text().splitlines() == ids, "Work table mismatch")
    core.require(core.sha(root / "source-manifest.json") == plan["source_manifest_sha256"], "Manifest hash mismatch")
    manifest = core.read(root / "source-manifest.json")
    source = root / "source"
    core.require(manifest.get("source_root") == str(source) and bool(manifest.get("files")), "Invalid source manifest")
    for name, expected in manifest["files"].items():
        path = source / name
        core.require(not Path(name).is_absolute() and ".." not in Path(name).parts and
                     path.resolve().is_relative_to(source.resolve()) and core.sha(path) == expected,
                     f"Changed frozen source: {name}")
    return plan


def submit(root, args):
    plan = validate(root)
    identity = dict(plan_sha256=core.sha(root / "plan.json"),
                    source_manifest_sha256=plan["source_manifest_sha256"],
                    work_items_sha256=core.sha(root / "work-items.tsv"), workers=args.workers,
                    account=args.account, partition=args.partition, mem=args.mem, time=args.time)
    if not args.dry_run:
        (root / "campaign/attempts").mkdir(parents=True, exist_ok=True)
    actions = []
    previous = None
    for kind in ("prepare", "pool", "summary"):
        tag = kind + "-attempt-0001"
        record_path = root / "campaign/attempts" / tag / "submission.json"
        if record_path.exists():
            saved = core.read(record_path)
            core.require(all(saved.get(k) == v for k, v in identity.items()) and saved.get("kind") == kind,
                         "Prior submission belongs to different inputs/resources")
            core.require(saved.get("status") == "submitted" and str(saved.get("job_id", "")).isdigit(),
                         "Uncertain prior submission; reconcile its stdout/stderr before retrying")
            actions.append(saved)
            previous = saved["job_id"]
            continue
        cmd = ["sbatch", "--parsable", "--job-name=sv2-retro-" + kind,
               "--account=" + args.account, "--partition=" + args.partition,
               "--cpus-per-task=1", "--export=ALL", "--chdir=" + str(root),
               "--output=" + str(record_path.parent / "slurm-%j.out"),
               "--error=" + str(record_path.parent / "slurm-%j.err")]
        if kind == "pool":
            cmd += [f"--ntasks={args.workers}", "--mem-per-cpu=" + args.mem, "--time=" + args.time]
        else:
            cmd += ["--ntasks=1", "--mem=16G", "--time=04:00:00"]
        if previous:
            cmd.append("--dependency=" + ("afterany:" if kind == "summary" else "afterok:") + previous)
            cmd.append("--kill-on-invalid-dep=yes")
        cmd += [str(root / "source/hpc/discovery/sparse_smart_v2_retrospective_bic.sbatch"),
                str(root), kind, str(args.workers)]
        record = core.submit(root, args, dict(identity, kind=kind, tag=tag, attempt=1, command=cmd))
        actions.append(record)
        previous = record.get("job_id", kind.upper() + "_JOB_ID")
    result = dict(schema_version=1, root=str(root), n_cases=plan["n_cases"],
                  workers=args.workers, dry_run=args.dry_run, actions=actions)
    core.atomic(root / ("submission-preview.json" if args.dry_run else "submission.json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--account", default="mkolar_1314")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    core.require(root.is_relative_to(Path("/scratch2")), "New output must be on /scratch2")
    core.require(args.workers > 0, "Workers must be positive")
    with (root / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(submit(root, args), indent=2))


if __name__ == "__main__":
    main()
