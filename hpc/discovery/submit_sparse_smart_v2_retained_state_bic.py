#!/usr/bin/env python3
"""Manually gate retained-state rescoring after a small independent preflight."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

spec = importlib.util.spec_from_file_location("_retained_submission_core",
    Path(__file__).with_name("submit_sparse_smart_v2_campaign.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
METHOD = "SparseSMARTv2RetainedStateBIC"


@contextmanager
def clean_scheduler_environment():
    prefixes = ("SLURM_", "SBATCH_", "SRUN_")
    previous = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    for key in previous: del os.environ[key]
    try:
        yield
    finally:
        for key in tuple(os.environ):
            if key.startswith(prefixes): del os.environ[key]
        os.environ.update(previous)


def validate(root):
    plan = core.read(root / "plan.json")
    core.require(plan.get("method") == METHOD and plan.get("schema_version") == 1
                 and plan.get("root") == str(root), "invalid retained-state plan")
    core.require(plan.get("selection_arms") == ["original_validation", "bic_validation_selected",
        "bic_validation_terminal", "bic_saved_current", "bic_all_retained"], "invalid selection arms")
    encoded = json.dumps({k: v for k, v in plan.items() if k != "plan_fingerprint"},
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    core.require(plan.get("plan_fingerprint") == hashlib.sha256(encoded).hexdigest(), "plan fingerprint mismatch")
    cases = [row["case_id"] for row in plan["cases"]]
    core.require(len(cases) == len(set(cases)) == plan["n_cases"] > 0 and
                 all(re.fullmatch(r"m[0-9]+_e[0-9]+_k[0-9]+_s[0-9]+", cid) for cid in cases), "invalid case IDs/counts")
    preflight = plan["preflight_case_ids"]
    chart = plan["preflight_chart_change_case_ids"]
    core.require(preflight and len(preflight) == len(set(preflight)) and set(preflight) < set(cases), "invalid preflight coverage")
    core.require(len(chart) == len(set(chart)) and set(chart) <= set(preflight), "invalid chart-change coverage")
    core.require({row["case"]["model_id"] for row in plan["cases"] if row["case_id"] in preflight} == {0, 1, 2},
                 "preflight must cover all three models")
    for name, expected in (("work-items.tsv", cases), ("preflight-work-items.tsv", preflight),
                           ("production-work-items.tsv", [x for x in cases if x not in set(preflight)])):
        path = root / name
        core.require(not path.is_symlink() and path.read_text().splitlines() == expected, "work table mismatch: " + name)
    core.require(core.sha(root / "source-manifest.json") == plan["source_manifest_sha256"], "source manifest hash mismatch")
    manifest, source = core.read(root / "source-manifest.json"), root / "source"
    core.require(manifest.get("schema_version") == 1 and manifest.get("source_root") == str(source)
                 and isinstance(manifest.get("files"), dict) and manifest["files"], "invalid source manifest")
    for name, expected in manifest["files"].items():
        path = source / name
        core.require(not Path(name).is_absolute() and ".." not in Path(name).parts and
                     path.resolve().is_relative_to(source.resolve()) and core.sha(path) == expected,
                     "changed frozen source: " + name)
    return plan


def preparation_ready(root, plan):
    marker = core.read(root / "preparation.json")
    core.require(marker.get("success") is True and marker.get("status") == "complete" and
                 marker.get("plan_sha256") == core.sha(root / "plan.json") and
                 marker.get("plan_fingerprint") == plan["plan_fingerprint"] and
                 marker.get("source_manifest_sha256") == plan["source_manifest_sha256"] and
                 marker.get("n_cases") == plan["n_cases"], "preparation not complete for this plan")
    core.require((root / "campaign/attempts/prepare-attempt-0001/exit-code.txt").read_text().strip() == "0" and
                 (root / "campaign/attempts/prepare-attempt-0001/source-integrity-exit-code.txt").read_text().strip() == "0",
                 "preparation stage did not finish cleanly")
    return marker


def preflight_records(root, plan):
    preparation = preparation_ready(root, plan)
    expected_inputs = {row["case_id"]: row["input_sha256"] for row in preparation["cases"]}
    core.require(len(expected_inputs) == len(preparation["cases"]) == plan["n_cases"] and
                 set(expected_inputs) == {row["case_id"] for row in plan["cases"]}, "prepared case coverage differs")
    rows = []
    for cid in plan["preflight_case_ids"]:
        path = root / "cases" / (cid + ".json")
        result = core.read(path)
        status = core.read(root / "cases" / (cid + ".status.json"))
        expected = dict(method=METHOD, schema_version=1, case_id=cid,
                        plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"])
        core.require(all(result.get(k) == status.get(k) == v for k, v in expected.items()), "preflight result identity differs")
        core.require(result.get("input_sha256") == status.get("input_sha256") == expected_inputs[cid]
                     == core.sha(root / "inputs" / (cid + ".json")),
                     "preflight prepared input differs")
        core.require(status.get("result_sha256") == core.sha(path) and status.get("status") == "finished"
                     and status.get("success") is True and result.get("success") is True, "preflight result did not pass")
        inventory = result.get("inventory", {})
        core.require(inventory.get("current_coverage_complete") is True and
                     inventory.get("all_retained_coverage_complete") is True and
                     result.get("historical_comparison", {}).get("passed") is True and
                     result.get("monotonicity", {}).get("passed") is True,
                     "preflight coverage/history/monotonicity failed")
        if cid in plan["preflight_chart_change_case_ids"]:
            core.require(type(inventory.get("chart_changing_candidates")) is int and
                         inventory["chart_changing_candidates"] > 0, "declared preflight did not exercise changing charts")
        for name in (cid + ".exit-code.txt", cid + ".process-exit-code.txt"):
            core.require((root / "logs/cases" / name).read_text().strip() == "0", "preflight process/launcher failed")
        rows.append(dict(case_id=cid, result_sha256=core.sha(path),
                         status_sha256=core.sha(root / "cases" / (cid + ".status.json")),
                         chart_changing_candidates=inventory.get("chart_changing_candidates", 0)))
    return rows


def record_preflight(root):
    plan = validate(root)
    records = preflight_records(root, plan)
    result = dict(success=True, plan_fingerprint=plan["plan_fingerprint"],
        plan_sha256=core.sha(root / "plan.json"), source_manifest_sha256=plan["source_manifest_sha256"],
        case_ids=plan["preflight_case_ids"], cases=records, checked_utc=core.now())
    core.atomic(root / "preflight.json", result)
    return result


def preflight_ready(root, plan):
    result = core.read(root / "preflight.json")
    core.require(result.get("success") is True and result.get("plan_fingerprint") == plan["plan_fingerprint"]
                 and result.get("plan_sha256") == core.sha(root / "plan.json") and
                 result.get("source_manifest_sha256") == plan["source_manifest_sha256"] and
                 result.get("case_ids") == plan["preflight_case_ids"], "preflight release belongs to different inputs")
    core.require(result.get("cases") == preflight_records(root, plan), "preflight outputs changed after verification")
    for name in ("exit-code.txt", "source-integrity-exit-code.txt"):
        core.require((root / "campaign/attempts/preflight-attempt-0001" / name).read_text().strip() == "0",
                     "preflight allocation did not finish cleanly")
    return result


def submit_one(root, args, plan, kind, dependency=None):
    tag = kind + "-attempt-0001"
    folder = root / "campaign/attempts" / tag
    work = "preflight-work-items.tsv" if kind == "preflight" else "production-work-items.tsv" if kind == "pool" else "work-items.tsv"
    identity = dict(plan_sha256=core.sha(root / "plan.json"), source_manifest_sha256=plan["source_manifest_sha256"],
        work_file=work, work_items_sha256=core.sha(root / work), workers=args.workers,
        account=args.account, partition=args.partition, mem=args.mem, time=args.time,
        kind=kind, tag=tag, attempt=1)
    cmd = ["sbatch", "--parsable", "--job-name=sv2-retained-" + kind,
        "--account=" + args.account, "--partition=" + args.partition, "--cpus-per-task=1", "--export=ALL",
        "--chdir=" + str(root), "--output=" + str(folder / "slurm-%j.out"), "--error=" + str(folder / "slurm-%j.err")]
    if kind in ("pool", "preflight"):
        cmd += ["--ntasks=" + str(args.workers), "--mem-per-cpu=" + args.mem,
                "--time=" + (args.time if kind == "pool" else "04:00:00")]
    else:
        cmd += ["--ntasks=1", "--mem=16G", "--time=04:00:00"]
    if dependency:
        cmd += ["--dependency=afterany:" + str(dependency), "--kill-on-invalid-dep=yes"]
    cmd += [str(root / "source/hpc/discovery/sparse_smart_v2_retained_state_bic.sbatch"), str(root), kind, str(args.workers)]
    identity["command"] = cmd
    path = folder / "submission.json"
    if path.exists():
        previous = core.read(path)
        core.require(all(previous.get(k) == v for k, v in identity.items()), "prior submission has different inputs/resources")
        core.require(previous.get("status") == "submitted" and re.fullmatch(r"[0-9]+", str(previous.get("job_id", ""))),
                     "uncertain prior submission; reconcile saved scheduler response before retrying")
        return previous
    with clean_scheduler_environment():
        return core.submit(root, args, identity)


def submit(root, args):
    plan = validate(root)
    if args.stage != "prepare":
        preparation_ready(root, plan)
    if args.stage in ("production", "summary"):
        preflight_ready(root, plan)
    actions = []
    if args.stage == "production":
        pool = submit_one(root, args, plan, "pool")
        actions = [pool, submit_one(root, args, plan, "summary", pool.get("job_id", "POOL_JOB_ID"))]
    elif args.stage == "summary":
        # Recover a missing summary submission without releasing another pool.
        previous = core.read(root / "campaign/attempts/pool-attempt-0001/submission.json")
        core.require(previous.get("status") == "submitted" and previous.get("plan_sha256") == core.sha(root / "plan.json")
                     and re.fullmatch(r"[0-9]+", str(previous.get("job_id", ""))), "pool submission must be reconciled first")
        actions = [submit_one(root, args, plan, "summary", previous["job_id"])]
    else:
        actions = [submit_one(root, args, plan, args.stage)]
    result = dict(schema_version=1, root=str(root), stage=args.stage, n_cases=plan["n_cases"],
        preflight_cases=len(plan["preflight_case_ids"]), workers=args.workers, dry_run=args.dry_run, actions=actions)
    core.atomic(root / (args.stage + ("-submission-preview.json" if args.dry_run else "-submission.json")), result)
    return result


def validate_submission(root, kind, workers, *, check_allocation=True):
    plan = validate(root)
    record = core.read(root / "campaign/attempts" / (kind + "-attempt-0001") / "submission.json")
    core.require(record.get("kind") == kind and record.get("workers") == workers
                 and record.get("plan_sha256") == core.sha(root / "plan.json")
                 and record.get("source_manifest_sha256") == plan["source_manifest_sha256"], "stage submission identity differs")
    expected_work = "preflight-work-items.tsv" if kind == "preflight" else "production-work-items.tsv" if kind == "pool" else "work-items.tsv"
    core.require(record.get("work_file") == expected_work and
                 record.get("work_items_sha256") == core.sha(root / expected_work), "submitted work file changed")
    if check_allocation:
        core.require(record.get("status") == "submitted" and str(record.get("job_id")) == os.environ.get("SLURM_JOB_ID") and
                     int(os.environ.get("SLURM_NTASKS", "0")) == (workers if kind in ("pool", "preflight") else 1) and
                     int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) == 1, "unexpected Slurm allocation")
    return dict(success=True, plan_sha256=record["plan_sha256"], source_manifest_sha256=record["source_manifest_sha256"],
                work_file=expected_work, work_items_sha256=record["work_items_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "preflight", "production", "summary"), default="prepare")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--account", default="mkolar_1314")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    core.require(args.run_dir.is_absolute() and root.is_relative_to(Path("/scratch2")) and root != Path("/scratch2"),
                 "new output must be below /scratch2")
    core.require(args.workers > 0 and re.fullmatch(r"[1-9][0-9]*[KMGT]?", args.mem)
                 and re.fullmatch(r"([0-9]+-)?[0-9]+(:[0-9]{2}){0,2}", args.time), "invalid resources")
    core.require(all(re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in (args.account, args.partition)), "invalid account/partition")
    (root / "campaign/attempts").mkdir(parents=True, exist_ok=True)
    with (root / ".submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(submit(root, args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
