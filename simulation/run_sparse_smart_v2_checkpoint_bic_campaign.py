#!/usr/bin/env python3
"""Prepare, canary-audit and summarize the isolated 100-seed checkpoint campaign."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import run_sparse_smart_v2_bic as fitting
from sparse_smart_v2_bic_plan import canonical, immutable, sha
from sparse_smart_v2_cheap_bic_plan import rank_selection
from sparse_smart_v2_checkpoint_bic_plan import freeze_plan

read, write, require = fitting.read, fitting.write, fitting.require
EXPECTED_DATASETS = 3000


def launcher(root):
    path = root / "source/hpc/discovery/submit_sparse_smart_v2_checkpoint_bic.py"
    spec = importlib.util.spec_from_file_location("_checkpoint_launcher", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def check_inputs(root):
    fitting.legacy._require_slurm(root)
    manifest, _ = fitting.legacy._manifest(root, full=True)
    settings = launcher(root).settings(root)
    reference_root = Path(settings["exhaustive_root"])
    require(reference_root.is_relative_to(Path("/scratch2")) and reference_root != root,
            "invalid exhaustive reference root")
    require(sha(reference_root / "plan.json") == settings["exhaustive_plan_sha256"], "exhaustive plan changed")
    reference = read(reference_root / "plan.json")
    require(reference["selection_scope"]["models"] == [0, 1, 2] and
            reference["selection_scope"]["seed_ids"] == list(range(100)), "reference campaign scope differs")
    return settings, manifest, reference_root, reference


def prepare(root, *, submit=True):
    _, _, old_root, reference = check_inputs(root)
    started, cpu = time.monotonic(), time.process_time()
    old_preparation = read(old_root / "preparation.json")
    require(old_preparation.get("success") is True and old_preparation.get("status") == "complete" and
            old_preparation.get("plan_fingerprint") == reference["plan_fingerprint"] and
            old_preparation.get("plan_sha256") == sha(old_root / "plan.json"), "reference preparation identity differs")
    prepared = {g["group_id"]: g for g in old_preparation["groups"]}
    records, timing = {}, []
    existing = read(root / "rank-records.json") if (root / "rank-records.json").exists() else {}
    for group in reference["groups"]:
        dataset = group["dataset_id"]
        if dataset in records:
            continue
        tick, task_cpu = time.monotonic(), time.process_time()
        meta_path = old_root / "groups" / group["group_id"] / "group.json"
        require(sha(meta_path) == prepared[group["group_id"]]["group_json_sha256"], "reference group hash differs")
        meta = read(meta_path)
        require(meta["group"] == group and meta["plan_fingerprint"] == reference["plan_fingerprint"],
                "reference group identity differs")
        data_path = Path(meta["files"]["data.npz"]["path"])
        require(data_path.is_relative_to(Path("/scratch2")) and
                sha(data_path) == meta["files"]["data.npz"]["sha256"], "reference observations changed")
        if dataset in existing:
            record = existing[dataset]
        else:
            with np.load(data_path, allow_pickle=False) as archive:
                record = rank_selection(archive["X"], archive["Y"], archive["C0"])
        require(record["training_observed_input_fingerprint"] == meta["fingerprints"]["training_observed_input_fingerprint"],
                "rank selection data fingerprint differs")
        records[dataset] = record
        timing.append(dict(dataset_id=dataset, elapsed_seconds=time.monotonic()-tick,
                           process_cpu_seconds=time.process_time()-task_cpu, reused_rank_record=dataset in existing))
        if len(records) % 25 == 0:
            print(json.dumps(dict(ranked_datasets=len(records))), flush=True)
    require(len(records) == EXPECTED_DATASETS, "expected exactly 3000 distinct training datasets")
    immutable(root / "rank-records.json", canonical(records) + b"\n")
    write(root / "rank-selection-timing.json", dict(records=timing, ranked_datasets=len(records),
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu))
    plan = freeze_plan(root, sha(root / "source-manifest.json"), reference, records)
    require(plan["n_datasets"] == EXPECTED_DATASETS and plan["selection_scope"]["seed_ids"] == list(range(100)),
            "frozen plan differs from approved 100-seed scope")
    ready = fitting.prepare(root)
    # Every shared observed input must agree with the observations used by RSC,
    # including multiple spectral protocols for one aliased dataset.
    for group in plan["groups"]:
        meta = read(root / "groups" / group["group_id"] / "group.json")
        require(meta["fingerprints"]["training_observed_input_fingerprint"] ==
                records[group["dataset_id"]]["training_observed_input_fingerprint"],
                "fitting observations differ from RSC inputs")
    result = dict(success=True, status="complete", plan_fingerprint=plan["plan_fingerprint"],
        n_groups=plan["n_groups"], n_tasks=plan["n_tasks"], ranked_datasets=len(records),
        canary_tasks=len(plan["canary_task_ids"]), selected_rank_counts=dict(Counter(r["selected_rank"] for r in records.values())),
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), preparation_sha256=sha(root / "preparation.json"))
    require(ready["success"], "fitting preparation incomplete")
    write(root / "campaign-preparation.json", result)
    if submit:
        launcher(root).launch_phase(root, "canary")
    return result


def canary_audit(root, *, submit=True):
    check_inputs(root)
    import summarize_sparse_smart_v2_checkpoint_bic as analysis
    plan = read(root / "plan.json")
    report = analysis.audit(root, task_ids=plan["canary_task_ids"])
    result = dict(success=bool(report["success"]), report=report, task_ids=plan["canary_task_ids"],
        plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"],
        finished_at=fitting.legacy._now())
    write(root / "canary-audit.json", result)
    require(result["success"], "canary execution/numerical audit failed; production remains unsubmitted")
    if submit:
        launcher(root).launch_phase(root, "production")
    return result


def summarize(root):
    check_inputs(root)
    import summarize_sparse_smart_v2_checkpoint_bic as analysis
    result = analysis.summarize(root, root / "summary")
    completion = dict(success=bool(result["success"]), status="complete" if result["success"] else "failed",
        summary_root=str(root / "summary"), finished_at=fitting.legacy._now(), accepted_update_cap=200,
        plan_fingerprint=read(root / "plan.json")["plan_fingerprint"])
    write(root / "completion.json", completion)
    require(completion["success"], "campaign summary failed; inspect coverage and state audit")
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "canary-audit", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        result = {"prepare": lambda: prepare(root, submit=not args.no_submit),
                  "canary-audit": lambda: canary_audit(root, submit=not args.no_submit),
                  "summary": lambda: summarize(root)}[args.stage]()
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        write(root / (args.stage + "-failure.json"), dict(error_type=type(error).__name__, error=str(error),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"), recorded_utc=fitting.legacy._now()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
