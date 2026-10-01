#!/usr/bin/env python3
"""Slurm stages for the isolated, development-gated 200-update RSC pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_sparse_smart_v2_bic as fitting
from sparse_smart_v2_bic_plan import canonical, digest, immutable, sha
from sparse_smart_v2_cheap_bic_plan import build_plan, freeze_split, rank_selection, split_manifest
from sparse_smart_v2_cheap_bic_reuse import rrr_descriptor

read, write, require = fitting.read, fitting.write, fitting.require


def check_inputs(root):
    fitting.legacy._require_slurm(root)
    manifest, _ = fitting.legacy._manifest(root, full=True)
    settings = read(root / "pilot.json")
    require(settings["accepted_update_cap"] == 200 and settings["workers"] == 100, "invalid pilot settings")
    require(sha(root / "source-manifest.json") == settings["source_manifest_sha256"], "pilot source changed")
    old_root = Path(settings["exhaustive_root"])
    require(old_root.is_relative_to(Path("/scratch2")) and old_root != root, "invalid existing campaign root")
    require(sha(old_root / "plan.json") == settings["exhaustive_plan_sha256"], "existing plan changed")
    reference = read(old_root / "plan.json")
    require(read(root / "split-manifest.json") == split_manifest(reference), "prespecified split changed")
    return settings, manifest, old_root, reference


def launcher(root):
    path = root / "source/hpc/discovery/submit_sparse_smart_v2_cheap_bic.py"
    spec = importlib.util.spec_from_file_location("_cheap_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root, *, submit=True):
    settings, manifest, old_root, reference = check_inputs(root)
    started, cpu = time.monotonic(), time.process_time()
    split = read(root / "split-manifest.json")
    wanted = set(split["phases"]["development"]["group_ids"] + split["phases"]["assessment"]["group_ids"])
    groups = [g for g in reference["groups"] if g["group_id"] in wanted]
    rank_records, timing, errors = {}, [], []
    old_preparation = read(old_root / "preparation.json")
    require(old_preparation.get("success") is True and old_preparation.get("status") == "complete"
            and old_preparation.get("plan_fingerprint") == reference["plan_fingerprint"]
            and old_preparation.get("plan_sha256") == sha(old_root / "plan.json"),
            "existing preparation does not match frozen plan")
    prepared = {r["group_id"]: r for r in old_preparation["groups"]}
    # Split has already been frozen; ranks consume observed training only.
    for group in groups:
        dataset = group["dataset_id"]
        if dataset in rank_records:
            continue
        tick, task_cpu = time.monotonic(), time.process_time()
        try:
            path = old_root / "groups" / group["group_id"] / "group.json"
            require(sha(path) == prepared[group["group_id"]]["group_json_sha256"], "existing group metadata changed")
            meta = read(path)
            require(meta.get("group") == group and meta.get("plan_fingerprint") == reference["plan_fingerprint"],
                    "existing group metadata does not match frozen plan")
            data_path = Path(meta["files"]["data.npz"]["path"])
            require(data_path.is_relative_to(Path("/scratch2")) and sha(data_path) == meta["files"]["data.npz"]["sha256"],
                    "existing data identity changed")
            with np.load(data_path, allow_pickle=False) as archive:
                record = rank_selection(archive["X"], archive["Y"], archive["C0"])
            require(record["training_observed_input_fingerprint"] == meta["fingerprints"]["training_observed_input_fingerprint"],
                    "RSC training fingerprint mismatch")
            rank_records[dataset] = record
            timing.append(dict(dataset_id=dataset, elapsed_seconds=time.monotonic()-tick,
                               process_cpu_seconds=time.process_time()-task_cpu))
        except (ValueError, OSError, KeyError) as error:
            errors.append(dict(dataset_id=dataset, error=str(error)))
        if len(rank_records) % 25 == 0:
            print(json.dumps(dict(ranked_datasets=len(rank_records), errors=len(errors))), flush=True)
    write(root / "rank-selection-timing.json", dict(records=timing, errors=errors,
          elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu))
    require(not errors and len(rank_records) == 300, "RSC preflight failed; inspect rank-selection-timing.json")
    immutable(root / "rank-records.json", canonical(rank_records) + b"\n")
    phases = {}
    for phase in ("development", "assessment"):
        phase_root = root / phase
        phase_root.mkdir(exist_ok=True)
        link = phase_root / "source"
        if not link.exists():
            link.symlink_to(root / "source", target_is_directory=True)
        require(link.resolve() == (root / "source").resolve(), "phase source differs")
        immutable(phase_root / "source-manifest.json", (root / "source-manifest.json").read_bytes())
        freeze_split(phase_root, reference)
        plan = build_plan(phase_root, sha(phase_root / "source-manifest.json"), reference, rank_records, phase)
        reused, availability = {}, []
        for group in plan["groups"]:
            rank = group["selected_target_rank"]
            descriptor, reason = rrr_descriptor(old_root, reference, group, rank, manifest)
            if descriptor:
                require(descriptor["training_fingerprint"] == rank_records[group["dataset_id"]]["training_observed_input_fingerprint"],
                        "existing RRR data differs from ranked data")
                reused[group["group_id"]] = descriptor
            availability.append(dict(group_id=group["group_id"], rank=rank, rrr=reason,
                iterative="unavailable_exact_200_update_protocol; old terminal cap is 2000",
                old_rank_available=rank in reference["selection_scope"]["fitted_ranks"]))
        plan["rrr_reuse"] = reused
        plan["plan_fingerprint"] = digest({k: v for k, v in plan.items() if k != "plan_fingerprint"})
        require(plan["n_tasks"] == 990 and plan["n_candidates"] <= 18000, "pilot scope exceeded")
        immutable(phase_root / "rank-records.json", canonical(plan["cheap_bic"]["rank_records"]) + b"\n")
        immutable(phase_root / "plan.json", canonical(plan) + b"\n")
        immutable(phase_root / "work-items.tsv", "".join(f"{t['task_id']}\n" for t in plan["tasks"]).encode())
        write(phase_root / "retrospective-availability.json", dict(records=availability,
              reused_rrr_endpoints=len(reused), missing_rrr_endpoints=165-len(reused),
              iterative_reuse=False, reason="2000-update terminal records do not identify a 200-update terminal state"))
        fitting.prepare(phase_root)
        for group in plan["groups"]:
            bound = read(phase_root / "groups" / group["group_id"] / "group.json")
            require(bound["fingerprints"]["training_observed_input_fingerprint"] ==
                    rank_records[group["dataset_id"]]["training_observed_input_fingerprint"],
                    "prepared fitting data differ from ranked observations")
        phases[phase] = dict(groups=plan["n_groups"], tasks=plan["n_tasks"], candidates=plan["n_candidates"],
                            reused_rrr_endpoints=len(reused), plan_fingerprint=plan["plan_fingerprint"])
    result = dict(success=True, status="complete", phases=phases,
                  ranked_datasets=300, selected_rank_counts=dict(Counter(r["selected_rank"] for r in rank_records.values())),
                  elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
                  finished_at=fitting.legacy._now(), slurm_job_id=os.environ["SLURM_JOB_ID"])
    write(root / "preparation.json", result)
    if submit:
        launcher(root).launch_phase(root, "development")
    return result


def develop(root, *, submit=True):
    check_inputs(root)
    import summarize_sparse_smart_v2_cheap_bic as analysis
    phase_root = root / "development"
    result = analysis.develop(phase_root, phase_root / "selection")
    require(result["success"], "development audit/pair selection failed; assessment was not released")
    pair_path = phase_root / "selection/selected-pair.json"
    pair = analysis.load_pair(pair_path)
    assessment = root / "assessment"
    gate = dict(status="released_after_pair_freeze", pair_path=str(pair_path), pair_sha256=sha(pair_path),
                development_plan_fingerprint=read(phase_root / "plan.json")["plan_fingerprint"],
                assessment_plan_fingerprint=read(assessment / "plan.json")["plan_fingerprint"],
                pair_fingerprint=pair["artifact_fingerprint"], released_utc=fitting.legacy._now())
    gate_path = assessment / "assessment-gate.json"
    if gate_path.exists():
        previous = read(gate_path)
        require(all(previous.get(k) == v for k, v in gate.items() if k != "released_utc"), "assessment gate changed")
    else:
        immutable(gate_path, canonical(gate) + b"\n")
    if submit:
        launcher(root).launch_phase(root, "assessment")
    return result


def assess(root):
    settings, _, _, _ = check_inputs(root)
    import summarize_sparse_smart_v2_cheap_bic as analysis
    result = analysis.assess(root / "assessment", root / "development/selection/selected-pair.json",
                            root / "summary", exhaustive_root=Path(settings["exhaustive_root"]))
    result["rank_selection_timing"] = read(root / "rank-selection-timing.json")
    result["longer_budget_controls_run"] = False
    write(root / "summary/assessment-report.json", result)
    write(root / "completion.json", dict(success=result["success"], status="complete" if result["success"] else "failed",
          summary_root=str(root / "summary"), finished_at=fitting.legacy._now(),
          accepted_update_cap=200, development_pair_sha256=sha(root / "development/selection/selected-pair.json")))
    require(result["success"], "assessment audit failed; inspect summary report")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "develop", "assess"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        result = {"prepare": lambda: prepare(root, submit=not args.no_submit),
                  "develop": lambda: develop(root, submit=not args.no_submit),
                  "assess": lambda: assess(root)}[args.stage]()
        print(json.dumps({k: result[k] for k in ("success", "status", "phases", "selected_rank_counts") if k in result}, indent=2))
        return 0
    except Exception as error:
        write(root / f"{args.stage}-failure.json", dict(error_type=type(error).__name__, error=str(error),
              slurm_job_id=os.environ.get("SLURM_JOB_ID"), recorded_utc=fitting.legacy._now()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
