#!/usr/bin/env python3
"""Freeze current source for the 100-seed checkpoint-BIC campaign; no fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_sparse_smart_v2_cheap_bic_snapshot as previous

require, sha, canonical, git_state = previous.require, previous.sha, previous.canonical, previous.git_state
BASELINE, EXHAUSTIVE_ROOT = previous.BASELINE, previous.EXHAUSTIVE_ROOT
REQUIRED_ADDITIONS = (
    "smart/smart/rank_selector.py",
    "simulation/sparse_smart_v2_cheap_bic_plan.py",
    "simulation/sparse_smart_v2_cheap_bic_reuse.py",
    "hpc/discovery/build_sparse_smart_v2_cheap_bic_snapshot.py",
    "hpc/discovery/build_sparse_smart_v2_checkpoint_bic_snapshot.py",
    "hpc/discovery/submit_sparse_smart_v2_checkpoint_bic.py",
    "hpc/discovery/sparse_smart_v2_checkpoint_bic_stage.sbatch",
    "simulation/run_sparse_smart_v2_checkpoint_bic_campaign.py",
    "simulation/sparse_smart_v2_checkpoint_bic_plan.py",
    "simulation/summarize_sparse_smart_v2_checkpoint_bic.py",
    "simulation/tests/test_sparse_smart_v2_checkpoint_bic_operations.py",
    "simulation/tests/test_summarize_sparse_smart_v2_checkpoint_bic.py",
    "simulation/tests/test_summarize_sparse_smart_v2_bic.py",
)


def source_paths(baseline, code=None):
    code = CODE if code is None else Path(code)
    require(baseline.get("schema_version") == 1 and isinstance(baseline.get("files"), dict),
            "invalid baseline manifest")
    paths = set(baseline["files"]) | set(REQUIRED_ADDITIONS)
    for folder in (code / "simulation", code / "simulation/tests", code / "hpc/discovery"):
        paths.update(path.relative_to(code).as_posix() for path in folder.glob("*checkpoint_bic*") if path.is_file())
    for name in paths:
        part, source = Path(name), code / name
        require(not part.is_absolute() and ".." not in part.parts, "unsafe source path")
        require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(code.resolve()),
                f"missing or unsafe current source: {name}")
        require("result" not in part.parts and source.suffix not in (".npz", ".npy", ".pkl"),
                "raw data/results cannot enter the source bundle")
    return sorted(paths)


def write(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def build(local_root, remote_root, *, baseline_manifest=BASELINE / "source-manifest.json",
          reference_plan_path=BASELINE / "plan.json", comparison_baselines=None, workers=100):
    local_root, remote_root = Path(local_root), Path(remote_root)
    require(local_root.is_absolute(), "local root must be absolute")
    require(remote_root.is_absolute() and remote_root.is_relative_to(Path("/scratch2")) and
            remote_root != Path("/scratch2") and ".." not in remote_root.parts,
            "new remote root must be below /scratch2")
    require(type(workers) is int and workers > 0, "workers must be a positive integer")
    archive = local_root.with_name(local_root.name + ".tar.gz")
    receipt = local_root.with_name(local_root.name + ".bundle.json")
    require(not any(path.exists() or path.is_symlink() for path in (local_root, archive, receipt)),
            "snapshot, archive and receipt must be fresh")
    reference_bytes = Path(reference_plan_path).read_bytes()
    reference = json.loads(reference_bytes)
    require(reference["root"] == EXHAUSTIVE_ROOT, "reference plan belongs to another exhaustive campaign")
    require(str(remote_root) not in (reference["root"], reference["reference_root"]),
            "new campaign cannot replace an existing root")
    require(reference["selection_scope"]["models"] == [0, 1, 2] and
            reference["selection_scope"]["seed_ids"] == list(range(100)), "expected all models and 100 seeds")
    baseline = json.loads(Path(baseline_manifest).read_text())
    paths, before = source_paths(baseline), git_state()
    local_root.mkdir(parents=True)
    hashes = {}
    for name in paths:
        target = local_root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / name, target)
        hashes[name] = sha(target)
        require(sha(CODE / name) == hashes[name], f"source changed while copying {name}")
    after = git_state()
    require(before == after and all(sha(CODE / name) == value for name, value in hashes.items()),
            "source changed during snapshot; retry using a fresh local directory after edits settle")
    created = datetime.now(timezone.utc).isoformat()
    manifest = dict(schema_version=1, source_root=str(remote_root / "source"), files=hashes,
        git=after, created_utc=created, baseline_manifest_sha256=sha(baseline_manifest),
        source_policy="current source including dirty/untracked bytes; baseline supplies paths only")
    write(local_root / "source-manifest.json", manifest)
    config = dict(schema_version=1, method="SparseSMARTv2CheckpointBIC100", remote_root=str(remote_root),
        accepted_update_cap=200, initializer_pair=[.003, .1], tied_penalties=[.001, .0025, .01],
        checkpoint_iterations=[0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200],
        seed_ids=list(range(100)), models=[0, 1, 2], experiments=[0, 1, 2, 3],
        workers=workers, max_tasks_per_chunk=1000, pool_time="06:00:00", mem_per_cpu="4G",
        stage_time="06:00:00", stage_mem="16G", account="mkolar_1314", partition="main",
        exhaustive_root=EXHAUSTIVE_ROOT, exhaustive_plan_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        source_manifest_sha256=sha(local_root / "source-manifest.json"), created_utc=created,
        resource_policy="one GNU Parallel allocation at a time, exclusive single-CPU srun steps; no arrays",
        gate_policy="production follows complete canary execution and numerical state audit",
        selection_policy="original training RSC once per dataset; minimum training BIC over checkpoints and candidates",
        storage_policy="read existing data/source arrays in place; no raw copies or archival",
        existing_jobs_policy="do not modify, cancel, or resubmit older campaigns")
    if comparison_baselines is not None:
        content = Path(comparison_baselines).read_bytes()
        require(isinstance(json.loads(content), dict), "comparison baseline must be a JSON object")
        (local_root / "comparison-baselines.json").write_bytes(content)
        config["comparison_baselines_sha256"] = sha(local_root / "comparison-baselines.json")
    write(local_root / "campaign.json", config)
    write(local_root / "snapshot-provenance.json", dict(schema_version=1, git=after,
        source_files=len(hashes), remote_root=str(remote_root), local_source_root=str(CODE),
        source_manifest_sha256=config["source_manifest_sha256"], campaign_sha256=sha(local_root / "campaign.json"),
        exhaustive_plan_sha256=config["exhaustive_plan_sha256"], created_utc=created))
    with tarfile.open(archive, "w:gz", compresslevel=6) as bundle:
        for path in sorted(local_root.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(local_root).as_posix(), recursive=False)
    result = dict(local_root=str(local_root), remote_root=str(remote_root), archive=str(archive),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, source_files=len(hashes),
        source_manifest_sha256=config["source_manifest_sha256"], git_commit=after["commit"])
    write(receipt, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, default=BASELINE / "source-manifest.json")
    parser.add_argument("--reference-plan", type=Path, default=BASELINE / "plan.json")
    parser.add_argument("--comparison-baselines", type=Path)
    parser.add_argument("--workers", type=int, default=100)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.local_root, args.remote_root, baseline_manifest=args.baseline_manifest,
        reference_plan_path=args.reference_plan, comparison_baselines=args.comparison_baselines,
        workers=args.workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
