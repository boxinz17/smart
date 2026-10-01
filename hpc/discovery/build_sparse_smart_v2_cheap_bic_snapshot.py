#!/usr/bin/env python3
"""Build a fresh source-only transfer bundle for the bounded cheap BIC pilot.

This local operation reads source, Git metadata and a frozen reference plan. It
does not read observed/evaluation arrays, fit an estimator, use SSH, or submit
jobs. Baseline manifest paths select CURRENT repository files, not historical
snapshot bytes. All files are hashed, including dirty and untracked additions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "simulation"))
from sparse_smart_v2_bic_plan import canonical, sha
from sparse_smart_v2_cheap_bic_plan import split_manifest

BASELINE = CODE / "simulation/result/sparse-smart-v2-bic100-20260913T190017Z"
EXHAUSTIVE_ROOT = "/scratch2/mkolar/smart/runs/sparse-smart-v2-bic100-20260913T190017Z"
REQUIRED_ADDITIONS = (
    "smart/smart/rank_selector.py",
    "hpc/discovery/build_sparse_smart_v2_cheap_bic_snapshot.py",
    "hpc/discovery/submit_sparse_smart_v2_cheap_bic.py",
    "hpc/discovery/sparse_smart_v2_cheap_bic_stage.sbatch",
    "hpc/discovery/CHEAP_BIC_PILOT.md",
    "simulation/run_sparse_smart_v2_cheap_bic.py",
    "simulation/sparse_smart_v2_cheap_bic_plan.py",
    "simulation/sparse_smart_v2_cheap_bic_reuse.py",
    "simulation/summarize_sparse_smart_v2_cheap_bic.py",
    "simulation/tests/test_sparse_smart_v2_cheap_bic_plan.py",
    "simulation/tests/test_sparse_smart_v2_cheap_bic_operations.py",
    "simulation/tests/test_summarize_sparse_smart_v2_cheap_bic.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_state(code=CODE):
    def command(*args):
        return subprocess.run(["git", *args], cwd=code, check=True, capture_output=True).stdout
    status = command("status", "--porcelain=v1", "--untracked-files=all").decode()
    patch = command("diff", "--binary", "HEAD", "--")
    return dict(commit=command("rev-parse", "HEAD").decode().strip(), status_porcelain=status,
                diff_sha256=hashlib.sha256(patch).hexdigest(),
                diff_scope="git diff --binary HEAD --; untracked contents bound by source file hashes")


def source_paths(baseline_manifest, code=CODE):
    require(baseline_manifest.get("schema_version") == 1 and isinstance(baseline_manifest.get("files"), dict),
            "invalid baseline source manifest")
    paths = set(baseline_manifest["files"]) | set(REQUIRED_ADDITIONS)
    # Include any additional focused cheap-pilot tests or helper modules added
    # before this snapshot, without scanning result folders or raw data.
    for folder in (code / "simulation", code / "simulation/tests", code / "hpc/discovery"):
        paths.update(path.relative_to(code).as_posix() for path in folder.glob("*cheap_bic*") if path.is_file())
    for name in paths:
        relative, source = Path(name), code / name
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"unsafe source path: {name}")
        require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(code.resolve()),
                f"source is missing or not a regular repository file: {name}")
        require("result" not in relative.parts and source.suffix not in (".npz", ".npy", ".pkl"),
                f"raw result/data file is not allowed in the source bundle: {name}")
    return sorted(paths)


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def build(local_root, remote_root, *, baseline_manifest=BASELINE / "source-manifest.json",
          reference_plan_path=BASELINE / "plan.json"):
    local_root, remote_root = Path(local_root), Path(remote_root)
    require(local_root.is_absolute(), "local root must be absolute")
    require(remote_root.is_absolute() and remote_root.is_relative_to(Path("/scratch2"))
            and ".." not in remote_root.parts and remote_root != Path("/scratch2"),
            "remote root must be a fresh absolute directory below /scratch2")
    archive = local_root.with_name(local_root.name + ".tar.gz")
    receipt = local_root.with_name(local_root.name + ".bundle.json")
    require(not local_root.exists() and not local_root.is_symlink() and not archive.exists() and not receipt.exists(),
            "local root, sibling archive and receipt must all be fresh")
    reference_bytes = Path(reference_plan_path).read_bytes()
    reference = json.loads(reference_bytes)
    require(reference.get("root") == EXHAUSTIVE_ROOT, "reference plan belongs to a different exhaustive campaign")
    require(str(remote_root) not in (reference["root"], reference["reference_root"]),
            "new pilot cannot replace an existing campaign")
    # Freeze cohort and criterion before any rank or evaluation information is
    # opened. This module never opens those records at all.
    split = split_manifest(reference)
    baseline = json.loads(Path(baseline_manifest).read_text())
    paths = source_paths(baseline)
    before = git_state()
    created = datetime.now(timezone.utc).isoformat()
    local_root.mkdir(parents=True)
    write_json(local_root / "split-manifest.json", split)
    hashes = {}
    for relative in paths:
        current, target = CODE / relative, local_root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, target)
        hashes[relative] = sha(target)
        require(sha(current) == hashes[relative], f"source changed while copying: {relative}")
    after = git_state()
    require(before == after, "Git worktree changed during snapshot; use a new local root after edits settle")
    require(all(sha(CODE / relative) == value for relative, value in hashes.items()),
            "source changed during snapshot; use a new local root after edits settle")
    manifest = dict(schema_version=1, source_root=str(remote_root / "source"), created_utc=created,
        files=hashes, git=after, baseline_manifest_sha256=sha(baseline_manifest),
        baseline_manifest_used_for="path inventory only; all bytes copied from the current repository",
        additions=sorted(set(paths) - set(baseline["files"])),
        source_policy="frozen current source, including declared dirty changes; no raw observations or fit directories")
    write_json(local_root / "source-manifest.json", manifest)
    pilot = dict(schema_version=1, method="SparseSMARTv2CheapRSCT200", remote_root=str(remote_root),
        accepted_update_cap=200, workers=100, pool_time="06:00:00", account="mkolar_1314", partition="main",
        exhaustive_root=EXHAUSTIVE_ROOT,
        exhaustive_plan_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        source_manifest_sha256=sha(local_root / "source-manifest.json"),
        split_fingerprint=split["split_fingerprint"], development_seed_ids=list(range(5)),
        assessment_seed_ids=list(range(5, 10)),
        assessment_policy="fit assessment only after development freezes one global initializer pair",
        resource_policy="one 100-CPU GNU Parallel pool at a time, exclusive one-CPU srun workers; no arrays",
        scientific_policy="training RSC theta=1, tied penalties, terminal BIC, stationarity or 200 accepted updates; no longer-budget controls",
        existing_jobs_policy="no cancellations, releases, resource changes or resubmissions of existing campaigns",
        created_utc=created)
    write_json(local_root / "pilot.json", pilot)
    write_json(local_root / "snapshot-provenance.json", dict(schema_version=1, git=after,
        local_source_root=str(CODE), remote_root=str(remote_root), created_utc=created,
        baseline_manifest_sha256=sha(baseline_manifest), exhaustive_plan_sha256=pilot["exhaustive_plan_sha256"],
        source_manifest_sha256=pilot["source_manifest_sha256"], source_files=len(hashes),
        split_manifest_sha256=sha(local_root / "split-manifest.json"),
        builder="hpc/discovery/build_sparse_smart_v2_cheap_bic_snapshot.py"))
    with tarfile.open(archive, "w:gz", compresslevel=6) as bundle:
        for path in sorted(local_root.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(local_root).as_posix(), recursive=False)
    output = dict(local_root=str(local_root), remote_root=str(remote_root), archive=str(archive),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        source_files=len(hashes), source_manifest_sha256=pilot["source_manifest_sha256"],
        split_fingerprint=split["split_fingerprint"], git_commit=after["commit"])
    # This local receipt intentionally lives outside the transferred snapshot.
    write_json(receipt, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--remote-root", required=True, type=Path)
    parser.add_argument("--baseline-manifest", type=Path, default=BASELINE / "source-manifest.json")
    parser.add_argument("--reference-plan", type=Path, default=BASELINE / "plan.json")
    args = parser.parse_args(argv)
    try:
        result = build(args.local_root, args.remote_root, baseline_manifest=args.baseline_manifest,
                       reference_plan_path=args.reference_plan)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
