#!/usr/bin/env python3
"""Freeze current code and existing case identities for read-only BIC rescoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

CODE = Path(__file__).resolve().parents[2]
DEFAULT_PREVIOUS = CODE / "simulation/result/sparse-smart-v2-retrospective-bic-20260914T000000Z"
METHOD = "SparseSMARTv2RetainedStateBIC"
ARMS = ("original_validation", "bic_validation_selected", "bic_validation_terminal",
        "bic_saved_current", "bic_all_retained")
EXPECTED_CASES = 6600
REQUIRED = (
    "hpc/discovery/build_sparse_smart_v2_retained_state_bic_snapshot.py",
    "hpc/discovery/submit_sparse_smart_v2_retained_state_bic.py",
    "hpc/discovery/sparse_smart_v2_retained_state_bic.sbatch",
    "hpc/discovery/sparse_smart_v2_retained_state_bic_worker.sh",
    "simulation/run_sparse_smart_v2_retained_state_bic.py",
    "simulation/summarize_sparse_smart_v2_retained_state_bic.py",
    "simulation/sparse_smart_v2_retained_state_scoring.py",
    "simulation/tests/test_sparse_smart_v2_retained_state_scoring.py",
    "simulation/tests/test_sparse_smart_v2_retained_state_bic_operations.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=CODE, check=True, capture_output=True).stdout
    patch = git("diff", "--binary", "HEAD", "--")
    return dict(commit=git("rev-parse", "HEAD").decode().strip(),
        status_porcelain=git("status", "--porcelain=v1", "--untracked-files=all").decode(),
        diff_sha256=hashlib.sha256(patch).hexdigest(),
        diff_scope="git diff --binary HEAD --; all included untracked bytes additionally bound by source hashes")


def source_paths(manifest, code=None):
    code = CODE if code is None else Path(code)
    require(manifest.get("schema_version") == 1 and isinstance(manifest.get("files"), dict)
            and manifest["files"], "invalid baseline source manifest")
    names = set(manifest["files"]) | set(REQUIRED)
    for folder in ("simulation", "simulation/tests", "hpc/discovery"):
        names.update(path.relative_to(code).as_posix() for path in (code / folder).glob("*retained_state*") if path.is_file())
    for name in names:
        part, path = Path(name), code / name
        require(not part.is_absolute() and ".." not in part.parts and path.is_file()
                and not path.is_symlink() and path.resolve().is_relative_to(code.resolve()),
                f"missing or unsafe current source: {name}")
        require("result" not in part.parts and path.suffix not in (".npz", ".npy", ".pkl"),
                f"raw result/data cannot be bundled: {name}")
    return sorted(names)


def write(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def build(local_root, remote_root, *, previous_plan=DEFAULT_PREVIOUS / "plan.json",
          baseline_manifest=None, preflight_case_ids, chart_change_case_ids=()):
    local_root, remote_root, previous_plan = map(Path, (local_root, remote_root, previous_plan))
    baseline_manifest = previous_plan.parent / "source-manifest.json" if baseline_manifest is None else Path(baseline_manifest)
    require(local_root.is_absolute(), "local root must be absolute")
    require(remote_root.is_absolute() and remote_root.is_relative_to(Path("/scratch2"))
            and remote_root != Path("/scratch2") and ".." not in remote_root.parts,
            "new remote root must be below /scratch2")
    archive = local_root.with_name(local_root.name + ".tar.gz")
    receipt = local_root.with_name(local_root.name + ".bundle.json")
    require(not any(p.exists() or p.is_symlink() for p in (local_root, archive, receipt)),
            "snapshot, archive and receipt must all be fresh")
    old = json.loads(previous_plan.read_text())
    require(old.get("method") == "SparseSMARTv2RetrospectiveBIC" and old.get("schema_version") == 1
            and old.get("plan_fingerprint") == digest({k: v for k, v in old.items() if k != "plan_fingerprint"}),
            "invalid previous retrospective plan")
    require(old.get("n_cases") == len(old["cases"]) == EXPECTED_CASES, "unexpected previous case coverage")
    require(sha(previous_plan.parent / "source-manifest.json") == old["source_manifest_sha256"],
            "previous source manifest identity differs")
    for name in ("root", "source_campaign_root"):
        prior = Path(old[name])
        require(prior.is_absolute() and prior.is_relative_to(Path("/scratch2")) and
                not remote_root.is_relative_to(prior) and not prior.is_relative_to(remote_root),
                "new output overlaps existing campaign or is outside scratch2")
    ids = [r["case_id"] for r in old["cases"]]
    require(len(set(ids)) == len(ids) and all(re.fullmatch(r"m[0-9]+_e[0-9]+_k[0-9]+_s[0-9]+", x) for x in ids),
            "invalid previous case identifiers")
    chosen, chart = list(preflight_case_ids), list(chart_change_case_ids)
    require(chosen and len(chosen) == len(set(chosen)) and set(chosen) <= set(ids), "invalid preflight cases")
    require(len(chart) == len(set(chart)) and set(chart) <= set(chosen), "chart-change cases must be a preflight subset")
    require({r["case"]["model_id"] for r in old["cases"] if r["case_id"] in chosen} == {0, 1, 2},
            "preflight must cover all three models")
    require(len(chosen) < len(ids), "preflight cannot consume the full campaign")
    baseline = json.loads(baseline_manifest.read_text())
    names, before = source_paths(baseline), git_state()
    local_root.mkdir(parents=True)
    hashes = {}
    for name in names:
        target = local_root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / name, target)
        hashes[name] = sha(target)
        require(sha(CODE / name) == hashes[name], "source changed during copying")
    after = git_state()
    require(before == after and all(sha(CODE / name) == h for name, h in hashes.items()),
            "source changed during snapshot; retry in a fresh root after edits settle")
    created = datetime.now(timezone.utc).isoformat()
    manifest = dict(schema_version=1, source_root=str(remote_root / "source"), files=hashes,
        git=after, created_utc=created, baseline_manifest_sha256=sha(baseline_manifest),
        source_policy="baseline paths select current code bytes; dirty and untracked additions are explicitly hashed")
    write(local_root / "source-manifest.json", manifest)
    plan = {k: v for k, v in old.items() if k not in ("plan_fingerprint", "created_utc", "selection_arms")}
    plan.update(method=METHOD, schema_version=1, root=str(remote_root), created_utc=created,
        source_manifest_sha256=sha(local_root / "source-manifest.json"), previous_retrospective_root=old["root"],
        previous_plan_sha256=sha(previous_plan), previous_source_manifest_sha256=old["source_manifest_sha256"],
        preflight_case_ids=chosen, preflight_chart_change_case_ids=chart,
        preflight_chart_coverage_reason=("declared_cases_have_observed_chart_changes" if chart else
            "original_sample_no_chart_events; chart switches and recenters covered by focused saved-state reconstruction tests"),
        selection_arms=list(ARMS),
        no_refitting=True, no_raw_copies=True, validation_influenced_trajectories=True,
        selection_scope="all retained states reconstructed with their own saved chart; original validation-influenced trajectories unchanged",
        operations=dict(workers=32, mem_per_cpu="4G", pool_time="24:00:00", account="mkolar_1314", partition="main",
                        stage_mem="16G", stage_time="04:00:00", preflight_time="04:00:00",
                        production_gate="manual release only after frozen preflight reconstruction, coverage, history and monotonicity checks"))
    completion_path = previous_plan.parent / "completion.json"
    if completion_path.exists():
        plan["previous_completion_sha256"] = sha(completion_path)
    plan["plan_fingerprint"] = digest(plan)
    write(local_root / "plan.json", plan)
    for name, rows in (("work-items.tsv", ids), ("preflight-work-items.tsv", chosen),
                       ("production-work-items.tsv", [cid for cid in ids if cid not in set(chosen)])):
        (local_root / name).write_text("".join(cid + "\n" for cid in rows))
    write(local_root / "snapshot-provenance.json", dict(schema_version=1, git=after, source_files=len(hashes),
        plan_sha256=sha(local_root / "plan.json"), source_manifest_sha256=plan["source_manifest_sha256"],
        previous_plan_sha256=plan["previous_plan_sha256"], remote_root=str(remote_root), created_utc=created))
    with tarfile.open(archive, "w:gz", compresslevel=6) as bundle:
        for path in sorted(local_root.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(local_root).as_posix(), recursive=False)
    result = dict(local_root=str(local_root), remote_root=str(remote_root), archive=str(archive),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, source_files=len(hashes),
        plan_fingerprint=plan["plan_fingerprint"], n_cases=len(ids), preflight_cases=len(chosen),
        source_manifest_sha256=plan["source_manifest_sha256"])
    write(receipt, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--remote-root", required=True, type=Path)
    parser.add_argument("--previous-plan", type=Path, default=DEFAULT_PREVIOUS / "plan.json")
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--preflight-case-ids", required=True, nargs="+")
    parser.add_argument("--chart-change-case-ids", nargs="*", default=[])
    args = parser.parse_args(argv)
    print(json.dumps(build(args.local_root, args.remote_root, previous_plan=args.previous_plan,
        baseline_manifest=args.baseline_manifest, preflight_case_ids=args.preflight_case_ids,
        chart_change_case_ids=args.chart_change_case_ids), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
