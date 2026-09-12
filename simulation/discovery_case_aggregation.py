"""Freeze and submit bounded Discovery case audits; planning is stdlib-only.

The default 8 GiB / four-hour request is provisional and must be measured in a
pilot. One array element audits up to 25 cases sequentially on one CPU. Only
analysis code is copied, once; raw fit outputs and frozen fit sources stay put.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHOD = "DiscoveryCaseAggregationLaunch"
SHA = re.compile(r"[0-9a-f]{64}\Z")
JOB = re.compile(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\Z")
FIXED_SOURCE = (
    ".python-version", "python-constraints.txt", "environment/check_runtime.py",
    "hpc/discovery/env.sh", "hpc/discovery/case_aggregation.sbatch",
    "hpc/discovery/case_summary.sbatch", "hpc/discovery/submit_case_aggregation.sh",
    "simulation/data/random_seeds/experiment_seeds.csv",
)
PACKAGE_DIRS = ("smart/smart", "bi-smart/bi_smart", "sparse-smart/src/sparse_smart")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    path = Path(path)
    require(not path.is_symlink(), f"Refusing symlink: {path}")
    return json.loads(path.read_text())


def atomic_json(value, path):
    path = Path(path)
    require(not path.is_symlink(), f"Refusing symlink: {path}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def safe(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    require(path.is_relative_to(root), f"Path escapes root: {path}")
    for item in (path, *path.parents):
        if item == root:
            break
        require(not item.is_symlink(), f"Refusing symlink: {item}")
    return path


@contextmanager
def lock(path):
    path = Path(path)
    safe(path, path.parent)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Launcher operation already active: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_inventory(root):
    """Curated code/runtime files only: never traverse results or fit snapshots."""
    root = Path(root).resolve()
    paths = [root / name for name in FIXED_SOURCE]
    paths.extend((root / "simulation").glob("*.py"))
    for name in PACKAGE_DIRS:
        directory = safe(root / name, root)
        require(directory.is_dir(), f"Missing analysis package: {directory}")
        paths.extend(directory.rglob("*.py"))
    result = {}
    for path in sorted(set(paths)):
        safe(path, root)
        require(path.is_file(), f"Missing analysis source: {path}")
        result[path.relative_to(root).as_posix()] = sha(path)
    for name in ("aggregate_sparse_smart_cases.py", "summarize_sparse_smart_campaign.py",
                 "discovery_case_aggregation.py", "discovery_budget_study.py"):
        require(f"simulation/{name}" in result, f"Missing analysis module: {name}")
    return result


def freeze_source(root, destination):
    before = source_inventory(root)
    destination.mkdir()
    for name, expected in before.items():
        source, target = Path(root) / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(sha(target) == expected, f"Source changed while copying: {name}")
        target.chmod(0o444)
    require(source_inventory(root) == before, "Analysis source changed during snapshot")
    return before


def resolve_run_roots(run_roots=None, campaign_submission=None):
    require(bool(run_roots) != bool(campaign_submission),
            "Specify either run roots or a campaign submission JSON")
    if campaign_submission is not None:
        submission = read_json(campaign_submission)
        jobs = submission.get("jobs")
        require(isinstance(jobs, list) and jobs, "Campaign submission needs a jobs list")
        run_roots = [entry["run_dir"] for entry in jobs]
    return [str(Path(root).resolve()) for root in run_roots]


def prepare_launch(run_roots, output_root, *, models=None, publication_mode="compact",
                   cases_per_chunk=25, concurrency=4, memory="8G", time_limit="04:00:00",
                   account="mkolar_1314", partition="main", source_root=ROOT,
                   allowed_missing_tasks=None):
    for value, name in ((cases_per_chunk, "Cases per chunk"), (concurrency, "Concurrency")):
        require(type(value) is int and value > 0, f"{name} must be a positive integer")
    require(re.fullmatch(r"[1-9][0-9]*(?:[KMGTP])?", memory), "Invalid Slurm memory request")
    require(re.fullmatch(r"(?:[0-9]+-)?[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]", time_limit)
            and any(c in "123456789" for c in time_limit), "Invalid Slurm time limit")
    require(all(re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in (account, partition)),
            "Invalid account or partition")
    output, source_root = Path(output_root).resolve(), Path(source_root).resolve()
    require("\\" not in str(output), "Analysis output path cannot contain a backslash")
    roots = [Path(root).resolve() for root in run_roots]
    require(roots and len(set(roots)) == len(roots), "Require unique nonempty run roots")
    require(all(not output.is_relative_to(root) and not root.is_relative_to(output) for root in roots),
            "Analysis output must be separate from raw run roots")
    require(not output.is_relative_to(source_root) and not source_root.is_relative_to(output),
            "Analysis output must be separate from its source checkout")
    # Importing this planner performs no numerical work or estimator imports.
    import aggregate_sparse_smart_cases as cases
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / ".launch-prepare.lock"):
        launch_root = safe(output / "launcher", output)
        require(not launch_root.exists(), "Launcher already prepared; use its existing launch.json")
        temporary = Path(tempfile.mkdtemp(prefix=".launcher-", dir=output))
        try:
            files = freeze_source(source_root, temporary / "source")
            index = cases.prepare_campaign(roots, output, models=models, publication_mode=publication_mode,
                                           allowed_missing_tasks=allowed_missing_tasks)
            ordered = sorted(index["cases"], key=lambda row: (row["run_root"], row["case_key"]))
            chunks = [dict(chunk_id=i // cases_per_chunk, case_keys=[row["case_key"]
                      for row in ordered[i:i + cases_per_chunk]])
                      for i in range(0, len(ordered), cases_per_chunk)]
            atomic_json(chunks, temporary / "chunks.json")
            for name in ("logs", "reports"):
                (temporary / name).mkdir()
            index_path = output / "campaign-index.json"
            launch = dict(schema_version=1, method=METHOD, created_utc=datetime.now(timezone.utc).isoformat(),
                output_root=str(output), index=str(index_path), index_sha256=sha(index_path),
                index_fingerprint=index["index_fingerprint"], scope=index["scope"],
                publication_mode=index["publication_mode"], case_count=len(ordered),
                applicable_cases=sum(row["cell"]["inapplicability_reason"] is None for row in ordered),
                chunks_sha256=sha(temporary / "chunks.json"), chunk_count=len(chunks),
                cases_per_chunk=cases_per_chunk, source_root=str(launch_root / "source"),
                source=dict(original_root=str(source_root), files=files, fingerprint=digest(files)),
                resources=dict(account=account, partition=partition, nodes=1, ntasks=1,
                    cpus_per_task=1, memory=memory, time_limit=time_limit, concurrency=concurrency,
                    pilot_requests_provisional=True))
            if "allowed_missing_tasks" in index:
                launch["allowed_missing_tasks"] = index["allowed_missing_tasks"]
            atomic_json(launch, temporary / "launch.json")
            os.replace(temporary, launch_root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return launch_root / "launch.json"


def verify_launch(path, expected_sha=None, *, executing_source=None):
    path = Path(path).absolute()
    safe(path, path.parent.parent)
    if expected_sha is not None:
        require(SHA.fullmatch(expected_sha) and sha(path) == expected_sha, "Launch SHA256 mismatch")
    value = read_json(path)
    require(value.get("schema_version") == 1 and value.get("method") == METHOD, "Invalid launch manifest")
    output = path.parent.parent.resolve()
    require("\\" not in str(output), "Analysis output path cannot contain a backslash")
    require(path == output / "launcher/launch.json" and value["output_root"] == str(output),
            "Launch manifest location mismatch")
    index_path, source = output / "campaign-index.json", path.parent / "source"
    require(value["index"] == str(index_path) and value["source_root"] == str(source), "Launch paths mismatch")
    safe(source, output)
    require(sha(safe(index_path, output)) == value["index_sha256"], "Campaign index changed")
    require(sha(safe(path.parent / "chunks.json", output)) == value["chunks_sha256"], "Chunk roster changed")
    require(source_inventory(source) == value["source"]["files"]
            and digest(value["source"]["files"]) == value["source"]["fingerprint"],
            "Frozen analysis source changed")
    if executing_source is not None:
        require(Path(executing_source).resolve() == source.resolve(), "Run workers from the frozen source snapshot")
    import aggregate_sparse_smart_cases as cases
    index = cases.load_index(index_path)
    require(index["index_fingerprint"] == value["index_fingerprint"] and index["scope"] == value["scope"]
            and index["publication_mode"] == value["publication_mode"], "Index identity differs from launch")
    require(index.get("allowed_missing_tasks") == value.get("allowed_missing_tasks"),
            "Missing-task policy differs from frozen index")
    chunks = read_json(path.parent / "chunks.json")
    ordered = [row["case_key"] for row in sorted(index["cases"], key=lambda row: (row["run_root"], row["case_key"]))]
    expected = [dict(chunk_id=i // value["cases_per_chunk"], case_keys=ordered[i:i + value["cases_per_chunk"]])
                for i in range(0, len(ordered), value["cases_per_chunk"])]
    require(chunks == expected and len(chunks) == value["chunk_count"] and len(ordered) == value["case_count"],
            "Chunks do not own the precise case roster")
    resources = value["resources"]
    require(all(resources[name] == 1 for name in ("nodes", "ntasks", "cpus_per_task")),
            "Each aggregation element must request exactly one CPU")
    return value, chunks


def sbatch_arguments(launch_path, launch, *, summary=False, array_job_id=None):
    resources = launch["resources"]
    source = Path(launch["source_root"])
    script = source / "hpc/discovery" / ("case_summary.sbatch" if summary else "case_aggregation.sbatch")
    logname = "summary-%j" if summary else "case-%A_%a"
    # Slurm expands filename patterns after argv parsing. Escape only literal
    # directory percent signs; the filename's job/array substitutions stay live.
    log_directory = str(Path(launch_path).parent / "logs").replace("%", "%%")
    arguments = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
        f"--account={resources['account']}", f"--partition={resources['partition']}",
        f"--mem={resources['memory']}", f"--time={resources['time_limit']}",
        f"--job-name=smart-case-{'summary' if summary else 'audit'}",
        f"--output={log_directory}/{logname}.out",
        f"--error={log_directory}/{logname}.err"]
    if summary:
        require(array_job_id is not None and JOB.fullmatch(str(array_job_id)), "Invalid array job ID")
        arguments.append(f"--dependency=afterany:{array_job_id}")
    else:
        arguments.append(f"--array=0-{launch['chunk_count'] - 1}%{resources['concurrency']}")
    return arguments + [str(script), str(source), str(Path(launch_path).absolute()), sha(launch_path)]


def _submit_once(path, launch, stage, receipt, *, array_job_id=None):
    require(stage not in receipt["stages"], f"{stage} submission already attempted; reconcile the receipt before further action")
    argv = sbatch_arguments(path, launch, summary=stage == "summary", array_job_id=array_job_id)
    entry = dict(state="intent", argv=argv, started_utc=datetime.now(timezone.utc).isoformat())
    receipt["stages"][stage] = entry
    receipt_path = path.parent / "submission.json"
    # Persist before invoking Slurm. A crash/ambiguous response must never cause
    # an automatic duplicate array or summary submission.
    atomic_json(receipt, receipt_path)
    try:
        result = subprocess.run(argv, check=False, text=True, capture_output=True)
        entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        match = JOB.fullmatch(result.stdout.strip())
        if result.returncode != 0 or match is None:
            entry["state"] = "uncertain"
            raise RuntimeError(f"{stage} sbatch outcome needs manual reconciliation; see {receipt_path}")
        entry.update(state="submitted", job_id=match.group(1))
        return entry["job_id"]
    except OSError as error:
        entry.update(state="uncertain", error=str(error))
        raise RuntimeError(f"{stage} submission interrupted; reconcile {receipt_path}") from error
    finally:
        atomic_json(receipt, receipt_path)


def submit_launch(path, *, summary_only=False):
    path = Path(path).absolute()
    launch, _ = verify_launch(path)
    with lock(path.parent / ".submission.lock"):
        receipt_path = path.parent / "submission.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            require(receipt.get("launch_sha256") == sha(path), "Submission belongs to a different launch")
        else:
            receipt = dict(schema_version=1, launch_sha256=sha(path), stages={})
        if summary_only:
            array = receipt["stages"].get("array", {})
            require(array.get("state") == "submitted", "Summary recovery requires a known submitted array")
            array_id = array["job_id"]
        else:
            require(not receipt["stages"], "Submission already attempted; no automatic resubmission")
            array_id = _submit_once(path, launch, "array", receipt)
        _submit_once(path, launch, "summary", receipt, array_job_id=array_id)
        return receipt


def run_chunk(path, chunk_id, expected_sha, *, executing_source=ROOT):
    launch, chunks = verify_launch(path, expected_sha, executing_source=executing_source)
    require(type(chunk_id) is int and 0 <= chunk_id < len(chunks), "Invalid chunk ID")
    import aggregate_sparse_smart_cases as cases
    reports = cases.aggregate_cases(launch["index"], workers=1, case_keys=chunks[chunk_id]["case_keys"])
    require([row["case_key"] for row in reports] == chunks[chunk_id]["case_keys"],
            "Case API returned incomplete or reordered chunk coverage")
    def execution_passed(row):
        execution = row.get("execution", {})
        if execution.get("fit_success") is True and execution.get("launch_success") is True:
            return True
        expected = sorted(item["task_id"] for item in launch.get("allowed_missing_tasks", [])
                          if item["case_key"] == row["case_key"])
        reported = execution.get("allowed_missing_task_ids")
        return (bool(expected) and execution.get("available_tasks_success") is True
                and isinstance(reported, list) and bool(reported)
                and all(isinstance(item, str) for item in reported)
                and len(set(reported)) == len(reported) and set(reported) <= set(expected))

    failed = [row["case_key"] for row in reports if row["scientific"]["audit_passed"] is not True
              or not execution_passed(row)]
    report = dict(chunk_id=chunk_id, launch_sha256=expected_sha, completed_utc=datetime.now(timezone.utc).isoformat(),
                  reports=reports, failed_cases=failed, job_id=os.environ.get("SLURM_JOB_ID"),
                  array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"))
    atomic_json(report, Path(path).parent / "reports" / f"chunk-{chunk_id}-{uuid4().hex}.json")
    print(json.dumps(dict(chunk_id=chunk_id, cases=len(reports), failed_cases=failed)))
    return int(bool(failed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    roots = prepare.add_mutually_exclusive_group(required=True)
    roots.add_argument("--run-roots", nargs="+", type=Path)
    roots.add_argument("--campaign-submission", type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--models", nargs="+", type=int, choices=range(3))
    prepare.add_argument("--allowed-missing-tasks", type=Path,
                         help="JSON policy listing explicitly permitted cancelled task IDs and evidence")
    prepare.add_argument("--publication-mode", choices=("compact", "full"), default="compact")
    prepare.add_argument("--cases-per-chunk", type=int, default=25)
    prepare.add_argument("--concurrency", type=int, default=4)
    prepare.add_argument("--mem", default="8G", help="Provisional memory per one-CPU element; pilot first")
    prepare.add_argument("--time", default="04:00:00", help="Provisional time per chunk and summary")
    prepare.add_argument("--account", default="mkolar_1314")
    prepare.add_argument("--partition", default="main")
    mode = prepare.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Freeze metadata/code and print requests; no Slurm or numerical work")
    mode.add_argument("--submit", action="store_true")
    for name in ("submit", "submit-summary", "verify", "worker", "summary"):
        command = commands.add_parser(name)
        command.add_argument("--launch", required=True, type=Path)
        if name in ("verify", "worker", "summary"):
            command.add_argument("--expected-sha", required=True)
        if name == "worker":
            command.add_argument("--chunk-id", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            path = prepare_launch(resolve_run_roots(args.run_roots, args.campaign_submission), args.output_root,
                models=args.models, publication_mode=args.publication_mode, cases_per_chunk=args.cases_per_chunk,
                concurrency=args.concurrency, memory=args.mem, time_limit=args.time,
                account=args.account, partition=args.partition,
                allowed_missing_tasks=(None if args.allowed_missing_tasks is None
                                       else read_json(args.allowed_missing_tasks)))
            launch, _ = verify_launch(path)
            print(json.dumps(dict(launch=str(path), case_count=launch["case_count"], chunk_count=launch["chunk_count"],
                resources=launch["resources"], no_audits_performed=True, no_fits_performed=True,
                array_argv=sbatch_arguments(path, launch), summary_dependency="afterany:<array-job-id>"), indent=2))
            if args.submit:
                print(json.dumps(submit_launch(path), indent=2))
            return 0
        if args.command in ("submit", "submit-summary"):
            print(json.dumps(submit_launch(args.launch, summary_only=args.command == "submit-summary"), indent=2))
            return 0
        if args.command == "worker":
            return run_chunk(args.launch, args.chunk_id, args.expected_sha)
        launch, _ = verify_launch(args.launch, args.expected_sha, executing_source=ROOT)
        if args.command == "verify":
            print(json.dumps(dict(verified=True, case_count=launch["case_count"])))
            return 0
        import summarize_sparse_smart_campaign as summary
        return summary.main(["--index", launch["index"], "--output-root", str(Path(launch["output_root"]) / "summary")])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Case aggregation launcher failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
