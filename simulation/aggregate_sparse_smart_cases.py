"""Independent, restartable case audits of existing SparseSMART fit outputs.

Preparation and dry-run use only the standard library. Numerical work happens
one case at a time; no estimator or scheduler is called. Frozen fit provenance
and current analysis provenance are deliberately recorded separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import inspect
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile

import discovery_budget_study as study

METHOD = "SparseSMARTCaseAggregation"
HERE = Path(__file__).resolve().parent
CASE_RE = re.compile(r"m[0-2]_e[0-3]_s(?:0|[1-9][0-9]?)_k[0-6]\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
MARKERS = ("process-exit-code.txt", "exit-code.txt", "launcher-exit-code.txt", "archive-status.txt")


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _safe(path, root):
    """Reject symlinks, including parents below the declared trusted root."""
    path, root = Path(path).absolute(), Path(root).resolve()
    study.require(path.is_relative_to(root), f"Path escapes declared root: {path}")
    for item in (path, *path.parents):
        if item == root:
            break
        study.require(not item.is_symlink(), f"Symlink in artifact path: {item}")
    return path


def _stamp(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_plan(root):
    root = Path(root).resolve()
    for name in ("study-plan.json", "work-items.tsv"):
        _safe(root / name, root)
    return _cached_plan(str(root), _stamp(root / "study-plan.json"), _stamp(root / "work-items.tsv"))


@lru_cache(maxsize=2)
def _cached_plan(root, plan_stamp, table_stamp):
    root = Path(root)
    path, table = (_safe(root / name, root) for name in ("study-plan.json", "work-items.tsv"))
    plan = study.read_json(path)
    study.validate_plan(plan)
    with table.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    tasks = study.planned_tasks(plan)
    study.require(rows == [[str(task[key]) for key in study.TABLE_FIELDS] for task in tasks],
                  "Work-item table differs from study plan")
    return plan, _sha(path), _sha(table)


def _candidate_count(entry):
    config = entry["configuration"]
    return len(config["init_penalties"]) * len(config["penalties_u"]) * len(config["penalties_v"])


def _policy_order(item):
    return item["run_root"], item["case_key"], item["task_id"]


def _validate_cancellation_reference(item):
    reference = item["cancellation_evidence"]
    study.require(isinstance(reference, dict) and set(reference) == {
        "path", "sha256", "job_id", "step_id", "task_id"}, "Invalid cancellation evidence reference")
    study.require(isinstance(reference["path"], str) and Path(reference["path"]).is_absolute()
                  and isinstance(reference["sha256"], str) and SHA_RE.fullmatch(reference["sha256"])
                  and isinstance(reference["job_id"], str) and re.fullmatch(r"[1-9][0-9]*", reference["job_id"])
                  and isinstance(reference["step_id"], str)
                  and re.fullmatch(re.escape(reference["job_id"]) + r"\.[0-9]+", reference["step_id"])
                  and reference["task_id"] == item["task_id"], "Invalid cancellation evidence identity")


def _validate_missing_policy(index):
    """Bind explicit omissions to the declared case roster and original grid."""
    if index["schema_version"] != 3:
        study.require("allowed_missing_tasks" not in index, "Missing-task policy requires schema version 3")
        return []
    policy = index.get("allowed_missing_tasks")
    study.require(isinstance(policy, list) and policy, "Missing-task policy must be a nonempty list")
    entries = {entry["case_key"]: entry for entry in index["cases"]}
    seen, candidate_ids = set(), set()
    for item in policy:
        common = {"run_root", "case_key", "task_id", "grid_candidate_ids", "reason"}
        study.require(isinstance(item, dict) and set(item) in (
            common | {"launcher_exit_code"}, common | {"cancellation_evidence"}),
            "Invalid missing-task policy entry")
        entry = entries.get(item["case_key"])
        study.require(entry is not None and item["run_root"] == entry["run_root"]
                      and Path(item["run_root"]).is_absolute()
                      and entry["cell"]["inapplicability_reason"] is None,
                      "Missing-task policy refers to an unknown or inapplicable case")
        study.require(isinstance(item["task_id"], str)
                      and re.fullmatch(re.escape(item["case_key"]) + r"_g(?:0|[1-9][0-9]*)", item["task_id"]),
                      "Missing-task policy must identify a tuning shard")
        study.require(isinstance(item["reason"], str) and item["reason"].strip(),
                      "Missing-task policy needs a reason")
        if "cancellation_evidence" in item:
            _validate_cancellation_reference(item)
        else:
            study.require(type(item["launcher_exit_code"]) is int and 0 < item["launcher_exit_code"] <= 255,
                          "Missing-task policy needs a nonzero launcher exit code")
        identity = (item["run_root"], item["task_id"])
        study.require(identity not in seen, "Duplicate missing-task permission")
        seen.add(identity)
        ids = item["grid_candidate_ids"]
        study.require(isinstance(ids, list) and ids and all(type(j) is int and 0 <= j < _candidate_count(entry)
                      for j in ids) and ids == sorted(set(ids)), "Invalid missing-task candidate IDs")
        for j in ids:
            key = (item["case_key"], j)
            study.require(key not in candidate_ids, "Overlapping missing-task candidate permissions")
            candidate_ids.add(key)
    study.require(policy == sorted(policy, key=_policy_order), "Missing-task policy must be canonical")
    for key, entry in entries.items():
        study.require(sum(case == key for case, _ in candidate_ids) < _candidate_count(entry),
                      "Missing-task policy cannot omit an entire tuning grid")
    return policy


def _normalize_missing_policy(value, entries):
    study.require(isinstance(value, list) and value, "Missing-task policy must be a nonempty list")
    allowed_roots = {entry["run_root"] for entry in entries}
    task_maps, result = {}, []
    for item in value:
        common = {"run_root", "task_id", "reason"}
        study.require(isinstance(item, dict) and set(item) in tuple(
            common | identity | {evidence}
            for identity in (set(), {"case_key", "grid_candidate_ids"})
            for evidence in ("launcher_exit_code", "cancellation_evidence")),
            "Invalid missing-task policy entry")
        study.require(isinstance(item["run_root"], str) and Path(item["run_root"]).is_absolute(),
                      "Missing-task run root must be absolute")
        root = str(Path(item["run_root"]).resolve())
        study.require(root in allowed_roots, "Missing-task policy root is outside selected cases")
        if root not in task_maps:
            plan, _, _ = _read_plan(root)
            task_maps[root] = {task["task_id"]: task for task in study.planned_tasks(plan)}
        task = task_maps[root].get(item["task_id"])
        study.require(task is not None and "grid_candidate_ids" in task,
                      "Missing-task policy must identify a planned tuning shard")
        normalized = dict(run_root=root, case_key=task["cell_task_id"], task_id=task["task_id"],
            grid_candidate_ids=task["grid_candidate_ids"], reason=item["reason"])
        evidence_key = "cancellation_evidence" if "cancellation_evidence" in item else "launcher_exit_code"
        normalized[evidence_key] = item[evidence_key]
        if "case_key" in item:
            study.require(item["case_key"] == normalized["case_key"]
                          and item["grid_candidate_ids"] == normalized["grid_candidate_ids"],
                          "Missing-task policy differs from the planned shard")
        result.append(normalized)
    result.sort(key=_policy_order)
    _validate_missing_policy(dict(schema_version=3, cases=entries, allowed_missing_tasks=result))
    return result


def prepare_campaign(run_roots, output_root, *, models=None, publication_mode="full", allowed_missing_tasks=None):
    """Freeze an explicit, nonoverlapping case roster without reading fit files."""
    study.require(publication_mode in ("full", "compact"), "Invalid publication mode")
    selected_models = None if models is None else list(models)
    if selected_models is not None:
        study.require(selected_models and all(type(model) is int and model in range(3)
                      for model in selected_models) and len(set(selected_models)) == len(selected_models),
                      "Models must be unique zero-based IDs 0, 1, or 2")
        selected_models.sort()
    roots = [Path(root).resolve() for root in run_roots]
    study.require(roots and len(set(roots)) == len(roots), "Require unique nonempty run roots")
    output = Path(output_root).resolve()
    # An analysis tree must never replace or be placed inside a raw run tree.
    study.require(all(output != root and not output.is_relative_to(root)
                      and not root.is_relative_to(output) for root in roots),
                  "Aggregation output must be separate from raw run roots")
    entries, seen = [], set()
    for root in sorted(roots):
        plan, plan_sha, table_sha = _read_plan(root)
        for cell in plan["cells"]:
            if selected_models is not None and cell["model"] not in selected_models:
                continue
            key = cell["task_id"]
            study.require(CASE_RE.fullmatch(key) and key not in seen,
                          f"Duplicate or invalid scientific case: {key}")
            seen.add(key)
            source = dict(scheme=plan["source"]["scheme"], implementation={
                k: dict(fingerprint=v["fingerprint"]) for k, v in plan["source"]["implementation"].items()})
            entries.append(dict(case_key=key, run_root=str(root), plan_fingerprint=plan["plan_fingerprint"],
                plan_sha256=plan_sha, work_table_sha256=table_sha, cell=cell,
                configuration=plan["configuration"], seed_file_sha256=plan["seed_file_sha256"], source=source))
    present_models = sorted({entry["cell"]["model"] for entry in entries})
    study.require(entries and (selected_models is None or selected_models == present_models),
                  "Selected model scope has no cases for one or more requested models")
    index = dict(schema_version=2, method=METHOD, scope=dict(models=present_models),
                 publication_mode=publication_mode, cases=sorted(entries, key=lambda row: row["case_key"]))
    if allowed_missing_tasks is not None:
        index.update(schema_version=3, allowed_missing_tasks=_normalize_missing_policy(allowed_missing_tasks, entries))
    index["index_fingerprint"] = study.digest(index)
    output.mkdir(parents=True, exist_ok=True)
    with _lock(output / ".prepare.lock"):
        path = _safe(output / "campaign-index.json", output)
        if path.exists():
            study.require(study.read_json(path) == index,
                          "Existing campaign index differs; use a fresh aggregation output root")
        else:
            study.atomic_json(index, path)
    return index


def load_index(index_path):
    path = Path(index_path).absolute()
    study.require(not path.is_symlink(), "Campaign index must not be a symlink")
    return _cached_index(str(path), _stamp(path))


@lru_cache(maxsize=1)
def _cached_index(path, stamp):
    value = study.read_json(Path(path))
    study.require(type(value.get("schema_version")) is int and value["schema_version"] in (1, 2, 3)
                  and value.get("method") == METHOD,
                  "Unsupported case aggregation index")
    study.require(study.digest({k: v for k, v in value.items() if k != "index_fingerprint"})
                  == value.get("index_fingerprint"), "Campaign index fingerprint mismatch")
    rows = value.get("cases")
    study.require(isinstance(rows, list) and rows, "Campaign index has no cases")
    keys = [row["case_key"] for row in rows]
    study.require(len(set(keys)) == len(keys) and all(CASE_RE.fullmatch(key) for key in keys),
                  "Duplicate or invalid case keys")
    if value["schema_version"] >= 2:
        scope = value.get("scope")
        study.require(isinstance(scope, dict) and set(scope) == {"models"}, "Invalid model scope")
        models = scope["models"]
        study.require(isinstance(models, list) and models and all(type(model) is int and model in range(3)
                      for model in models) and models == sorted(set(models)), "Invalid model scope")
        study.require(models == sorted({row["cell"]["model"] for row in rows})
                      and all(row["case_key"] == row["cell"]["task_id"]
                              and row["case_key"].startswith(f"m{row['cell']['model']}_") for row in rows),
                      "Case identities differ from declared model scope")
        study.require(value.get("publication_mode") in ("full", "compact"), "Invalid publication mode")
    else:
        study.require("publication_mode" not in value and "scope" not in value,
                      "Legacy index cannot declare a scoped publication mode")
    _validate_missing_policy(value)
    return value


@lru_cache(maxsize=512)
def _source_sha(path, stamp):
    return _sha(path)


def _analysis_sha(path):
    return _source_sha(str(path), _stamp(path))


def analysis_provenance(generate_data_fn=None):
    """Content identities for the analysis implementation, generator and runtime."""
    names = set(study.SOURCE_FILES) | {
        "discovery_budget_study.py", "aggregate_sparse_smart_cases.py", "summarize_sparse_smart_budget_study.py"}
    files = {f"simulation/{name}": _analysis_sha(HERE / name) for name in sorted(names)}
    for directory, prefix in ((HERE.parent / "smart" / "smart", "generator/smart"),
                               (HERE.parent / "sparse-smart" / "src" / "sparse_smart", "sparse_smart")):
        for path in sorted(directory.rglob("*.py")):
            files[f"{prefix}/{path.relative_to(directory).as_posix()}"] = _analysis_sha(path)
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    value = dict(files=files, runtime=dict(python=platform.python_version(),
                 platform=platform.platform(), packages=packages), numerical_threads=1)
    if generate_data_fn is not None:
        path = inspect.getsourcefile(generate_data_fn)
        study.require(path is not None, "Injected generator needs inspectable source")
        value["test_generator"] = dict(name=generate_data_fn.__qualname__, sha256=_sha(path))
    return value


@contextmanager
def _lock(path):
    path = Path(path)
    _safe(path, path.parent.resolve())
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Aggregation already active: {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _parse_result(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            study.require(key not in value, f"Duplicate scientific JSON key: {key}")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON value: {value}")))


def _file_evidence(path, root, *, captured=None):
    path = _safe(path, root)
    if not path.exists():
        return dict(path=path.relative_to(root).as_posix(), present=False)
    study.require(path.is_file(), f"Artifact is not a regular file: {path}")
    if captured is None:
        sha, size = _sha(path), path.stat().st_size
    else:
        # Hash and parse the same bytes. The final snapshot below still performs
        # a complete content rehash; file timestamps never substitute for it.
        raw = path.read_bytes()
        sha, size = hashlib.sha256(raw).hexdigest(), len(raw)
        try:
            record, error = _parse_result(raw), None
        except (ValueError, UnicodeError) as failure:
            record, error = None, str(failure)
        captured[str(path)] = (record, sha, error)
    return dict(path=path.relative_to(root).as_posix(), present=True, sha256=sha, size=size)


def _snapshot(root, tasks, *, captured=None, allowed_missing=()):
    """Fingerprint exact inputs, including absence and all discovered results."""
    evidence = []
    cancellation = {item["task_id"]: item["cancellation_evidence"]
                    for item in allowed_missing if "cancellation_evidence" in item}
    for task in tasks:
        base = _safe(root / "tasks" / task["task_id"], root)
        results = _safe(base / "results", root)
        paths = [base / name for name in MARKERS]
        canonical = results / "budget_study_manifest.json"
        paths.append(canonical)
        attempts = _safe(results / "budget_study_manifest_attempts", root)
        paths.extend(sorted(attempts.glob("*.json")))
        expected = results / study.relative_result(task)
        paths.append(expected)
        paths.extend(sorted(results.rglob("BudgetStudy_result_*.json")))
        if task["task_id"] in cancellation:
            paths.append(base / "environment.txt")
        row = dict(task_id=task["task_id"], files=[
            _file_evidence(path, root, captured=captured if path == expected else None)
            for path in sorted(set(paths))])
        if task["task_id"] in cancellation:
            path = Path(cancellation[task["task_id"]]["path"])
            row["cancellation_evidence"] = dict(
                _file_evidence(path, path.parent.resolve()), path=str(path))
        evidence.append(row)
    return evidence


def _execution(root, tasks):
    outcomes, issues, archive_only = [], [], []
    fit_ok, launch_ok = True, True
    for task in tasks:
        base = root / "tasks" / task["task_id"]
        codes = {}
        for name in MARKERS[:3]:
            path = base / name
            try:
                token = path.read_text().strip()
                study.require(re.fullmatch(r"[0-9]+", token), "Invalid exit marker")
                codes[name] = int(token)
            except (OSError, ValueError):
                codes[name] = None
        archive_path = base / "archive-status.txt"
        archive = archive_path.read_text().strip() if archive_path.is_file() else "unknown"
        process, worker, launch = (codes[name] for name in MARKERS[:3])
        # Both forms were observed in legacy dispatch/worker archive traps.
        # A code 74 alone never excuses an unsuccessful or unknown fit process.
        only_archive = process == 0 and worker in (0, 74) and launch == 74 and archive == "failed"
        if only_archive:
            archive_only.append(task["task_id"])
        else:
            if process != 0:
                fit_ok = False
                issues.append(dict(task_id=task["task_id"], kind="fit_execution", exit_code=process))
            if worker != 0 or launch != 0:
                launch_ok = False
                issues.append(dict(task_id=task["task_id"], kind="worker_or_launch_execution",
                                   worker_exit_code=worker, launcher_exit_code=launch))
        outcomes.append(dict(task_id=task["task_id"], exit_codes=codes, archive_status=archive,
                             archive_only_failure=only_archive))
    statuses = Counter(row["archive_status"] for row in outcomes)
    return dict(fit_success=fit_ok, launch_success=launch_ok,
        archive_status=next(iter(statuses)) if len(statuses) == 1 else "mixed",
        archive_status_counts=dict(statuses), archive_only_failures=archive_only, issues=issues,
        task_outcomes=outcomes)


def _verify_slurm_cancellation(root, base, item):
    """Verify frozen external accounting when cancellation prevented every exit trap."""
    reference = item["cancellation_evidence"]
    path = _safe(Path(reference["path"]), Path(reference["path"]).parent.resolve())
    raw = path.read_bytes()
    study.require(hashlib.sha256(raw).hexdigest() == reference["sha256"],
                  "Cancellation evidence SHA256 mismatch")
    evidence = _parse_result(raw)
    study.require(isinstance(evidence, dict) and evidence.get("schema_version") == 1
                  and evidence.get("kind") == "slurm_cancelled_task"
                  and evidence.get("run_root") == str(root)
                  and all(evidence.get(key) == reference[key] for key in ("job_id", "step_id", "task_id")),
                  "Cancellation evidence does not match the planned task")
    accounting = evidence.get("sacct")
    study.require(isinstance(accounting, dict), "Missing Slurm cancellation accounting")
    for kind in ("job", "step"):
        record = accounting.get(kind)
        study.require(isinstance(record, dict) and record.get("JobIDRaw") == reference[f"{kind}_id"]
                      and isinstance(record.get("State"), str)
                      and re.fullmatch(r"CANCELLED(?: by [0-9]+)?", record["State"])
                      and isinstance(record.get("ExitCode"), str)
                      and re.fullmatch(r"[0-9]+:[0-9]+", record["ExitCode"]),
                      "Slurm job and step must both have matching cancellation records")
    environment = _safe(base / "environment.txt", root).read_bytes()
    study.require(hashlib.sha256(environment).hexdigest() == evidence.get("environment_sha256"),
                  "Cancelled task environment SHA256 mismatch")
    lines = environment.decode("utf-8").splitlines()
    expected = {"Job": {reference["job_id"]}, "Step": {
        reference["step_id"], reference["step_id"].split(".", 1)[1]}, "Task": {item["task_id"]}}
    for key, values in expected.items():
        found = [line[len(key) + 1:].strip() for line in lines if line.startswith(key + ":")]
        study.require(len(found) == 1 and found[0] in values,
                      "Cancelled task environment identity mismatch")
    study.require(all(not _safe(base / name, root).exists() for name in MARKERS[:3]),
                  "External cancellation evidence requires all task exit markers to be absent")


def _unavailable_tasks(root, tasks, policy, plan):
    """Consume a permission only for an absent result with matching cancellation evidence.

    A present result is always audited normally, even if permission exists. A
    corrupt file, success manifest, or different exit code is never excused.
    """
    lookup = {task["task_id"]: task for task in tasks}
    unavailable = []
    for item in policy:
        task = lookup.get(item["task_id"])
        study.require(task is not None and task.get("grid_candidate_ids") == item["grid_candidate_ids"]
                      and task.get("cell_task_id") == item["case_key"],
                      "Missing-task permission differs from the frozen task")
        base = _safe(root / "tasks" / task["task_id"], root)
        results = _safe(base / "results", root)
        expected = _safe(results / study.relative_result(task), root)
        if expected.exists() or list(results.rglob("BudgetStudy_result_*.json")):
            continue
        if "cancellation_evidence" in item:
            _verify_slurm_cancellation(root, base, item)
        else:
            for name in MARKERS[:3]:
                path = _safe(base / name, root)
                if not path.exists():
                    study.require(name != "launcher-exit-code.txt", "Missing cancellation launcher evidence")
                    continue
                token = path.read_text().strip()
                study.require(re.fullmatch(r"[0-9]+", token) and int(token) == item["launcher_exit_code"],
                              "Missing-task cancellation exit marker mismatch")
        archive_path = _safe(base / "archive-status.txt", root)
        study.require(not archive_path.exists() or archive_path.read_text().strip() in ("deferred", "unknown"),
                      "Missing-task cancellation archive marker mismatch")
        manifests = [results / "budget_study_manifest.json"]
        manifests.extend(sorted((results / "budget_study_manifest_attempts").glob("*.json")))
        local = study.task_plan(task, plan)
        expected_manifest = dict(schema_version=1, method=study.METHOD, models=[task["model"]],
            experiments=[task["experiment"]], seed_ids=[task["seed"]], profile=local["profile"],
            setting_index=task["setting"], expected_cells=1, seed_file_sha256=local["seed_file_sha256"],
            configuration=local["configuration"])
        expected_cell = dict(model=f"model{task['model']+1}", experiment=f"exp{task['experiment']+1}",
                             setting=task["simulation_setting"]["suffix"], seed_id=task["seed"])
        for path in manifests:
            _safe(path, root)
            if not path.exists():
                continue
            manifest = _parse_result(path.read_bytes())
            study.require(isinstance(manifest, dict)
                          and all(manifest.get(key) == value for key, value in expected_manifest.items())
                          and manifest.get("attempt_status") in ("running", "interrupted")
                          and manifest.get("errors") == []
                          and isinstance(manifest.get("cells"), list) and len(manifest["cells"]) <= 1
                          and all(isinstance(row, dict) and row.get("success") is not True
                                  and row.get("status") not in ("complete", "completed")
                                  and all(row.get(key) == value for key, value in expected_cell.items())
                                  and ("path" not in row or row["path"] == str(expected))
                                  for row in manifest["cells"]),
                          "Missing-task permission requires a valid incomplete task manifest")
        unavailable.append(item)
    return unavailable


def _missing_tuning(entry, allowed, unavailable):
    ids = sorted(j for item in unavailable for j in item["grid_candidate_ids"])
    return dict(allowed_tasks=allowed, unavailable_tasks=unavailable, unavailable_grid_candidate_ids=ids,
                planned_candidate_count=_candidate_count(entry), available_candidate_count=_candidate_count(entry)-len(ids))


def _permitted_execution(execution):
    return bool((execution.get("fit_success") and execution.get("launch_success"))
                or (execution.get("available_tasks_success") is True and execution.get("allowed_missing_task_ids")))


def _collect(root, tasks, plan, *, captured=None):
    records, artifacts, issues = [], [], []
    for task in tasks:
        try:
            local = study.task_plan(task, plan)
            results = root / "tasks" / task["task_id"] / "results"
            manifest, errors = study.inspect_cell_manifest(results, task, local)
            study.require(manifest is not None and not errors, f"Incomplete task manifest: {errors}")
            path = _safe(results / study.relative_result(task), root)
            found = sorted(results.rglob("BudgetStudy_result_*.json"))
            study.require(found == [path], "Missing, unexpected or duplicate scientific result")
            if captured is None:
                raw = path.read_bytes()
                record, sha = _parse_result(raw), hashlib.sha256(raw).hexdigest()
            else:
                study.require(str(path) in captured, "Scientific input was absent in initial snapshot")
                record, sha, error = captured[str(path)]
                study.require(error is None, f"Invalid scientific JSON: {error}")
            study.validate_record_identity(record, task, local)
            manifest_row = study.read_json(Path(manifest["path"]))["cells"][0]
            study.require(manifest_row.get("path") == str(path)
                          and manifest_row.get("status") == record["status"]
                          and type(manifest_row.get("success")) is bool
                          and manifest_row["success"] == record["success"],
                          "Manifest result path/status/success mismatch")
            records.append(record)
            artifact = dict(task_id=task["task_id"], source_relative=path.relative_to(root).as_posix(),
                            sha256=sha, cell_manifest=manifest)
            if "grid_candidate_ids" in task:
                artifact["grid_candidate_ids"] = task["grid_candidate_ids"]
            artifacts.append(artifact)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(task_id=task["task_id"], kind="invalid_or_missing_input", message=str(error)))
    return records, artifacts, issues


def _paper(record, *, selected_factors=None):
    """Scalar tuning diagnostics plus retained winners, without duplicate factors."""
    selected = []
    for cap in record["cap_outcomes"]:
        row = dict(iteration_budget=cap["iteration_budget"], success=cap["success"],
                   coverage_complete=cap["coverage_complete"])
        if cap["success"]:
            candidate = record["selection_history"][cap["winner_candidate_id"]]
            row.update(params=candidate["params"], grid_candidate_id=cap["winner_grid_candidate_id"],
                selected_iteration=cap["selected_iteration"], selected_factor_key=cap["selected_factor_key"],
                validation_mse=cap["validation_mse"], coefficient_error=cap["coefficient_error"])
            if selected_factors is not None:
                row["selected_factor_id"] = f"g{cap['winner_grid_candidate_id']}_i{cap['selected_factor_key']}"
        selected.append(row)
    return dict(selected=selected, candidate_grid=record["configuration"]["candidate_grid"],
        selection_history=record["selection_history"],
        trajectories=[{key: t[key] for key in ("grid_candidate_id", "params", "status", "success", "n_iter",
            "termination_reason", "validation_history") if key in t} for t in record["trajectories"]],
        fit_time_sec=record.get("fit_time_sec"), elapsed_time_sec=record.get("elapsed_time_sec"),
        time_scope="sum of tuning shard work; not allocation wall time",
        factor_location=("selected-factors.json retains distinct selected states; all other factors remain in raw inputs"
                         if selected_factors is not None else
                         "merged.json retains all original factor states; selected keys are references only"))


def _selected_factors(record, *, entry, index, generation, artifacts):
    """Export already-audited winners once, with exact links to raw shards.

    Selection and numerical auditing are unchanged. This is a publication
    format, not a shortened record pretending to satisfy the full-fit schema.
    """
    sources = {}
    shard_sources = record.get("tuning_shard_merge", {}).get("source_artifacts")
    if shard_sources is not None:
        for artifact in shard_sources:
            for local_id, global_id in artifact["local_to_global_grid_ids"].items():
                study.require(global_id not in sources, "Duplicate selected-factor source mapping")
                sources[global_id] = (artifact, int(local_id))
    else:
        study.require(len(artifacts) == 1, "Unsplit factor export requires one raw source")
        sources = {t["grid_candidate_id"]: (artifacts[0], t["grid_candidate_id"])
                   for t in record["trajectories"]}
    trajectories = {t["grid_candidate_id"]: t for t in record["trajectories"]}
    factors, selections = {}, []
    for cap in record["cap_outcomes"]:
        selected = dict(iteration_budget=cap["iteration_budget"], success=cap["success"],
                        coverage_complete=cap["coverage_complete"], factor_id=None)
        if cap["success"]:
            grid, key = cap["winner_grid_candidate_id"], cap["selected_factor_key"]
            trajectory = trajectories[grid]
            state = trajectory["factor_states"][key]
            study.require(str(state["iteration"]) == key and state["iteration"] == cap["selected_iteration"],
                          "Selected factor identity differs from audited winner")
            factor_id = f"g{grid}_i{key}"
            if factor_id not in factors:
                artifact, local_id = sources[grid]
                factors[factor_id] = dict(grid_candidate_id=grid, params=trajectory["params"], factor_key=key,
                    selected_iteration=state["iteration"], state=state,
                    source=dict(run_root=entry["run_root"], source_relative=artifact["source_relative"],
                                sha256=artifact["sha256"], task_id=artifact["task_id"],
                                local_grid_candidate_id=local_id))
            selected["factor_id"] = factor_id
        selections.append(selected)
    return dict(schema_version=1, method="SparseSMARTSelectedFactors", case_key=entry["case_key"],
        generation=generation, index_fingerprint=index["index_fingerprint"],
        plan_fingerprint=entry["plan_fingerprint"], identity=entry["cell"],
        factor_encoding=record.get("factor_encoding", "C_hat=(left*singular_values)@right.T"),
        factors=factors, selections=selections, no_fits_performed=True)


def _read_generation(directory, *, generation, entry, index, input_fingerprint, analysis_fingerprint,
                     missing_tuning=None, execution=None):
    receipt_path = _safe(directory / "receipt.json", directory.parent.parent)
    receipt = study.read_json(receipt_path)
    version, mode = index["schema_version"], index.get("publication_mode", "full")
    expected = dict(schema_version=version, generation=generation, case_key=entry["case_key"],
        index_fingerprint=index["index_fingerprint"], plan_fingerprint=entry["plan_fingerprint"],
                    input_fingerprint=input_fingerprint, analysis_fingerprint=analysis_fingerprint)
    if version >= 2:
        expected["publication_mode"] = mode
    else:
        study.require("publication_mode" not in receipt, "Legacy receipt cannot declare a publication mode")
    study.require(all(receipt.get(k) == v for k, v in expected.items()), "Existing generation receipt mismatch")
    study.require(study.digest(receipt["inputs"]) == input_fingerprint
                  and study.digest(receipt["analysis"]) == analysis_fingerprint,
                  "Published receipt provenance payload mismatch")
    roster = {"audit", "merged", "selected_factors"} if version >= 2 else {"audit", "merged"}
    study.require(set(receipt["outputs"]) == roster
                  and isinstance(receipt["outputs"]["audit"], dict), "Invalid published output roster")
    for name, artifact in receipt["outputs"].items():
        if artifact is None:
            continue
        filename = "selected-factors.json" if name == "selected_factors" else f"{name}.json"
        study.require(name in roster and artifact["path"] == filename,
                      "Unexpected generation artifact")
        path = _safe(directory / artifact["path"], directory.parent.parent)
        study.require(_sha(path) == artifact["sha256"], f"Published {name} hash mismatch")
    audit = study.read_json(directory / "audit.json")
    study.require(version >= 2 or "publication_mode" not in audit,
                  "Legacy audit cannot declare a publication mode")
    study.require(audit.get("schema_version") == version
                  and (version == 1 or audit.get("publication_mode") == mode)
                  and audit["generation"] == generation and audit["identity"] == entry["cell"]
                  and audit["configuration"] == entry["configuration"]
                  and audit["case_key"] == entry["case_key"]
                  and audit["index_fingerprint"] == index["index_fingerprint"]
                  and audit["plan_fingerprint"] == entry["plan_fingerprint"], "Published audit identity mismatch")
    output_key = "selected_factors" if mode == "compact" else "merged"
    study.require((receipt["outputs"][output_key] is not None) == audit["scientific"]["audit_passed"]
                  and (version == 1 or receipt["outputs"]["merged" if mode == "compact" else "selected_factors"] is None),
                  "Published scientific output presence mismatch")
    if version == 3:
        study.require(audit.get("missing_tuning") == missing_tuning and audit["execution"] == execution,
                      "Published missing-task policy or execution evidence mismatch")
    else:
        study.require("missing_tuning" not in audit and not {"allowed_missing_task_ids", "available_tasks_success"}
                      .intersection(audit["execution"]), "Legacy audit cannot declare missing-task exceptions")
    return audit


def aggregate_case(index_path, case_key, *, generate_data_fn=None):
    """Audit one planned case and atomically publish an immutable generation."""
    index_path = Path(index_path).absolute()
    index = load_index(index_path)
    selected = [row for row in index["cases"] if row["case_key"] == case_key]
    study.require(len(selected) == 1, f"Unknown case key: {case_key}")
    entry = selected[0]
    version, mode = index["schema_version"], index.get("publication_mode", "full")
    root = Path(entry["run_root"]).resolve()
    base = _safe(index_path.parent / "cases" / case_key, index_path.parent.resolve())
    base.mkdir(parents=True, exist_ok=True)
    with _lock(base / ".lock"):
        # Withdraw the mutable pointer while checking a new attempt. Immutable
        # generations remain untouched. Interrupted/failed attempts therefore
        # cannot silently expose a stale success to the campaign reducer.
        current = _safe(base / "current.json", index_path.parent.resolve())
        current.unlink(missing_ok=True)
        plan, plan_sha, table_sha = _read_plan(root)
        study.require(plan_sha == entry["plan_sha256"] and table_sha == entry["work_table_sha256"]
                      and plan["plan_fingerprint"] == entry["plan_fingerprint"], "Frozen plan or work table changed")
        cell = entry["cell"]
        study.require(cell in plan["cells"] and entry["configuration"] == plan["configuration"]
                      and entry["seed_file_sha256"] == plan["seed_file_sha256"], "Case index differs from frozen plan")
        expected_source = dict(scheme=plan["source"]["scheme"], implementation={
            key: dict(fingerprint=value["fingerprint"]) for key, value in plan["source"]["implementation"].items()})
        study.require(entry["source"] == expected_source, "Case source differs from frozen plan")
        tasks = [task for task in study.planned_tasks(plan)
                 if task.get("cell_task_id", task["task_id"]) == case_key]
        study.require(tasks, "Case has no planned work")
        allowed = [item for item in index.get("allowed_missing_tasks", []) if item["case_key"] == case_key]
        captured = {}
        evidence = _snapshot(root, tasks, captured=captured, allowed_missing=allowed)
        input_identity = dict(plan_sha256=plan_sha, work_table_sha256=table_sha,
                              seed_file_sha256=plan["seed_file_sha256"], tasks=evidence)
        input_fingerprint = study.digest(input_identity)
        analysis = analysis_provenance(generate_data_fn)
        analysis_fingerprint = study.digest(analysis)
        generation = study.digest(dict(index_fingerprint=index["index_fingerprint"], case_key=case_key,
            input_fingerprint=input_fingerprint, analysis_fingerprint=analysis_fingerprint))
        generations = _safe(base / "generations", index_path.parent.resolve())
        generations.mkdir(exist_ok=True)
        directory = _safe(generations / generation, index_path.parent.resolve())
        policy_issues, unavailable = [], []
        execution = _execution(root, tasks)
        missing_tuning = None
        if version == 3:
            try:
                unavailable = _unavailable_tasks(root, tasks, allowed, plan)
            except (OSError, ValueError, KeyError, TypeError) as error:
                policy_issues.append(dict(kind="invalid_missing_task_permission", message=str(error)))
            missing_tuning = _missing_tuning(entry, allowed, unavailable)
            missing_ids = {item["task_id"] for item in unavailable}
            execution.update(allowed_missing_task_ids=sorted(missing_ids),
                available_tasks_success=not policy_issues and all(issue.get("task_id") in missing_ids
                                                               for issue in execution["issues"]))
        if directory.exists():
            captured.clear()
            audit = _read_generation(directory, generation=generation, entry=entry, index=index,
                input_fingerprint=input_fingerprint, analysis_fingerprint=analysis_fingerprint,
                missing_tuning=missing_tuning, execution=execution)
            study.require(_snapshot(root, tasks, allowed_missing=allowed) == evidence
                and _sha(root / "study-plan.json") == plan_sha
                and _sha(root / "work-items.tsv") == table_sha
                and analysis_provenance(generate_data_fn) == analysis,
                "Inputs or analysis changed while verifying a published generation")
            study.atomic_json(dict(generation=generation, receipt_sha256=_sha(directory / "receipt.json")), current)
            return dict(audit, resumed=True)
        unavailable_ids = {item["task_id"] for item in unavailable}
        available_tasks = [task for task in tasks if task["task_id"] not in unavailable_ids]
        records, artifacts, issues = _collect(root, available_tasks, plan, captured=captured)
        issues.extend(policy_issues)
        captured.clear()
        merged = summary = paper = selected_factors = None
        if not issues:
            try:
                # Import numerical packages only after preparation/input checks.
                from threadpoolctl import threadpool_limits
                import run_sparse_smart_budget_study as runner
                import summarize_sparse_smart_budget_study as validator
                from merge_sparse_smart_budget_shards import merge_records_with_audit
                generator = generate_data_fn or runner.external_runner.old_runner._load_generator()
                with threadpool_limits(limits=1):
                    if "work_items" in plan and cell["inapplicability_reason"] is None:
                        config_options = dict(plan["configuration"])
                        config_options.setdefault("validation_iterations", [])
                        # An old plan declares the pre-stopping policy. Its
                        # absence must never activate today's runner defaults.
                        config_options.setdefault("validation_interval", config_options["checkpoint_interval"])
                        config_options.setdefault("validation_patience", None)
                        config_options.setdefault("validation_min_iterations", 500)
                        config_options.setdefault("validation_min_relative_improvement", .001)
                        omissions = [dict(grid_candidate_id=j, task_id=item["task_id"], reason=item["reason"])
                                     for item in unavailable for j in item["grid_candidate_ids"]]
                        merged, summary = merge_records_with_audit(records,
                            config=runner.RunnerConfig(**config_options), source_artifacts=artifacts,
                            generate_data_fn=generator, **({"unavailable_candidates": omissions} if omissions else {}))
                        merged["tuning_shard_merge"]["plan_fingerprint"] = plan["plan_fingerprint"]
                    else:
                        study.require(len(records) == 1, "Unsplit case has multiple records")
                        merged = records[0]
                        summary = validator.validate_record(merged, study.relative_result(cell),
                            setting=runner.SimulationSetting(**cell["simulation_setting"]), model_id=cell["model"],
                            exp_id=cell["experiment"], seed_id=cell["seed"], random_seed=cell["random_seed"],
                            data_cache={}, generate_data_fn=generator)
                study.validate_record_identity(merged, cell, plan)
                if mode == "compact":
                    selected_factors = _selected_factors(merged, entry=entry, index=index,
                                                         generation=generation, artifacts=artifacts)
                paper = _paper(merged, selected_factors=selected_factors)
            except (OSError, ValueError, KeyError, TypeError, IndexError, ArithmeticError) as error:
                issues.append(dict(kind="scientific_audit_failure", message=str(error)))
                merged = summary = paper = selected_factors = None
        scientific = dict(audit_passed=not issues,
            status=summary["status"] if summary else "invalid_or_missing",
            budget_coverage_complete=bool(summary and (not summary["applicable"]
                or all(cap["coverage_complete"] for cap in summary["caps"]))), issues=issues)
        audit = dict(schema_version=version, case_key=case_key, generation=generation,
            index_fingerprint=index["index_fingerprint"], plan_fingerprint=plan["plan_fingerprint"],
            identity=cell, configuration=entry["configuration"], execution=execution, scientific=scientific,
            summary=summary, paper=paper, no_fits_performed=True)
        if version >= 2:
            audit["publication_mode"] = mode
        if version == 3:
            audit["missing_tuning"] = missing_tuning
        # Changed inputs cannot produce a receipt claiming an earlier snapshot.
        study.require(_snapshot(root, tasks, allowed_missing=allowed) == evidence,
                      "Inputs changed during case audit; retry after fitting finishes")
        study.require(_sha(root / "study-plan.json") == plan_sha and _sha(root / "work-items.tsv") == table_sha,
                      "Plan changed during case audit")
        study.require(analysis_provenance(generate_data_fn) == analysis, "Analysis code changed during case audit")
        staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=base))
        try:
            study.atomic_json(audit, staging / "audit.json")
            outputs = dict(audit=dict(path="audit.json", sha256=_sha(staging / "audit.json")), merged=None)
            if version >= 2:
                outputs["selected_factors"] = None
            if merged is not None and mode == "full":
                study.atomic_json(merged, staging / "merged.json")
                outputs["merged"] = dict(path="merged.json", sha256=_sha(staging / "merged.json"))
            if selected_factors is not None:
                study.atomic_json(selected_factors, staging / "selected-factors.json")
                outputs["selected_factors"] = dict(path="selected-factors.json",
                                                    sha256=_sha(staging / "selected-factors.json"))
            receipt = dict(schema_version=version, case_key=case_key, generation=generation,
                index_fingerprint=index["index_fingerprint"], plan_fingerprint=plan["plan_fingerprint"],
                input_fingerprint=input_fingerprint, analysis_fingerprint=analysis_fingerprint,
                inputs=input_identity, analysis=analysis, source=plan["source"], source_artifacts=artifacts,
                outputs=outputs, completed_utc=datetime.now(timezone.utc).isoformat())
            if version >= 2:
                receipt["publication_mode"] = mode
            study.atomic_json(receipt, staging / "receipt.json")
            os.rename(staging, directory)
            study.atomic_json(dict(generation=generation, receipt_sha256=_sha(directory / "receipt.json")), base / "current.json")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return dict(audit, resumed=False)


def _run_case(arguments):
    path, key = arguments
    try:
        audit = aggregate_case(path, key)
        # Pool coordination must not retain thousands of full audit histories.
        return dict(case_key=key, generation=audit["generation"], resumed=audit["resumed"],
            scientific={k: audit["scientific"][k] for k in ("audit_passed", "status", "budget_coverage_complete")},
            execution={k: audit["execution"][k] for k in ("fit_success", "launch_success", "archive_status",
                "allowed_missing_task_ids", "available_tasks_success") if k in audit["execution"]})
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        # Independent cases continue; no stale pointer is treated as this success.
        return dict(case_key=key, error=str(error), scientific=dict(audit_passed=False))


def aggregate_cases(index_path, *, workers=1, case_keys=None):
    study.require(type(workers) is int and workers > 0, "Workers must be a positive integer")
    index = load_index(index_path)
    keys = [row["case_key"] for row in index["cases"]] if case_keys is None else list(case_keys)
    known = {row["case_key"] for row in index["cases"]}
    study.require(keys and len(set(keys)) == len(keys) and set(keys) <= known, "Invalid or duplicate case selection")
    args = [(str(Path(index_path).absolute()), key) for key in keys]
    if workers == 1:
        return [_run_case(arg) for arg in args]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_case, args))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Freeze raw run roots and their exact case roster; no fits or audits")
    prepare.add_argument("--run-roots", nargs="+", required=True, type=Path)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--models", nargs="+", type=int, choices=(0, 1, 2),
                         help="Restrict the declared model scope; zero-based ID 1 is Model II")
    prepare.add_argument("--publication-mode", choices=("compact", "full"), default="compact",
                         help="Compact keeps audited winners; full also duplicates all factors (default: compact)")
    prepare.add_argument("--allowed-missing-tasks", type=Path,
                         help="JSON list of exact permitted absent tuning tasks and cancellation evidence")
    case = sub.add_parser("case", help="Audit one complete case; suitable for an independent job")
    case.add_argument("--index", type=Path, required=True)
    case.add_argument("--case-key", required=True)
    run = sub.add_parser("run", help="Audit independent cases with a bounded process pool")
    run.add_argument("--index", type=Path, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--case-keys", nargs="+")
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            index = prepare_campaign(args.run_roots, args.output_root, models=args.models,
                                     publication_mode=args.publication_mode,
                                     allowed_missing_tasks=(_parse_result(args.allowed_missing_tasks.read_bytes())
                                         if args.allowed_missing_tasks is not None else None))
            print(json.dumps(dict(index=str(args.output_root / "campaign-index.json"), cases=len(index["cases"]),
                                  scope=index["scope"], publication_mode=index["publication_mode"])))
            return 0
        if args.command == "case":
            results = [aggregate_case(args.index, args.case_key)]
        elif args.dry_run:
            index = load_index(args.index)
            keys = [row["case_key"] for row in index["cases"]] if args.case_keys is None else args.case_keys
            study.require(args.workers > 0 and len(set(keys)) == len(keys)
                and set(keys) <= {row["case_key"] for row in index["cases"]}, "Invalid workers or case selection")
            print(json.dumps(dict(no_fits_performed=True, no_audits_performed=True, workers=args.workers, case_keys=keys)))
            return 0
        else:
            results = aggregate_cases(args.index, workers=args.workers, case_keys=args.case_keys)
        failures = [r["case_key"] for r in results if not r["scientific"]["audit_passed"]
                    or not _permitted_execution(r.get("execution", {}))]
        print(json.dumps(dict(cases=len(results), resumed=sum(r.get("resumed", False) for r in results),
                              failed_cases=failures, errors=[r for r in results if "error" in r])))
        return int(bool(failures))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
