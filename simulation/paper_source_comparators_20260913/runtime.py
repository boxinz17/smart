"""Bind immutable paper inputs and fit one independently tuned comparator."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import os
import subprocess
from pathlib import Path
import time
import traceback
import numpy as np
from .common import read, write, immutable, sha, digest, require, load_plan, verify_imports, require_slurm

def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

def fingerprint(values):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        h.update(name.encode()); h.update(str((value.shape, value.dtype.str)).encode()); h.update(value.tobytes())
    return h.hexdigest()

def prepare(root):
    require_slurm(root)
    plan = load_plan(root, full_source=True)
    references, meta_by_case, groups = {}, {}, {}
    from run_sparse_smart_v2_pilot import _fingerprints
    for ref in plan["references"]:
        old = Path(ref["root"])
        for key, name in (("plan_sha256", "plan.json"), ("preparation_sha256", "preparation.json"),
                          ("summary_sha256", "summary/summary.json"),
                          ("source_manifest_sha256", "source-manifest.json"),
                          ("numerical_audit_sha256", "summary/numerical-audit.json"),
                          ("numerical_audit_cases_sha256", "summary/numerical-audit-cases.jsonl")):
            require(sha(old / name) == ref[key], f"reference changed: {old}/{name}")
        old_plan, prep, summary = (read(old / name) for name in ("plan.json", "preparation.json", "summary/summary.json"))
        require(old_plan["method"] == "SparseSMARTv2Pilot" and prep["success"] and summary["success"],
                "reference incomplete or different method")
        require(read(old / "summary/numerical-audit.json")["success"], "reference numerical audit failed")
        require(prep["plan_sha256"] == ref["plan_sha256"] and summary["plan_sha256"] == ref["plan_sha256"],
                "reference preparation/summary plan binding differs")
        references[str(old)] = ref
        for item in prep["cases"]:
            path = old / "cases" / item["case_id"] / "case.json"
            require(sha(path) == item["case_json_sha256"], "reference case metadata changed")
            meta_by_case[(str(old), item["case_id"])] = (read(path), path)
    # Verify actual arrays for every display alias before any reuse is permitted.
    for i, display in enumerate(plan["display_cases"]):
        case, gid = display["case"], display["group_id"]
        meta, path = meta_by_case[(display["reference_root"], case["case_id"])]
        require(meta["case"] == case, "reference case fields differ")
        for name in ("data.npz", "source.npz"):
            require(sha(path.parent / name) == meta["files"][name], "reference data/source hash differs")
        data, source = arrays(path.parent / "data.npz"), arrays(path.parent / "source.npz")
        require(_fingerprints(data) == meta["fingerprints"], "reference numerical input fingerprints differ")
        require(data["X"].shape == (case["n_train"], case["p"]) and data["Y"].shape == (case["n_train"], case["q"])
                and data["X_validation"].shape == (200, case["p"]) and data["Y_validation"].shape == (200, case["q"]),
                "data budget/shape mismatch")
        require(source["left"].shape == (case["p"], case["p"]) and source["right"].shape == (case["q"], case["q"]),
                "source must contain full observed frames")
        full_fingerprint = fingerprint({k:source[k] for k in ("left", "right")})
        if gid in groups:
            require(groups[gid]["fingerprints"] == meta["fingerprints"], "alias observations differ")
            require(groups[gid]["full_source_fingerprint"] == full_fingerprint, "alias full source frames differ")
        else:
            groups[gid] = dict(group_id=gid, fingerprints=meta["fingerprints"], full_source_fingerprint=full_fingerprint,
                files={name:dict(path=str(path.parent / name), sha256=meta["files"][name]) for name in ("data.npz", "source.npz")},
                reference_case_json_path=str(path), reference_case_json_sha256=sha(path))
        if (i+1) % 200 == 0:
            print(f"Verified {i+1}/{plan['n_display_cases']} display inputs", flush=True)
    records = []
    for group in plan["groups"]:
        record = dict(groups[group["group_id"]], group=group, plan_fingerprint=plan["plan_fingerprint"])
        path = root / "groups" / group["group_id"] / "group.json"
        immutable(path, record)
        records.append(dict(group_id=group["group_id"], group_json_sha256=sha(path)))
    marker = dict(schema_version=1, status="complete", success=True, n_groups=plan["n_groups"], n_tasks=plan["n_tasks"],
                  plan_fingerprint=plan["plan_fingerprint"], plan_sha256=sha(root / "plan.json"),
                  source_manifest_sha256=plan["source_manifest_sha256"], references=references,
                  groups=records, verified_display_aliases=plan["n_display_cases"], slurm_job_id=os.environ["SLURM_JOB_ID"])
    own_manifest = read(root / "source-manifest.json")["files"]
    initializer_files = ("sparse-smart-v2/src/sparse_smart_v2/initialization.py",
                         "sparse-smart/src/sparse_smart/initialization.py", "sparse-smart/src/sparse_smart/source.py")
    for ref in references.values():
        old_files = read(Path(ref["root"]) / "source-manifest.json")["files"]
        require(all(own_manifest[name] == old_files[name] for name in initializer_files),
                "raw initializer primitive differs from frozen reference")
    marker["initializer_implementation_identical_to_reference"] = list(initializer_files)
    write(root / "preparation.json", marker)
    return marker

def load_group(root, plan, gid):
    prep = read(root / "preparation.json")
    require(prep["success"] and prep["plan_sha256"] == sha(root / "plan.json"), "preparation differs")
    expected = next(g for g in prep["groups"] if g["group_id"] == gid)
    path = root / "groups" / gid / "group.json"
    require(sha(path) == expected["group_json_sha256"], "group metadata changed")
    meta = read(path)
    require(meta["plan_fingerprint"] == plan["plan_fingerprint"] and meta["group_id"] == gid, "group identity differs")
    for item in meta["files"].values():
        require(sha(item["path"]) == item["sha256"], "reference file changed")
    data = arrays(meta["files"]["data.npz"]["path"])
    source = arrays(meta["files"]["source.npz"]["path"])
    return data, source, meta

def metrics(coefficient, data):
    error = coefficient - data["C_star"]
    return dict(coefficient_rmse=float(np.sqrt(np.mean(error**2))),
                coefficient_frobenius_squared=float(np.sum(error**2)),
                validation_mse=float(np.mean((data["Y_validation"] - data["X_validation"] @ coefficient)**2)),
                training_prediction_mse=float(np.mean((data["X"] @ error)**2)))

def audit_selected(coefficient, selection, data, source, task):
    """Independent selected-state checks; all other candidate scores are metadata."""
    x, y, c = data["X"], data["Y"], coefficient
    require(c.shape == (x.shape[1], y.shape[1]) and np.isfinite(c).all(), "invalid selected coefficient")
    candidates = selection["candidate_results"]
    eligible = [v for v in candidates if v["status"] == "eligible"]
    winner = min(eligible, key=lambda v:(v["validation_mse"], v["index"]))
    require(winner["index"] == selection["selected_index"], "selected candidate is not validation winner")
    val = float(np.mean((data["Y_validation"]-data["X_validation"] @ c)**2))
    require(np.isclose(val, selection["validation_mse"], rtol=1e-10, atol=1e-12), "selected validation score differs")
    method, params = task["method"], winner["parameters"]
    checks = dict(success=True, selected_validation_recomputed=True, metadata_argmin_verified=True,
                  all_candidate_states_retained=False, all_candidate_scores_independently_recomputed=False)
    if method in ("ridge_to_source", "nuclear_contrast"):
        correction = c - data["C0"]
        residual = y - x @ c
        if method == "ridge_to_source":
            gradient = -x.T @ residual / len(x) + params["ridge"] * correction
            scale = max(1., np.linalg.norm(x.T @ y / len(x)), np.linalg.norm(params["ridge"] * data["C0"]))
            require(np.linalg.norm(gradient) <= 1e-8 * scale, "centered ridge normal equations failed")
            checks["normal_equation_residual"] = float(np.linalg.norm(gradient))
        else:
            penalty = params["nuclear_penalty"]
            if penalty > 0:
                dual = residual / len(x)
                norm = np.linalg.norm(x.T @ dual, 2)
                dual *= min(1., penalty / norm) if norm else 1.
                primal = np.sum(residual**2)/(2*len(x)) + penalty*np.linalg.svd(correction, compute_uv=False).sum()
                dual_value = np.sum((y-x@data["C0"])*dual)-len(x)*np.sum(dual**2)/2
                gap = max(0., float(primal-dual_value))
                require(gap <= params["tolerance"]*max(1., abs(primal))*(1+1e-6)+1e-9, "nuclear dual gap failed")
                checks["duality_gap"] = gap
            else:
                require(np.linalg.norm(x.T @ residual) <= 1e-8*max(1., np.linalg.norm(x.T@y)),
                        "zero-penalty correction normal equations failed")
    if method not in ("ridge_to_source", "nuclear_contrast", "source_target_mixture"):
        sv = np.linalg.svd(c, compute_uv=False)
        numerical_rank = int(np.sum(sv > max(c.shape)*np.finfo(float).eps*max(1., sv[0])))
        require(numerical_rank <= task["rank"], "rank upper bound failed")
        checks["rank_bound_verified"] = True
    if method.startswith("source_subspace"):
        du, dv = params["source_rank"]
        u, v = source["left"][:, :du], source["right"][:, :dv]
        require(np.allclose(c, u@(u.T@c@v)@v.T, rtol=1e-8, atol=1e-10), "restricted source span failed")
        checks["source_span_verified"] = True
    if method == "source_target_mixture" and params["alpha"] == 1:
        require(np.array_equal(c, data["C0"]), "source-only mixture endpoint differs")
    return checks

def audit_initializer_reference(plan, task, selection, coefficient, source):
    """Check the raw selected initializer against the original cached SVD factors."""
    display = next(d for d in plan["display_cases"] if d["task_ids"]["initializer_only"] == task["task_id"])
    folder = Path(display["reference_root"]) / "cases" / display["case"]["case_id"]
    meta = read(folder / "case.json")
    require(sha(folder / "initializers.npz") == meta["files"]["initializers.npz"], "reference initializer archive changed")
    record = next(r for r in meta["initializers"] if r["init_penalty"] == selection["selected_parameters"]["penalty"])
    initial = arrays(folder / "initializers.npz")
    prefix = f"i{record['index']}_"
    if prefix + "P" not in initial:
        return dict(reference_initializer_match=None, reference_initializer_unavailable=True)
    du, dv = selection["selected_parameters"]["source_rank"]
    expected = source["left"][:, :du] @ ((initial[prefix+"P"]*initial[prefix+"d"]) @ initial[prefix+"Q"].T) @ source["right"][:, :dv].T
    require(np.allclose(coefficient, expected, rtol=1e-9, atol=1e-11), "fresh raw initializer differs from cached reference factors")
    return dict(reference_initializer_match=True, reference_initializer_max_abs_error=float(np.max(np.abs(coefficient-expected))))

def fit(root, task_id):
    require_slurm(root)
    plan = load_plan(root)
    require(type(task_id) is int and 0 <= task_id < plan["n_tasks"], "invalid task")
    task = plan["tasks"][task_id]
    destination = root / "tasks" / f"{task_id:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    identity = dict(task=task, group_id=task["group_id"], plan_fingerprint=plan["plan_fingerprint"],
                    source_manifest_sha256=plan["source_manifest_sha256"])
    with (destination / ".fit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tick = time.monotonic()
        write(destination / "status.json", dict(identity, status="running", slurm_job_id=os.environ["SLURM_JOB_ID"]))
        result = dict(identity, schema_version=1, status="failed", execution_success=False,
                      selection=None, metrics=None, audit=None, files={})
        try:
            data, source, meta = load_group(root, plan, task["group_id"])
            from .fitting import fit_method
            verify_imports(root)
            observed = {k:data[k] for k in ("X", "Y", "C0", "X_validation", "Y_validation")}
            selection, coefficient = fit_method(observed, source, task["method"], task["rank"], plan["configuration"])
            result.update(selection=selection, fingerprints=meta["fingerprints"], full_source_fingerprint=meta["full_source_fingerprint"])
            if coefficient is not None:
                result["audit"] = audit_selected(coefficient, selection, observed, source, task)
                if task["method"] == "initializer_only":
                    result["audit"].update(audit_initializer_reference(plan, task, selection, coefficient, source))
                result["metrics"] = metrics(coefficient, data)
                np.savez_compressed(destination / "selected.npz", coefficient=coefficient)
                result["files"]["selected.npz"] = sha(destination / "selected.npz")
            else:
                result["audit"] = dict(success=True, no_eligible_candidate=True)
            verify_imports(root)
            result.update(status="complete", execution_success=True)
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        result["elapsed_seconds"] = time.monotonic()-tick
        result["slurm"] = dict(job_id=os.environ["SLURM_JOB_ID"], step_id=os.environ.get("SLURM_STEP_ID"))
        write(destination / "result.json", result)
        write(destination / "status.json", dict(identity, status="finished", result_sha256=sha(destination / "result.json"),
                                                execution_success=result["execution_success"], slurm=result["slurm"]))
        return result

def canary_ids(plan):
    """Fixed seed-zero coverage of dimensions, rank eleven and source-noise edges."""
    selected = set()
    for display in plan["display_cases"]:
        c = display["case"]
        if c["seed_id"] != 0:
            continue
        if ((c["experiment_id"] == 0 and c["setting_index"] == 0)
                or (c["model_id"] == 2 and c["experiment_id"] == 3 and c["setting_index"] in (0,5))
                or (c["model_id"] == 2 and c["experiment_id"] == 1 and c["rank"] == 11)):
            selected.update(display["task_ids"].values())
    return sorted(selected)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit", "summary", "canary"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        value = prepare(root)
    elif args.action == "canary":
        require_slurm(root)
        plan = load_plan(root)
        results = []
        for tid in canary_ids(plan):
            process = subprocess.run(["bash", str(root / "source/hpc/discovery/paper_source_comparators_worker.sh"),
                                      "dispatch", str(root), str(tid)], check=False)
            path = root / "tasks" / f"{tid:06d}" / "result.json"
            result = read(path) if path.exists() else {}
            results.append(dict(task_id=tid, execution_success=process.returncode == 0 and result.get("execution_success") is True,
                                elapsed_seconds=result.get("elapsed_seconds"), error=result.get("error"),
                                selection_status=(result.get("selection") or {}).get("status")))
            print(results[-1], flush=True)
        value = dict(success=all(r["execution_success"] for r in results), tasks=results,
                     plan_fingerprint=plan["plan_fingerprint"])
        write(root / "canary.json", value)
    elif args.action == "summary":
        require_slurm(root)
        from .report import summarize
        value = summarize(root)
    else:
        value = fit(root, args.task)
    print({key:value.get(key) for key in ("status", "success", "execution_success", "n_tasks", "elapsed_seconds")}, flush=True)
    if value.get("execution_success") is False or value.get("success") is False:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
