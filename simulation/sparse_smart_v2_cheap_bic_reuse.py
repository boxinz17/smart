"""Audited reuse of an existing RRR endpoint, and the assessment release gate.

No iterative state is inferred from a longer terminal run. Existing coefficient
arrays are read in place; only the selected RRR coefficient enters the new
task's normal compact output. Every descriptor is bound into the new plan.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from sparse_smart_v2_bic_plan import digest, sha


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def verify_assessment_gate(root, plan):
    root = Path(root)
    gate = read(root / "assessment-gate.json")
    pair_path = Path(gate["pair_path"])
    require(pair_path.is_absolute() and pair_path.is_relative_to(root.parent), "pair path escapes pilot")
    require(gate["assessment_plan_fingerprint"] == plan["plan_fingerprint"], "assessment gate plan mismatch")
    require(sha(pair_path) == gate["pair_sha256"], "frozen initializer pair changed")
    require(gate.get("status") == "released_after_pair_freeze", "assessment has not been released")
    pair = read(pair_path)
    require(pair.get("phase") == "develop" and pair.get("evaluation_metrics_used") is False,
            "pair is not development-only")
    require(pair.get("artifact_fingerprint") == digest({k: v for k, v in pair.items() if k != "artifact_fingerprint"}),
            "pair artifact fingerprint mismatch")
    require(pair.get("iterations") == 200, "pair has another iteration budget")
    require(pair.get("development_plan_fingerprint") == gate.get("development_plan_fingerprint"),
            "gate and development pair disagree")
    development = read(pair_path.parent.parent / "plan.json")
    require(development.get("plan_fingerprint") == pair.get("development_plan_fingerprint"), "development plan changed")
    require(development["cheap_bic"]["split_fingerprint"] == plan["cheap_bic"]["split_fingerprint"], "assessment split differs")
    require(pair.get("source_files_fingerprint") == digest(read(root / "source-manifest.json")["files"]),
            "assessment source differs from development")
    require(plan["configuration"].get("cheap_bic_phase") == "assessment", "gate is for assessment only")
    return gate


def rrr_descriptor(old_root, old_plan, group, rank, new_manifest):
    """Return a fully bound endpoint descriptor, or explicit unavailability."""
    old_root = Path(old_root)
    old_manifest = read(old_root / "source-manifest.json")
    for name in ("sparse-smart-v2/src/sparse_smart_v2/rrr.py", "sparse-smart-v2/src/sparse_smart_v2/selection.py"):
        # RRR has also lived under target_rrr.py; verify whichever is present.
        if name.endswith("/rrr.py") and name not in new_manifest["files"]:
            name = "sparse-smart-v2/src/sparse_smart_v2/target_rrr.py"
        require(name in new_manifest["files"] and old_manifest["files"].get(name) == new_manifest["files"][name],
                "RRR/scoring implementation differs from existing endpoint")
    tasks = [t for t in old_plan["tasks"] if t["group_id"] == group["group_id"] and t["rank"] == rank and t["include_rrr"]]
    if not tasks:
        return None, "estimated_rank_absent_from_existing_grid"
    require(len(tasks) == 1, "ambiguous old RRR task")
    folder = old_root / "tasks" / f"{tasks[0]['task_id']:06d}"
    if not (folder / "result.json").exists():
        return None, "existing_task_not_complete"
    result, status = read(folder / "result.json"), read(folder / "status.json")
    require(status.get("result_sha256") == sha(folder / "result.json") and status.get("status") == "finished",
            "old endpoint result/status mismatch")
    require(result.get("task") == tasks[0] and result.get("plan_fingerprint") == old_plan["plan_fingerprint"],
            "old endpoint plan/task mismatch")
    require(result.get("status") == "complete" and result.get("execution_success") is True,
            "old RRR task has execution failure")
    matches = [r for r in result["outcomes"] if r.get("fit_method") == "target_rrr"]
    require(len(matches) == 1, "old RRR endpoint missing/duplicated")
    row = matches[0]
    require(row.get("success") is True and row.get("n_iter") == 0 and row.get("rrr_certificate", {}).get("certified") is True,
            "old RRR endpoint is not certified")
    require(isinstance(row.get("state_key"), str), "old RRR state not retained")
    require(sha(folder / "states.npz") == result["files"]["states.npz"], "old RRR states hash changed")
    return dict(result_path=str(folder / "result.json"), result_sha256=sha(folder / "result.json"),
                states_path=str(folder / "states.npz"), states_sha256=result["files"]["states.npz"],
                candidate_id=row["candidate_id"], state_key=row["state_key"], rank=rank,
                originating_plan_fingerprint=old_plan["plan_fingerprint"],
                training_fingerprint=result["fingerprints"]["training_observed_input_fingerprint"]), "reusable_rrr"


def reuse_rrr(descriptor, group, task, config, data, candidate_id):
    from run_sparse_smart_v2_pilot import _metrics
    from sparse_smart_v2_cheap_bic_plan import _training_fingerprint
    from sparse_smart_v2.selection import bic_score
    started = time.monotonic()
    cpu_started = time.process_time()
    require(task["rank"] == descriptor["rank"], "reused RRR rank mismatch")
    require(_training_fingerprint(data["X"], data["Y"], data["C0"]) == descriptor["training_fingerprint"],
            "reused RRR observed data mismatch")
    require(sha(descriptor["result_path"]) == descriptor["result_sha256"], "reused RRR result changed")
    require(sha(descriptor["states_path"]) == descriptor["states_sha256"], "reused RRR arrays changed")
    result = read(descriptor["result_path"])
    require(result["plan_fingerprint"] == descriptor["originating_plan_fingerprint"], "reused RRR provenance changed")
    rows = [r for r in result["outcomes"] if r["candidate_id"] == descriptor["candidate_id"]]
    require(len(rows) == 1, "reused RRR identity missing")
    old = rows[0]
    require(old.get("fit_method") == "target_rrr" and old.get("success") and old.get("execution_success")
            and old.get("n_iter") == 0 and old.get("rrr_certificate", {}).get("certified"), "invalid reused RRR")
    with np.load(descriptor["states_path"], allow_pickle=False) as archive:
        coefficient = archive[descriptor["state_key"] + "coefficient"].copy()
    require(coefficient.shape == (group["p"], group["q"]) and np.isfinite(coefficient).all(), "invalid RRR coefficient")
    score = bic_score(data["X"], data["Y"], coefficient, rank=task["rank"], direct_rrr=True,
                      support_tolerance=config["support_tolerance"]).as_dict()
    require(np.isclose(score["score"], old["selection"]["score"], rtol=1e-10, atol=1e-9), "reused RRR BIC mismatch")
    row = deepcopy(old)
    row.update(candidate_id=candidate_id, init_penalty=task["init_penalty"], state_key=None,
               selection=score, metrics=_metrics(coefficient, data), reuse_provenance=descriptor,
               original_elapsed_seconds=old["elapsed_seconds"], elapsed_seconds=time.monotonic()-started,
               process_cpu_seconds=time.process_time()-cpu_started,
               numerical_work={"rrr_reused_from_existing_campaign": True, "accepted_updates": 0})
    return row, {"coefficient": coefficient}, None
