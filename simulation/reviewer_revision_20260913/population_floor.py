"""Evaluation-only population approximation floors for audited source spans.

Requires a complete passing main audit in ROOT. Uses only C0 from data.npz,
C_star/Sigma_x from truth.npz, and saved selected coefficients; no response
array is loaded and no estimator is fitted. Production execution is Slurm-only.
This is a truth-based diagnostic for a restricted coefficient class, not a
competitor or an attainable risk bound for general transfer estimators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
_code = Path(__file__).resolve().parents[2]
for _path in (_code / "simulation", _code / "sparse-smart/src", _code / "sparse-smart-v2/src"):
    sys.path.insert(0, str(_path))
import numpy as np
from reviewer_revision_20260913 import audit as common


def population_floor(C_star, Sigma_x, U, V):
    """Return the Sigma-weighted projection onto {U B V.T : arbitrary B}."""
    if any(np.iscomplexobj(value) for value in (C_star, Sigma_x, U, V)):
        raise ValueError("Expected finite real matrices")
    C, sigma, u, v = [np.asarray(value, dtype=float) for value in (C_star, Sigma_x, U, V)]
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in (C, sigma, u, v)):
        raise ValueError("Expected finite real matrices")
    p, q = C.shape
    if (p == 0 or q == 0 or sigma.shape != (p, p) or u.shape[0] != p or v.shape[0] != q
            or not 0 < u.shape[1] <= p or not 0 < v.shape[1] <= q):
        raise ValueError("Incompatible coefficient, covariance, or frame dimensions")
    if not np.allclose(sigma, sigma.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Sigma_x must be symmetric positive definite")
    try:
        root = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as error:
        raise ValueError("Sigma_x must be symmetric positive definite") from error
    if not (np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-9, rtol=1e-9)
            and np.allclose(v.T @ v, np.eye(v.shape[1]), atol=1e-9, rtol=1e-9)):
        raise ValueError("Source frames must have orthonormal columns")
    gram = u.T @ sigma @ u
    reduced = np.linalg.solve(gram, u.T @ sigma @ C @ v)
    coefficient = u @ reduced @ v.T
    error = coefficient - C
    risk = float(np.sum((root.T @ error)**2) / q)
    return dict(coefficient=coefficient, reduced_coefficient=reduced, risk=risk,
                floor_rank=int(np.linalg.matrix_rank(reduced)), truth_rank=int(np.linalg.matrix_rank(C)),
                gram_condition=float(np.linalg.cond(gram)),
                normal_equation_residual=float(np.linalg.norm(u.T @ sigma @ error @ v)))


def _wanted(method):
    return method.startswith(("source_subspace_", "initializer_only", "v2"))


def _restricted(method):
    return method.startswith(("source_subspace_", "initializer_only"))


def _json(path):
    return json.loads(path.read_text())


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_inputs(root, plan, digest):
    receipt = _json(root / "audit.json")
    count = len(plan["tasks"])
    if (receipt.get("audit_passed") is not True or receipt.get("all_planned_tasks_complete") is not True
            or receipt.get("plan_sha256") != digest or receipt.get("n_planned_tasks") != count
            or receipt.get("n_validated_complete_tasks") != count or receipt.get("n_errors") != 0
            or receipt.get("coverage_scope") == "task_subset"):
        raise RuntimeError("Population floors require a complete passing audit for the unchanged original plan")
    reports = receipt.get("tasks", [])
    if (Counter(row.get("task_index") for row in reports) != Counter(range(count))
            or any(row.get("state") != "complete" for row in reports)):
        raise RuntimeError("Main audit task coverage is incomplete or duplicated")
    with (root / "per_replication.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = Counter((index, method) for index, task in enumerate(plan["tasks"])
                       for method in common._expected_methods(plan, task))
    actual = Counter((int(row["task_index"]), row["method"]) for row in rows)
    if expected != actual or any(count != 1 for count in actual.values()):
        raise RuntimeError("Audited CSV does not cover all original task/method outcomes exactly once")
    for row in rows:
        task = plan["tasks"][int(row["task_index"])]
        case = plan["cases"][task["case_index"]]
        if (row["case_id"] != case["case_id"] or int(row["seed"]) != task["seed"]
                or row["status"] not in ("success", "failed")):
            raise RuntimeError("Audited CSV identity or outcome status differs from the complete study")
    return {(int(row["task_index"]), row["method"]): row for row in rows}


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"], row["method"]].append(row)
    result = []
    for (case_id, method), members in sorted(groups.items()):
        successful = [row for row in members if row["status"] == "success"]
        record = dict(case_id=case_id, method=method, n_planned=len(members), n_success=len(successful),
                      n_failure=len(members)-len(successful), restricted_to_observed_spans=members[0]["restricted_to_observed_spans"],
                      n_below_floor=sum(row["below_floor"] for row in successful))
        for name, values in (("floor", [row["floor_risk"] for row in members]),
                             ("selected_risk", [row["selected_risk"] for row in successful]),
                             ("risk_minus_floor", [row["risk_minus_floor"] for row in successful]),
                             ("outside_span_norm", [row["outside_span_norm"] for row in successful])):
            record.update({f"{name}_{key}": value for key, value in common._moments(values).items()})
        result.append(record)
    return result


def evaluate(root, *, write=True, allow_tiny_local=False):
    root, plan, digest = common._load_plan(root, allow_tiny_local)
    audited = _audit_inputs(root, plan, digest)
    input_hashes = {name: _digest(root / name) for name in ("plan.json", "audit.json", "per_replication.csv")}
    rows, tasks, issues = [], [], []
    from sparse_smart_v2 import ObservedSource, prepare_source
    for index, task in enumerate(plan["tasks"]):
        case = plan["cases"][task["case_index"]]
        directory = root / "tasks" / f"task-{index:06d}"
        # Loading individual members is deliberate: no X, Y, validation or test
        # responses/covariates enter this evaluation-only calculation.
        with np.load(directory / "data.npz", allow_pickle=False) as archive:
            source = archive["C0"]
        with np.load(directory / "truth.npz", allow_pickle=False) as archive:
            truth, sigma = archive["C_star"], archive["Sigma_x"]
        prepared = prepare_source(ObservedSource(source), p=case["p"], q=case["q"],
                                  source_rank=case["source_rank"])
        u, v = prepared.leading_left, prepared.leading_right
        floor = population_floor(truth, sigma, u, v)
        receipt = _json(directory / "result.json")
        if (receipt.get("plan_sha256") != digest or receipt.get("complete") is not True
                or receipt.get("task") != task or receipt.get("case") != case):
            raise RuntimeError(f"Task {index} no longer matches the passing audit")
        tasks.append(dict(task_index=index, case_id=case["case_id"], seed=task["seed"],
            floor_risk=floor["risk"], floor_rank=floor["floor_rank"], truth_rank=floor["truth_rank"],
            gram_condition=floor["gram_condition"], normal_equation_residual=floor["normal_equation_residual"],
            evaluation_input_fingerprint=common._fingerprint(dict(C0=source, C_star=truth, Sigma_x=sigma,
                                                                  observed_left=u, observed_right=v))))
        with np.load(directory / "coefficients.npz", allow_pickle=False) as archive:
            for method in common._expected_methods(plan, task):
                if not _wanted(method):
                    continue
                previous = audited[index, method]
                recorded = receipt["methods"][method]
                for name in ("fit_data_fingerprint", "truth_fingerprint"):
                    if previous[name] != receipt[name]:
                        raise RuntimeError(f"Task {index} audit/receipt fingerprints differ")
                row = dict(task_index=index, case_id=case["case_id"], family=case.get("family"),
                    level=case.get("level"), seed=task["seed"], n_train=case["n_train"],
                    source_dimension=case["source_rank"], method=method, status=previous["status"],
                    floor_risk=floor["risk"], floor_rank=floor["floor_rank"], truth_rank=floor["truth_rank"],
                    restricted_to_observed_spans=_restricted(method))
                if previous["status"] == "failed":
                    if recorded.get("success") is not False:
                        raise RuntimeError(f"Task {index}/{method} changed its failure status")
                    rows.append(row)
                    continue
                if recorded.get("success") is not True:
                    raise RuntimeError(f"Task {index}/{method} changed its successful status")
                coefficient = archive[method]
                if coefficient.shape != truth.shape or not np.isfinite(coefficient).all():
                    raise RuntimeError(f"Task {index}/{method} invalid selected coefficient")
                error = coefficient - truth
                risk = float(np.sum(error * (sigma @ error)) / truth.shape[1])
                if not common._close(risk, float(previous["population_prediction_excess"])):
                    raise RuntimeError(f"Task {index}/{method} risk differs from the independently audited CSV")
                outside = float(np.linalg.norm(coefficient - u @ (u.T @ coefficient @ v) @ v.T))
                tolerance = 2e-8 * max(1., abs(risk), abs(floor["risk"]))
                below = risk < floor["risk"] - tolerance
                fitted_rank = recorded.get("parameters", {}).get("rank", case.get("fitted_rank", case["target_rank"]))
                row.update(selected_risk=risk, risk_minus_floor=risk-floor["risk"], below_floor=below,
                    outside_span_norm=outside, comparison_tolerance=tolerance, fitted_rank=fitted_rank,
                    rank_sufficient_for_exact_floor=fitted_rank >= floor["truth_rank"],
                    selected_method=recorded.get("selected_method"))
                if row["restricted_to_observed_spans"]:
                    if outside > 2e-8 * max(1., float(np.linalg.norm(coefficient))):
                        issues.append(dict(task_index=index, method=method, code="restricted_estimator_outside_observed_spans"))
                    if below:
                        issues.append(dict(task_index=index, method=method, code="restricted_estimator_below_population_floor"))
                elif below and outside <= 2e-8 * max(1., float(np.linalg.norm(coefficient))):
                    issues.append(dict(task_index=index, method=method, code="below_floor_without_resolved_outside_component"))
                rows.append(row)
    if any(_digest(root / name) != value for name, value in input_hashes.items()):
        raise RuntimeError("The frozen plan or its independent audit changed during evaluation")
    report = dict(schema=1, diagnostic_passed=not issues, issues=issues, plan_sha256=digest,
        source_frame_convention="sparse_smart_v2.prepare_source full deterministic observed frames; same operational prefix as the main study",
        n_planned_tasks=len(plan["tasks"]), n_evaluated_tasks=len(tasks), input_sha256=input_hashes, tasks=tasks,
        scope="Evaluation-only covariance-weighted approximation floor; no fitting or training-response access; not a competitor or a floor for unrestricted v2",
        limitations=["Passing main audit is required; only selected risks and span membership are rechecked here, not the complete candidate audit.",
                     "Rank equality refers to a rank-at-most constraint. The unrestricted span floor remains a lower bound when a smaller rank excludes its minimizer.",
                     "A v2 coefficient may beat this floor through components outside the retained observed spans; this is not a violation or a deployable oracle estimator.",
                     "Monte Carlo summaries retain every planned method outcome; selected-risk contrasts condition on successful fits."])
    summary = _summary(rows)
    if write:
        common._atomic_json(root / "population-floor-audit.json", report)
        for name, output in (("population-floor-per-replication.csv", rows), ("population-floor-summary.csv", summary)):
            common._csv(root / name, output, list(dict.fromkeys(key for row in output for key in row)))
    return dict(audit=report, per_replication=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.root, write=not args.no_write)
    print(json.dumps({key: result["audit"][key] for key in ("diagnostic_passed", "n_planned_tasks", "n_evaluated_tasks")}, sort_keys=True))
    raise SystemExit(0 if result["audit"]["diagnostic_passed"] else 1)


if __name__ == "__main__":
    main()
