"""Audit the targeted 66-cell SparseSMART solver regression without fitting.

The comparison preserves the existing observations, tuning grids and iteration
budgets. It is deliberately selected on earlier failures and is not a new
unbiased comparison against the paper or an updated full-grid benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from run_sparse_smart import _digest_json


HERE = Path(__file__).resolve().parent
PATTERN = "model*/exp*/SparseSMARTExternal_result_*.json"
POLICY_KEYS = ("initialization_spectrum", "refinement_solver")
HASH_KEYS = ("training_observed_input_fingerprint", "validation_observed_input_fingerprint",
             "evaluation_truth_fingerprint")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value), f"Nonfinite or invalid {name}")
    return float(value)


def _close(left, right, message):
    _require(math.isclose(_finite(left, message), _finite(right, message),
                         rel_tol=1e-10, abs_tol=1e-12), message)


def _validation_candidate(candidate, budget):
    count = candidate["n_iter"]
    _require(type(count) is int and 0 <= count <= budget, "Invalid candidate iteration count")
    if not candidate["success"]:
        _require(candidate["validation_mse"] is None, "Failed candidate has an eligible validation loss")
        return
    _require(candidate["status"] in ("completed", "converged"), "Invalid successful candidate status")
    history = candidate["validation_history"]
    _require([v["iteration"] for v in history] == list(range(count + 1)),
             "Incomplete candidate validation history")
    for value in history:
        _finite(value["loss"], "validation loss")
    best = min(history, key=lambda v: v["loss"])
    _require(candidate["selected_iteration"] == best["iteration"], "Candidate did not select minimum validation iterate")
    _close(candidate["validation_mse"], best["loss"], "Candidate validation score mismatch")
    if candidate["status"] == "completed":
        _require(count == budget and candidate["termination_reason"] == "max_iterations",
                 "Completed candidate did not exhaust its configured iteration budget")
    else:
        _require(candidate["termination_reason"] == "stationarity", "Converged candidate lacks stationarity termination")
    diagnostics = candidate.get("diagnostics", {})
    _require(diagnostics.get("stationarity_scope") == "full_chart_constraints"
             and diagnostics.get("diagnostic_coordinates") == "omega,d,H",
             "Candidate does not report the full-constraint H-coordinate diagnostic")
    _require(diagnostics.get("optimization_converged") == (candidate["termination_reason"] == "stationarity"),
             "Candidate optimization convergence flag mismatch")
    _require(diagnostics.get("selected_converged") ==
             (candidate["termination_reason"] == "stationarity" and best["iteration"] == count),
             "Candidate selected convergence flag mismatch")


def validate_pair(old, new):
    """Validate one old/new pair and return compact outcome diagnostics."""
    _require(old["applicable"] is True and new["applicable"] is True, "Repair cell must remain applicable")
    for key in ("schema_version", "method", "model", "experiment", "rd_seed_id", "random_seed", "setting", "generator_arguments",
                "n_train", "n_validation", "all_training_rows_used", "training_matches_legacy",
                "refit_on_all_data", "validation_seed_metadata", "split", "input_fingerprint", *HASH_KEYS):
        _require(old[key] == new[key], f"Before/after data protocol mismatch: {key}")
    for key in (*HASH_KEYS, "implementation_fingerprint"):
        _require(isinstance(new[key], str) and re.fullmatch(r"[0-9a-f]{64}", new[key]), f"Invalid {key}")
    _require(new["n_train"] == new["setting"]["n"] and new["n_validation"] == 100,
             "Training/validation sample counts changed")
    _require(new["all_training_rows_used"] is True and new["training_matches_legacy"] is True
             and new["refit_on_all_data"] is False, "Incorrect data-use flags")
    before_config, after_config = old["configuration"], new["configuration"]
    before_options, after_options = dict(before_config["runner"]), dict(after_config["runner"])
    for options in (before_options, after_options):
        _require(options.get("iteration_budgets") is None, "Repair comparison requires single-budget runs")
        options.setdefault("iteration_budgets", None)
    _require(after_options.get("initialization_spectrum") == "projected"
             and after_options.get("refinement_solver") == "anchor_projected", "Repair policies not enabled")
    for key in POLICY_KEYS:
        before_options.pop(key, None)
        after_options.pop(key, None)
    _require(before_options == after_options, "Tuning grid or iteration budget changed")
    _require({k: v for k, v in before_config.items() if k != "runner"}
             == {k: v for k, v in after_config.items() if k != "runner"}, "Non-policy configuration changed")
    for record in (old, new):
        identity = {key: record[key] for key in ("schema_version", "method", "model", "experiment",
            "rd_seed_id", "random_seed", "setting", "configuration", "generator_arguments")}
        _require(record["configuration_fingerprint"] == _digest_json(identity), "Configuration fingerprint mismatch")
        combined = dict(training_observed=record[HASH_KEYS[0]], validation_observed=record[HASH_KEYS[1]],
                        evaluation_truth=record[HASH_KEYS[2]])
        _require(record["input_fingerprint"] == _digest_json(combined), "Combined input fingerprint mismatch")
    budget = after_options["iterations"]
    before, after = old["selection_history"], new["selection_history"]
    _require(len(before) == len(after) == after_config["candidate_count"], "Candidate budget changed or incomplete")
    repaired = 0
    correction_norms = []
    for index, (a, b) in enumerate(zip(before, after)):
        _require(a["candidate_id"] == b["candidate_id"] == index and a["params"] == b["params"],
                 "Candidate grid/order changed")
        _validation_candidate(b, budget)
        diag = b.get("diagnostics", {})
        if "initialization_spectrum_repaired" in diag:
            did_repair = diag["initialization_spectrum_repaired"]
            _require(type(did_repair) is bool, "Invalid initialization repair flag")
            original = np.asarray(diag["initialization_singular_values_original"], dtype=float)
            projected = np.asarray(diag["initialization_singular_values_projected"], dtype=float)
            _require(original.shape == projected.shape == (new["setting"]["target_rank"],)
                     and np.isfinite(original).all() and np.isfinite(projected).all(), "Invalid initializer spectra")
            correction = _finite(diag["initialization_spectrum_correction_norm"], "spectral correction norm")
            _close(correction, float(np.linalg.norm(projected - original)), "Spectral correction norm mismatch")
            _require(did_repair == (not np.array_equal(original, projected)), "Spectral repair flag mismatch")
            _require(diag["initialization_spectrum"] == "projected", "Incorrect resolved initializer policy")
            _require(diag["refinement_solver"] == "anchor_projected", "Incorrect resolved refinement solver")
            repaired += int(did_repair)
            correction_norms.append(correction)
        elif b["success"]:
            raise ValueError("Successful candidate is missing initializer repair diagnostics")
    _require(new["fit_errors"] == [c for c in after if not c["success"]], "Candidate failure list mismatch")
    _require(type(new["success"]) is bool and new["success"] == (new["status"] == "complete"),
             "Result status/success mismatch")
    eligible = [c for c in after if c["success"]]
    if new["success"]:
        _require(bool(eligible), "Selected result has no eligible candidate")
        winner = min(eligible, key=lambda c: c["validation_mse"])
        _require(new["best_params"] == winner["params"] and new["selected_iteration"] == winner["selected_iteration"]
                 and new["n_iter"] == winner["n_iter"] and new["termination_reason"] == winner["termination_reason"],
                 "Selected model does not match minimum-validation winner")
        _close(new["validation_loss"], winner["validation_mse"], "Selected validation loss mismatch")
        _require(new["validation_history"] == winner["validation_history"], "Selected validation history mismatch")
        _finite(new["avg_err"], "coefficient error")
        coefficient = np.asarray(new["C_hat"], dtype=float)
        _require(coefficient.shape == (new["setting"]["p"], new["setting"]["q"])
                 and np.isfinite(coefficient).all(), "Invalid selected coefficient")
        _require(new["diagnostics"] == winner["diagnostics"], "Winner diagnostics mismatch")
    else:
        _require(not eligible and new["avg_err"] is None and new["C_hat"] is None,
                 "Failed result contains an eligible estimate or coefficient error")
    history = new["history"]
    if history:
        _require([h["iteration"] for h in history] == list(range(new["n_iter"] + 1)), "Incomplete optimization history")
        objective = [_finite(h["objective"], "accepted objective") for h in history]
        _require(not np.any(np.diff(objective) > 1e-10), "Accepted objective increased")
        floor = after_config["margins"]["anchor_min"]
        for record in history:
            _require(min(_finite(record["anchor_min_u"], "left anchor"),
                         _finite(record["anchor_min_v"], "right anchor")) >= floor - 1e-12,
                     "Accepted anchor violates declared floor")
            _close(record["objective"], record["smooth_loss"] + record["penalty_value"], "Objective decomposition mismatch")
    return dict(model=new["model"], experiment=new["experiment"], setting=new["setting"]["suffix"],
        seed_id=new["rd_seed_id"], old_status=old["status"], status=new["status"],
        old_error=old["avg_err"], error=new["avg_err"], old_validation_loss=old["validation_loss"],
        validation_loss=new["validation_loss"], termination_reason=new["termination_reason"],
        selected_iteration=new["selected_iteration"], n_iter=new["n_iter"],
        candidate_failures=len(new["fit_errors"]), candidate_count=len(after),
        candidate_statuses=dict(Counter(c["status"] for c in after)), initialization_repaired=repaired,
        maximum_initialization_correction=max(correction_norms, default=0.),
        optimization_converged=new["diagnostics"].get("optimization_converged", False),
        selected_converged=new["diagnostics"].get("selected_converged", False),
        selected_projected_gradient_norm=new["diagnostics"].get("projected_gradient_norm"),
        last_projected_gradient_norm=new["diagnostics"].get("last_projected_gradient_norm"),
        data_and_budget_verified=True,
        fingerprints={key: new[key] for key in HASH_KEYS})


def build_audit(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    _require(not manifest.get("errors"), "Regression manifest reports worker errors")
    _require(bool(manifest.get("finished")), "Regression manifest is not finished")
    _require(manifest["expected_cells"] == 66 and manifest["groups"] == {"previous_failure": 60, "control": 6},
             "Unexpected targeted regression design")
    _require(len(manifest["cells"]) == 66, "Missing regression manifest cells")
    baseline, output = Path(manifest["baseline_root"]).resolve(), Path(manifest["output_root"]).resolve()
    _require(output == manifest_path.parent and baseline != output, "Incorrect manifest result roots")
    old_records = {p.relative_to(baseline): json.loads(p.read_text()) for p in baseline.glob(PATTERN)}
    _require(len(old_records) == 360, "Expected complete 360-cell original pilot")
    _require(sum(not r["applicable"] for r in old_records.values()) == 45, "Original inapplicable count changed")
    expected = {}
    for path, old in old_records.items():
        if old["applicable"] and not old["success"]:
            expected[path] = "previous_failure"
        elif old["experiment"] == "exp4" and old["rd_seed_id"] == 0 and old["setting"]["sigma0"] in (0., .01):
            expected[path] = "control"
    _require(Counter(expected.values()) == {"previous_failure": 60, "control": 6}, "Original selected groups changed")
    _require({p.relative_to(output) for p in output.glob(PATTERN)} == set(expected), "Missing or unexpected repair result files")
    cells, seen, implementation = [], set(), set()
    for entry in manifest["cells"]:
        path = Path(entry["path"]).resolve()
        _require(output in path.parents, "Manifest cell path is outside repair root")
        relative = path.relative_to(output)
        _require(relative in expected and relative not in seen, "Unexpected or duplicate manifest cell")
        seen.add(relative)
        new = json.loads(path.read_text())
        cell = validate_pair(old_records[relative], new)
        cell.update(group=expected[relative], path=str(path), baseline_path=str(baseline / relative))
        for key in ("model", "experiment", "setting", "seed_id", "group", "old_status", "status", "old_error",
                    "error", "old_validation_loss", "validation_loss", "termination_reason", "selected_iteration",
                    "n_iter", "candidate_failures", "candidate_statuses", "initialization_repaired", "data_and_budget_verified"):
            _require(entry[key] == cell[key], f"Manifest/artifact mismatch: {key}")
        _require(cell["candidate_count"] == 9 and new["configuration"]["runner"]["iterations"] == 500,
                 "Expected original nine-candidate, 500-iteration budget")
        implementation.add(new["implementation_fingerprint"])
        cells.append(cell)
    _require(seen == set(expected), "Missing audited regression cells")
    _require(len(implementation) == 1, "Repair artifacts use different implementations")
    statuses = dict(Counter(c["status"] for c in cells))
    _require(manifest["status_counts"] == statuses, "Manifest status counts mismatch")
    failures = [c for c in cells if c["group"] == "previous_failure"]
    controls = [c for c in cells if c["group"] == "control"]
    return dict(valid=True, scope="targeted_solver_regression_selected_on_prior_failures_not_unbiased_benchmark",
        manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        baseline_root=str(baseline), output_root=str(output), expected_cells=66, audited_cells=len(cells),
        prior_failure_cells=60, control_cells=6, original_inapplicable_cells_unchanged=45,
        rescued_prior_failures=sum(c["status"] == "complete" for c in failures),
        unresolved_prior_failures=sum(c["status"] != "complete" for c in failures),
        successful_controls=sum(c["status"] == "complete" for c in controls), status_counts=statuses,
        optimizer_stationary_cells=sum(c["optimization_converged"] for c in cells),
        selected_stationary_cells=sum(c["selected_converged"] for c in cells),
        iteration_cap_cells=sum(c["termination_reason"] == "max_iterations" for c in cells),
        candidate_count=sum(c["candidate_count"] for c in cells),
        candidate_failures=sum(c["candidate_failures"] for c in cells),
        repaired_candidate_initializers=sum(c["initialization_repaired"] for c in cells),
        repaired_initializer_cells=sum(c["initialization_repaired"] > 0 for c in cells),
        candidate_status_counts=dict(sum((Counter(c["candidate_statuses"]) for c in cells), Counter())),
        implementation_fingerprint=next(iter(implementation)),
        checks=dict(manifest_complete=True, matching_observed_inputs_and_truth=True,
            unchanged_candidate_grids_budgets_and_splits=True, accepted_objective_monotone=True,
            accepted_anchor_floors_respected=True, minimum_validation_candidate_and_iterate=True,
            convergence_separated_from_completion=True, projection_diagnostics_verified=True),
        wall_seconds=manifest["wall_seconds"], cells=sorted(cells, key=lambda c:(c["group"], c["model"], c["experiment"], c["setting"], c["seed_id"])))


def _number(value):
    return "—" if value is None else f"{value:.6f}"


def render_markdown(audit):
    cells = audit["cells"]
    lines = ["# SparseSMART targeted solver regression", "",
        f"Of 60 previously failed applicable cells, **{audit['rescued_prior_failures']} now completed successfully**; "
        f"{audit['unresolved_prior_failures']} remain unresolved. "
        f"{audit['successful_controls']} of six previously successful controls completed successfully.", "",
        "This is a targeted regression selected on earlier failures, not a new unbiased benchmark or a rerun of the full paper comparison. "
        "Only SparseSMART was fitted. The 45 structurally inapplicable original cells remain unchanged.", "",
        "Every pair retains the same training observations, independent 100-observation tuning sample, coefficient truth, "
        "nine candidate penalty configurations, and 500-iteration cap. The changes are feasible spectral initialization "
        "and projection of the full fixed-chart constraints. The original objective and anchor floor are preserved.", "",
        f"**Convergence:** {audit['optimizer_stationary_cells']} terminal optimizers and "
        f"{audit['selected_stationary_cells']} selected iterates met the stationarity test; "
        f"{audit['iteration_cap_cells']} cells stopped at the 500-iteration cap. Completion at the cap does not establish convergence.", "",
        f"Spectral initialization was repaired for {audit['repaired_candidate_initializers']} of {audit['candidate_count']} candidates "
        f"across {audit['repaired_initializer_cells']} cells. {audit['candidate_failures']} candidate fits failed.", "",
        "## Previously failed cells", "",
        "Errors below are means among successful repairs of the selected prior failures; they are not full-grid averages.", "",
        "| Model | Experiment / setting | Prior failures | Successful repairs | Terminal stationary | Still unresolved | Mean repaired error |",
        "|---|---|---:|---:|---:|---:|---:|"]
    groups = defaultdict(list)
    for cell in cells:
        if cell["group"] == "previous_failure":
            groups[(cell["model"], cell["experiment"], cell["setting"])].append(cell)
    for (model, experiment, setting), group in sorted(groups.items()):
        success = [c for c in group if c["status"] == "complete"]
        error = float(np.mean([c["error"] for c in success])) if success else None
        lines.append(f"| {model} | {experiment}: {setting} | {len(group)} | {len(success)} | "
                     f"{sum(c['optimization_converged'] for c in group)} | {len(group)-len(success)} | {_number(error)} |")
    lines += ["", "The fitted-rank and source-rank experiments change estimator dimensions; true target/source ranks remain 5/10. "
        "SparseSMART source-rank sensitivity uses the paper grid but has a different parameter role from paper SMART's unpenalized source directions.",
        "", "## Six seed-zero controls", "",
        "| Model | Source noise | Old error | New error | Error change | New status | Selected / terminal iteration |",
        "|---|---|---:|---:|---:|---|---|"]
    for c in cells:
        if c["group"] == "control":
            change = None if c["error"] is None else c["error"]-c["old_error"]
            lines.append(f"| {c['model']} | {c['setting']} | {_number(c['old_error'])} | {_number(c['error'])} | "
                         f"{_number(change)} | {c['status']} | {c['selected_iteration']} / {c['n_iter']} |")
    lines += ["", "## High-noise failures at seed zero", "",
        "These two rows concern the previously failed source-noise 0.5 cases in Models II and III.", "",
        "| Model | Old status | New status | New error | Validation loss | Selected / terminal iteration | Terminal constrained residual |",
        "|---|---|---|---:|---:|---|---:|"]
    for c in cells:
        if c["group"] == "previous_failure" and c["experiment"] == "exp4" and c["setting"] == "sigma0=0.5":
            lines.append(f"| {c['model']} | {c['old_status']} | {c['status']} | {_number(c['error'])} | "
                         f"{_number(c['validation_loss'])} | {c['selected_iteration']} / {c['n_iter']} | "
                         f"{_number(c['last_projected_gradient_norm'])} |")
    unresolved = [c for c in cells if c["status"] != "complete"]
    lines += ["", "## Remaining status", ""]
    if unresolved:
        lines += ["| Model | Experiment / setting | Seed | Status |", "|---|---|---:|---|"]
        lines += [f"| {c['model']} | {c['experiment']}: {c['setting']} | {c['seed_id']} | {c['status']} |" for c in unresolved]
    else:
        lines.append("No targeted cell remains failed. Finite-iteration and local-stationarity limitations still apply.")
    lines += ["", "All artifact checks passed: manifest completeness, matching observed-data/truth fingerprints, unchanged tuning budgets and splits, "
        "nonincreasing accepted objectives, accepted anchor floors, minimum-validation selection, and projection diagnostics. "
        "No data were regenerated or estimators fitted by this summary.", "",
        "The machine-readable [repair audit](repair_audit.json) includes each paired outcome, hashes, candidate statuses, and convergence diagnostics.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=HERE / "result/sparse_smart_repairs")
    args = parser.parse_args(argv)
    audit = build_audit(args.result_root / "repair_manifest.json")
    (args.result_root / "repair_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.result_root / "repair_summary.md").write_text(render_markdown(audit))
    print(json.dumps({key:audit[key] for key in ("valid", "audited_cells", "rescued_prior_failures",
        "unresolved_prior_failures", "successful_controls", "optimizer_stationary_cells", "iteration_cap_cells")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
