#!/usr/bin/env python3
"""Read-only adaptive-chart artifact audit; stdout JSON, no fitting or writes.

Run with the project's Python environment. Results-root must contain its plan,
manifest, preparation, task outputs and case metadata. Cases-root supplies the
shared case NPZ files; it may belong to the original campaign. A frozen source
directory must exist at results-root/source or results-root.parent/source.
Use --task-ids to audit only an explicit set of campaign task IDs; omitted
means all task directories. Requested missing tasks fail explicitly. Imports
prefer the frozen code snapshot beside this script when it is available.

Uses schema 2 for chart states, schema 3 for direct physical RRR endpoints. Passing
does not change tuning eligibility, certify convergence, or independently
verify unsaved intermediate states/validation predictions or switch endpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
CODE = SCRIPT.parents[1]
for folder in (CODE / "simulation", CODE / "sparse-smart/src", CODE / "sparse-smart-v2/src"):
    sys.path.insert(0, str(folder))

import numpy as np
from sparse_smart.chart import AnchorChart
from sparse_smart_v2.support import FreeRows, choose_free_rows


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_fingerprint(data, names):
    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(data[name])
        digest.update(json.dumps((name, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _fingerprints(data):
    return dict(training_observed_input_fingerprint=_array_fingerprint(data, ("X", "Y", "C0")),
                validation_observed_input_fingerprint=_array_fingerprint(data, ("X_validation", "Y_validation")),
                evaluation_truth_fingerprint=_array_fingerprint(data, ("C_star",)))


def _metrics(coefficient, data):
    error = coefficient - data["C_star"]
    return dict(coefficient_rmse=float(np.linalg.norm(error) / np.sqrt(error.size)),
                coefficient_frobenius_squared=float(np.sum(error * error)),
                training_prediction_mse=float(np.mean((data["X"] @ error) ** 2)),
                validation_mse=float(np.mean((data["Y_validation"] - data["X_validation"] @ coefficient) ** 2)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative_file(root, name):
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), f"unsafe relative path: {name}")
    return path


def check_hash(path, expected):
    require(isinstance(expected, str) and sha(path) == expected, f"SHA256 mismatch: {path}")


def close(actual, expected, name, *, atol=2e-9, rtol=2e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            f"nonfinite value or shape mismatch: {name}")
    require(np.allclose(a, b, atol=atol, rtol=rtol), f"numerical mismatch: {name}")


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        require(value.dtype.kind in "biuf" and np.isfinite(value).all(), f"nonfinite/nonreal NPZ array: {key}")
    return arrays


def load_provenance(root):
    plan = read(root / "plan.json")
    fingerprint = plan["plan_fingerprint"]
    require(_digest({k: v for k, v in plan.items() if k != "plan_fingerprint"}) == fingerprint,
            "invalid canonical plan fingerprint")
    require(plan["n_cases"] == len(plan["cases"]) and plan["n_tasks"] == len(plan["tasks"]),
            "inconsistent plan counts")
    check_hash(root / "source-manifest.json", plan["source_manifest_sha256"])
    manifest = read(root / "source-manifest.json")
    source = next((p for p in (root / "source", root.parent / "source") if p.is_dir()), None)
    require(source is not None, "frozen source snapshot unavailable for source-file hash verification")
    for name, expected in manifest["files"].items():
        check_hash(relative_file(source, name), expected)
    prep = read(root / "preparation.json")
    require(prep["success"] and prep["status"] == "complete", "preparation not complete")
    require(prep["plan_sha256"] == sha(root / "plan.json") and
            prep["source_manifest_sha256"] == plan["source_manifest_sha256"], "preparation provenance mismatch")
    require(prep["n_cases"] == plan["n_cases"] and prep["n_tasks"] == plan["n_tasks"],
            "preparation count mismatch")
    if plan.get("reference_cases_sha256"):
        check_hash(root / "reference-cases.json", plan["reference_cases_sha256"])
    return plan, prep, source, len(manifest["files"])


def load_case(root, cases_root, case, prep_case, plan):
    cid = case["case_id"]
    # New case.json belongs to the new plan; its NPZ files can be read in-place
    # from an older download only after their exact hashes match the new record.
    meta_path = root / "cases" / cid / "case.json"
    check_hash(meta_path, prep_case["case_json_sha256"])
    meta = read(meta_path)
    require(meta["case"] == case and meta["plan_fingerprint"] == plan["plan_fingerprint"],
            "case metadata identity mismatch")
    require(meta["files"] == prep_case["files"] and meta["initializers"] == prep_case["initializers"],
            "case metadata/preparation mismatch")
    array_root = cases_root / "cases" / cid
    if Path(cases_root).resolve() == Path(root).resolve():
        # The caller already verified this preparation record. Reusing the
        # checked metadata avoids parsing the whole campaign preparation once
        # per case when auditing in place.
        old_meta = meta
    else:
        old_meta = read(array_root / "case.json")
        old_preparation = read(cases_root / "preparation.json")
        old_record = next(row for row in old_preparation["cases"] if row["case_id"] == cid)
        check_hash(array_root / "case.json", old_record["case_json_sha256"])
    require(old_meta["case"] == case and old_meta["files"] == meta["files"],
            "shared old/new case identity or file hashes differ")
    for name, expected in meta["files"].items():
        check_hash(relative_file(array_root, name), expected)
    data = load_npz(array_root / "data.npz")
    source = load_npz(array_root / "source.npz")
    n, p, q, nv = case["n_train"], case["p"], case["q"], plan["configuration"]["n_validation"]
    for key, shape in {"X": (n, p), "Y": (n, q), "C0": (p, q), "C_star": (p, q),
                       "X_validation": (nv, p), "Y_validation": (nv, q)}.items():
        require(data[key].shape == shape, f"case array shape mismatch: {key}")
    require(_fingerprints(data) == meta["fingerprints"] == old_meta["fingerprints"],
            "training/validation/truth fingerprint mismatch")
    fingerprint = _array_fingerprint(source, tuple(sorted(source)))
    require(fingerprint == meta["source_fingerprint"] == old_meta["source_fingerprint"],
            "source-frame fingerprint mismatch")
    r0 = case["initializer_source_rank"]
    close(source["left"].T @ source["left"], np.eye(p), "source left orthogonality")
    close(source["right"].T @ source["right"], np.eye(q), "source right orthogonality")
    close(source["leading_left"], source["left"][:, :r0], "source leading left")
    close(source["leading_right"], source["right"][:, :r0], "source leading right")
    singular = source["source_singular_values"]
    close((source["left"][:, :len(singular)] * singular) @ source["right"][:, :len(singular)].T,
          data["C0"], "observed source SVD reconstruction")
    return data, source, meta


def validation_winner(rows):
    incumbent = None
    previous = -1
    for row in rows:
        iteration = row["iteration"]
        require(isinstance(iteration, int) and iteration > previous, "invalid validation iteration order")
        require(np.isfinite(row["loss"]) and row["loss"] >= 0, "invalid validation loss")
        require(row["incumbent_iteration"] == incumbent, "inconsistent validation incumbent transcript")
        delta = row["selection_loss_difference"]
        if incumbent is None:
            require(delta is None, "first validation comparison must have no incumbent")
            incumbent = iteration
        else:
            require(delta is not None and np.isfinite(delta), "invalid paired validation difference")
            if delta < 0:
                incumbent = iteration
        previous = iteration
    return incumbent


GEOMETRY_FIELDS = ("anchors_u", "anchors_v", "center_u", "center_v")


def saved_chart(arrays, prefix, case):
    """Require explicit geometry; never infer an earlier state from aliases."""
    names = [prefix + "_" + field for field in GEOMETRY_FIELDS]
    require(all(name in arrays for name in names), f"missing own-chart geometry: {prefix}")
    chart = AnchorChart(case["p"], case["q"], *(arrays[name] for name in names))
    require(chart.rank == case["rank"], f"wrong saved chart rank: {prefix}")
    return chart


def fixed_free_rows(chart, physical_rows):
    masks = []
    for side, rows in zip(("u", "v"), physical_rows):
        require(np.isin(getattr(chart, f"anchors_{side}"), rows).all(),
                f"new anchors leave the physical free rows: {side}")
        complement = getattr(chart, f"complement_{side}")
        masks.append(np.repeat((~np.isin(complement, rows))[:, None], chart.rank, axis=1))
    return FreeRows(*physical_rows, *masks)


def switch_epochs(result, config, initial_chart, physical_rows):
    """Check the saved switch transcript; not a proof of unsaved invariants."""
    require(config["chart_continuation_rule"] == result["metadata"]["chart_continuation_rule"] ==
            "restricted_neighbor_fallback_numerical_gain_and_rotation_recenter_v4",
            "plan/result chart continuation rule mismatch")
    events = result["anchor_switches"]
    require(isinstance(events, list), "anchor switch transcript must be a list")
    require(events == result["metadata"]["anchor_switches"], "switch transcript/metadata mismatch")
    require(len(events) <= config["max_anchor_switches"], "anchor switch cap exceeded")
    require(config["adaptive_anchors"] or not events, "switches present with adaptive anchors disabled")
    epochs = [{side: {"anchors": getattr(initial_chart, f"anchors_{side}").tolist(),
                     "center_sha256": hashlib.sha256(np.asarray(getattr(initial_chart, f"center_{side}"),
                                                   dtype="<f8").tobytes(order="C")).hexdigest()}
               for side in ("u", "v")}]
    previous = -1
    for index, event in enumerate(events, 1):
        t = event["iteration"]
        require(isinstance(t, int) and previous <= t <= result["n_iter"], "switch iteration sequence mismatch")
        require(event["switch_index"] == index and event["kind"] == "adaptive_anchor_switch",
                "switch index/kind mismatch")
        require(event["rule"] == config["chart_continuation_rule"] ==
                "restricted_neighbor_fallback_numerical_gain_and_rotation_recenter_v4", "unknown chart continuation rule")
        require(event["previous_status"] in ("line_search_failed", "numerical_stagnation")
                and event["previous_rejection"] in (
                    "left anchor singular value is below anchor_min", "right anchor singular value is below anchor_min",
                    "left Cayley coordinate has operator norm above 1/2", "right Cayley coordinate has operator norm above 1/2"),
                "switch lacks an eligible chart-failure trigger")
        require(event["free_rows_unchanged"] is True and event["outside_weighted_coordinates_unchanged"] is True
                and event["initializer_certificate"] is False, "incorrect switch invariant claims")
        require(event["anchor_min"] == config["margins"]["anchor_min"]
                and event["trigger_multiple"] == 1.25 and event["acceptance_multiple"] is None
                and event["acceptance_reference"] == "current_anchor_margin"
                and event["relative_margin_multiple"] == 1.0 and event["minimum_relative_improvement"] == 0.0
                and event["numerical_gain_epsilon_multiple"] == 256
                and event["strict_gain_comparison"] is True
                and event["maximum_primary_search_starts"] == 2
                and event["maximum_fallback_search_starts"] == {s: initial_chart.rank * (len(rows)-initial_chart.rank) for s,rows in zip(("u","v"), physical_rows)}
                and event["maximum_search_starts"] == {s: 2 + initial_chart.rank * (len(rows)-initial_chart.rank) for s,rows in zip(("u","v"), physical_rows)}
                and event["fallback_trigger_rule"] == "primary_no_positive_numerical_gain"
                and event["fallback_stop_at_first_acceptable"] is True
                and event["swap_round_limit_scope"] == "per_start"
                and event["search_starts"] == ["current_anchors", "restricted_cpqr", "current_anchor_one_exchange"]
                and event["rotation_trigger"] == .45
                and event["maximum_swap_rounds"] == 16, "anchor switch thresholds changed")
        new, switched = {}, []
        for side, rows in zip(("u", "v"), physical_rows):
            detail = event["sides"][side]
            require(detail["old_anchors"] == epochs[-1][side]["anchors"], f"broken switch anchor chain: {side}")
            require(detail["old_center_sha256"] == epochs[-1][side]["center_sha256"], f"broken center chain: {side}")
            next_rows = detail["new_anchors"]
            next_hash = detail["new_center_sha256"]
            require(isinstance(next_hash, str) and len(next_hash) == 64
                    and all(c in "0123456789abcdef" for c in next_hash), "invalid center hash")
            new[side] = {"anchors": next_rows, "center_sha256": next_hash}
            require(event["free_rows"][side] == rows.tolist(), f"switch changes free rows: {side}")
            require(len(next_rows) == initial_chart.rank and sorted(set(next_rows)) == next_rows
                    and all(row in rows for row in next_rows), f"invalid switch anchors: {side}")
            old_margin, new_margin = detail["old_min_singular_value"], detail["new_min_singular_value"]
            require(np.isfinite(old_margin) and np.isfinite(new_margin) and min(old_margin, new_margin) > 0,
                    f"invalid switch anchor margins: {side}")
            rows_changed = next_rows != epochs[-1][side]["anchors"]
            action = detail["action"]
            require(action in ("none", "anchor_switch", "rotation_recenter", "anchor_switch_and_recenter"), "invalid chart action")
            changed = action != "none"
            require(detail["switched"] == changed, f"incorrect switched flag: {side}")
            require(rows_changed == (action in ("anchor_switch", "anchor_switch_and_recenter")), "anchor action mismatch")
            old_rotation, new_rotation = detail["old_cayley_operator_norm"], detail["new_cayley_operator_norm"]
            require(np.isfinite(old_rotation) and 0 <= old_rotation <= .5 + 1e-12, "invalid old rotation norm")
            rotation_triggered = old_rotation >= event["rotation_trigger"]
            require(rotation_triggered == (action in ("rotation_recenter", "anchor_switch_and_recenter")),
                    "chart action disagrees with rotation trigger")
            if changed:
                switched.append(side)
                slack = 64 * np.finfo(float).eps * max(1., old_margin)
                require(new_rotation == 0., "changed chart was not recentered")
                if rows_changed:
                    require(old_margin <= 1.25 * event["anchor_min"] + slack
                            and new_margin >= event["anchor_min"]
                            and new_margin > old_margin + 256 * np.finfo(float).eps * max(1., old_margin),
                            f"switch margin rule violated: {side}")
                if action in ("rotation_recenter", "anchor_switch_and_recenter"):
                    require(old_rotation >= .45 - slack, "rotation recenter below trigger")
            else:
                require(old_margin == new_margin, f"unchanged side has changed margin: {side}")
                require(old_rotation == new_rotation and detail["old_center_sha256"] == next_hash,
                        "unchanged side has changed rotation/center")
            if not rows_changed:
                require(old_margin == new_margin, "fixed anchor has changed margin")
            require(isinstance(detail["swap_rounds"], int) and 0 <= detail["swap_rounds"] <= 16,
                    f"invalid row-swap work: {side}")
            require(rows_changed or detail["swap_rounds"] == 0, "row-swap work without an anchor replacement")
            search = detail["anchor_search"]
            anchor_triggered = old_margin <= 1.25 * event["anchor_min"] + 64 * np.finfo(float).eps * max(1., old_margin)
            require((search is not None) == anchor_triggered, "anchor search trigger mismatch")
            if search is not None:
                require(type(search["fallback_triggered"]) is bool and type(search["fallback_accepted"]) is bool,
                        "fallback flags must be booleans")
                max_fallback = initial_chart.rank * (len(rows) - initial_chart.rank)
                require(search["maximum_fallback_starts"] == max_fallback
                        and search["maximum_search_starts"] == 2 + max_fallback,
                        "anchor search work bound mismatch")
                primary, fallback = search["primary_starts"], search["fallback_starts"]
                require(type(primary) is int and type(fallback) is int
                        and (primary == 0 if max_fallback == 0 else 1 <= primary <= 2)
                        and 0 <= fallback <= max_fallback, "invalid anchor search counts")
                primary_margin = search["primary_best_margin"]
                require(np.isfinite(primary_margin) and primary_margin >= old_margin - 1e-12,
                        "invalid primary anchor margin")
                gain_threshold = old_margin + 256 * np.finfo(float).eps * max(1., old_margin)
                require(search["fallback_triggered"] == bool(max_fallback and primary_margin <= gain_threshold),
                        "unnecessary or missing fallback search")
                if not search["fallback_triggered"]:
                    require(fallback == 0 and search["fallback_accepted"] is False,
                            "fallback work without fallback trigger")
                require(type(search["winning_swap_rounds"]) is int
                        and 0 <= search["winning_swap_rounds"] <= 16, "invalid winning search work")
                winner, seed = search["winning_start"], search["winning_seed"]
                if max_fallback == 0:
                    require(winner is None and seed is None and not search["fallback_accepted"],
                            "search winner with no alternative anchor")
                else:
                    require(winner in event["search_starts"] and isinstance(seed, list)
                            and len(seed) == initial_chart.rank and sorted(set(seed)) == seed
                            and all(row in rows for row in seed), "invalid winning anchor seed")
                    if winner == "current_anchors":
                        require(seed == detail["old_anchors"], "current-anchor seed changed")
                    if winner == "current_anchor_one_exchange":
                        require(search["fallback_triggered"] and fallback > 0
                                and len(set(seed).symmetric_difference(detail["old_anchors"])) == 2,
                                "fallback seed is not one current-anchor exchange")
                if search["fallback_accepted"]:
                    require(search["fallback_triggered"] and fallback > 0
                            and winner == "current_anchor_one_exchange" and rows_changed,
                            "accepted fallback did not change anchor")
                if rows_changed:
                    require(not search["fallback_triggered"] or search["fallback_accepted"] is True,
                            "fallback anchor change was not marked accepted")
                    require(detail["swap_rounds"] == search["winning_swap_rounds"],
                            "anchor change/search work mismatch")
            discrepancy = event["factor_frobenius_differences"][side]
            require(np.isfinite(discrepancy) and 0 <= discrepancy <= 1e-8,
                    f"reported factor preservation error: {side}")
        require(switched and event["switched_sides"] == switched, "switch changed no side or mislabels sides")
        epochs.append(new)
        previous = t
    return events, epochs


def check_chart_epoch(chart, epoch, *, name):
    """Bind a saved state's full chart geometry to its verified event epoch."""
    for side in ("u", "v"):
        require(getattr(chart, f"anchors_{side}").tolist() == epoch[side]["anchors"],
                f"saved geometry differs from switch epoch: {name}/{side}")
        digest = hashlib.sha256(np.asarray(getattr(chart, f"center_{side}"),
                                          dtype="<f8").tobytes(order="C")).hexdigest()
        require(digest == epoch[side]["center_sha256"], "saved center differs from declared chart epoch")


def smooth_gradient_norm(chart, state, design, response, solver):
    """Independently reconstruct the declared smooth-gradient coordinate metric."""
    require(solver in ("masked_chart_spectral_soft_hard", "masked_anchor_projected"),
            "unsupported refinement solver in audit")
    smooth, gradient = chart.value_gradient(state, design, response)
    if solver == "masked_anchor_projected":
        _, _, d, zu, zv = chart.unpack(state)
        gu, gv, gd, gzu, gzv = chart.unpack(gradient)
        hu, hv = zu / d, zv / d
        gd = gd + np.sum(gzu * hu, axis=0) + np.sum(gzv * hv, axis=0)
        gradient = chart.pack(gu, gv, gd, gzu * d, gzv * d)
    return smooth, float(np.linalg.norm(gradient))


def audit_rrr(result, arrays, history, data, case, config, initial, report):
    """Independent physical-coefficient rank, optimum and minimum-norm checks.

    This deliberately does not call the fitted estimator or its RRR solve.
    Singular vectors need not agree when response singular values tie.
    """
    require(config.get("rrr_shortcut", False) is True, "RRR output was not enabled in frozen plan")
    rank, p, q = case["rank"], case["p"], case["q"]
    free, caps = case["free_directions"], case["support_limits"]
    penalties = [result["task"]["penalty_u"], result["task"]["penalty_v"]]
    require(all(cap == (dim-count)*rank for dim,count,cap in zip((p,q),free,caps)),
            "RRR output with restrictive hard caps")
    require(all(lam == 0 or count == dim for lam,count,dim in zip(penalties,free,(p,q))),
            "RRR output with a nonzero effective penalty")
    metadata = result["metadata"]
    require(metadata.get("fit_method") == "target_rrr" and
            metadata.get("refinement_solver") == "target_rrr_closed_form", "RRR method metadata mismatch")
    require(metadata.get("requested_refinement_solver") == config.get("refinement_solver", "masked_chart_spectral_soft_hard"),
            "RRR requested solver differs from frozen plan")
    require(result["success"] and result["fit_status"] == "converged" and result["optimization_converged"]
            and result["termination_reason"] == "target_rrr_closed_form", "RRR completion metadata mismatch")
    require(result["n_iter"] == result["selected_iteration"] == 0 and not result["anchor_switches"],
            "RRR result reports a chart trajectory")
    require(result["initialization_cache"] == "not_used_target_rrr" and
            result["initialization_preflight_bypassed"] == (not initial["eligible"]),
            "RRR initializer bypass metadata mismatch")
    schema = arrays.get("states_schema_version")
    require(schema is not None and schema.shape == () and schema.dtype.kind in "iu" and schema.item() == 3,
            "RRR audit requires physical coefficient schema 3")
    expected_arrays = {"states_schema_version", "checkpoint_0_coefficient"}
    expected_arrays.update(f"{prefix}_{field}" for prefix in ("selected", "terminal")
                           for field in ("coefficient", "P", "d", "Q"))
    require(set(arrays) == expected_arrays, "RRR artifact has missing or unexpected chart/state arrays")
    C = arrays["selected_coefficient"]
    require(C.shape == (p,q), "RRR coefficient dimensions mismatch")
    for prefix in ("selected", "terminal"):
        left, d, right = (arrays[f"{prefix}_{field}"] for field in ("P", "d", "Q"))
        require(d.ndim == 1 and len(d) <= rank and left.shape == (p,len(d)) and right.shape == (q,len(d)),
                "RRR physical factor dimensions mismatch")
        require(np.all(d >= 0) and np.all(np.diff(d) <= 0), "RRR coefficient spectrum is not nonnegative ordered")
        close(left.T@left, np.eye(len(d)), "RRR left orthogonality")
        close(right.T@right, np.eye(len(d)), "RRR right orthogonality")
        close((left*d)@right.T, C, "RRR physical factor reconstruction")
        close(arrays[f"{prefix}_coefficient"], C, "RRR selected/terminal coefficient")
        metrics = _metrics(C, data)
        require(set(metrics) == set(result[f"{prefix}_metrics"]), "RRR metric names mismatch")
        for name, value in metrics.items():
            close(value, result[f"{prefix}_metrics"][name], f"RRR {prefix} {name}")
    close(arrays["checkpoint_0_coefficient"], C, "RRR physical checkpoint")
    # Numerical design-rank policy is explicit and independent of the source.
    X, Y = data["X"], data["Y"]
    u, sx, vh = np.linalg.svd(X, full_matrices=False)
    cutoff = np.finfo(float).eps * max(X.shape) * (sx[0] if sx.size else 0.)
    keep = sx > cutoff
    basis, row_basis = u[:,keep], vh[keep].T
    reduced = basis.T@Y
    singular = np.linalg.svd(reduced, compute_uv=False)
    residual = Y - basis@reduced
    optimal_sse = float(np.sum(residual*residual) + np.sum(singular[rank:]**2))
    observed_sse = float(np.sum((Y-X@C)**2))
    tolerance = 2e-9 * max(1., float(np.sum(Y*Y)), optimal_sse)
    require(abs(observed_sse-optimal_sse) <= tolerance, "RRR training loss does not attain rank-constrained optimum")
    certificate = result["rrr_certificate"]
    require(certificate["certified"] is True and certificate["scope"] == "numerical_design_range_at_recorded_svd_tolerance"
            and certificate["coefficient_convention"] == "minimum_norm_lift_of_selected_fitted_response",
            "RRR optimality certificate/convention mismatch")
    close(certificate["design_rank_tolerance"], cutoff, "RRR certified design rank cutoff", atol=0.)
    close(certificate["projected_response_singular_values"], singular, "RRR certified projected spectrum")
    require(metadata["rrr_certificate"] == certificate, "RRR metadata certificate mismatch")
    close(metadata["training_loss"], observed_sse/(2*len(X)), "RRR metadata training loss")
    close(metadata["optimal_training_loss"], optimal_sse/(2*len(X)), "RRR metadata optimal training loss")
    close(metadata["objective_gap"], (observed_sse-optimal_sse)/(2*len(X)), "RRR metadata objective gap")
    close(C, row_basis@(row_basis.T@C), "RRR coefficient minimum-norm row-space convention")
    close(result["avg_err"], metrics["coefficient_rmse"], "RRR paper metric")
    close(metadata["validation_mse"], metrics["validation_mse"], "RRR metadata validation MSE")
    validation = history["validation_history"]
    require(len(validation) == 1 and validation_winner(validation) == 0, "RRR validation must report only the closed-form endpoint")
    close(validation[0]["loss"], metrics["validation_mse"], "RRR checkpoint validation MSE")
    require(set(history["checkpoints"]) == {"0"}, "RRR checkpoint sequence mismatch")
    checkpoint = history["checkpoints"]["0"]
    require(checkpoint["iteration"] == checkpoint["selected_iteration"] == 0 and
            checkpoint["termination_reason"] == "target_rrr_closed_form", "RRR checkpoint metadata mismatch")
    close(checkpoint["best_validation_loss"], metrics["validation_mse"], "RRR checkpoint validation loss")
    require(checkpoint["certificate"] == certificate, "RRR checkpoint certificate mismatch")
    return dict(report, success=True, fit_method="target_rrr", accepted_states_checked=0,
                physical_coefficients_checked=3, checkpoints_checked=1,
                rrr_optimality_checked=True, minimum_norm_checked=True,
                design_rank=int(np.count_nonzero(keep)), design_singular_cutoff=float(cutoff),
                training_sse=observed_sse, optimal_training_sse=optimal_sse)


def audit_task(task_root, plan, case_cache):
    result = read(task_root / "result.json")
    status = read(task_root / "status.json")
    check_hash(task_root / "result.json", status["result_sha256"])
    for key in ("schema_version", "method", "task", "case", "plan_fingerprint", "source_manifest_sha256",
                "success", "execution_success", "fit_status"):
        require(result.get(key) == status.get(key), f"result/status mismatch: {key}")
    tid = int(task_root.name)
    require(result["task"] == plan["tasks"][tid] and result["task"]["task_id"] == tid,
            "task index/plan mismatch")
    require(result["schema_version"] == plan["schema_version"] and result["method"] == plan["method"],
            "result schema/method mismatch")
    require(status["status"] == "finished" and result["status"] == "complete" and result["execution_success"],
            "task execution did not finish cleanly; endpoint audit cannot replace execution recovery")
    require(result["plan_fingerprint"] == plan["plan_fingerprint"] and
            result["source_manifest_sha256"] == plan["source_manifest_sha256"], "result provenance mismatch")
    require(result["task"]["case_id"] == result["case"]["case_id"], "task/case identity mismatch")
    data, source, meta = case_cache[result["case"]["case_id"]]
    require(result["case"] == meta["case"], "result case mismatch")
    require(result["fingerprints"] == meta["fingerprints"] and
            result["source_fingerprint"] == meta["source_fingerprint"], "result input fingerprint mismatch")
    initial = next(row for row in meta["initializers"] if row["init_penalty"] == result["task"]["init_penalty"])
    require(result["initialization_record"] == initial, "result initializer/preflight mismatch")
    for name, expected in result["files"].items():
        check_hash(relative_file(task_root, name), expected)
    report = dict(task_id=tid, case_id=result["case"]["case_id"],
                  original_success=result["success"], original_fit_status=result["fit_status"],
                  eligible_under_original_policy=bool(result["execution_success"] and result["success"]),
                  n_iter=result["n_iter"], selected_iteration=result.get("selected_iteration"))
    case, config = result["case"], plan["configuration"]
    expects_rrr = bool(config.get("rrr_shortcut", False) and
        all(cap == (dim-free)*case["rank"] for dim,free,cap in
            zip((case["p"],case["q"]),case["free_directions"],case["support_limits"])) and
        all(lam == 0 or free == dim for lam,free,dim in
            zip((result["task"]["penalty_u"],result["task"]["penalty_v"]),
                case["free_directions"],(case["p"],case["q"]))))
    require(not expects_rrr or result.get("metadata", {}).get("fit_method") == "target_rrr",
            "frozen plan requires RRR for this task; source initialization may not exclude it")
    if result.get("refinement_skipped") == "initializer_failed_strict_preflight":
        require(not initial["eligible"] and not result["success"] and result["avg_err"] is None,
                "invalid excluded-initializer result")
        require(result.get("selected_iteration") is None and result["n_iter"] == 0,
                "excluded initializer reports refinement output")
        return dict(report, success=True, accepted_states_checked=0, no_accepted_prefix=True)
    require({"states.npz", "history.json.gz"}.issubset(result["files"]), "unbound state/history artifacts")
    arrays = load_npz(task_root / "states.npz")
    history = json.loads(gzip.decompress((task_root / "history.json.gz").read_bytes()))
    case, config = result["case"], plan["configuration"]
    if result["metadata"].get("fit_method") == "target_rrr":
        return audit_rrr(result, arrays, history, data, case, config, initial, report)
    require(arrays.get("states_schema_version", np.asarray(0)).item() != 3, "unlabeled RRR physical output")
    solver = config.get("refinement_solver", "masked_chart_spectral_soft_hard")
    require(solver in ("masked_chart_spectral_soft_hard", "masked_anchor_projected"), "unsupported plan solver")
    require(result["metadata"].get("refinement_solver", "masked_chart_spectral_soft_hard") == solver,
            "result/plan refinement solver mismatch")
    expected_coordinates = "omega_d_H" if solver == "masked_anchor_projected" else "omega_d_Z"
    require(result["metadata"].get("optimization_coordinates", "omega_d_Z") == expected_coordinates,
            "result/plan optimization metric mismatch")
    if solver == "masked_anchor_projected":
        require(case["support_limits"] == [(dimension-free)*case["rank"] for dimension,free in
            zip((case["p"],case["q"]),case["free_directions"])], "anchor projected solver cannot impose restrictive hard caps")
    schema = arrays.get("states_schema_version")
    require(schema is not None and schema.shape == () and schema.dtype.kind in "iu" and schema.item() == 2,
            "adaptive audit requires scalar integer states_schema_version=2")
    initial_chart = saved_chart(arrays, "checkpoint_0_terminal", case)
    for side in ("u", "v"):
        require(getattr(initial_chart, f"anchors_{side}").tolist() == initial["metadata"]["anchors"][side],
                "initial checkpoint changed initializer anchor rows")
    initial_free = choose_free_rows(initial_chart, tuple(case["free_directions"]))
    physical_rows = (initial_free.rows_u, initial_free.rows_v)
    events, epochs = switch_epochs(result, config, initial_chart, physical_rows)
    for side in ("u", "v"):
        require(np.array_equal(arrays[f"free_rows_{side}"], getattr(initial_free, f"rows_{side}")),
                f"physical free rows changed after initialization: {side}")
    terminal_chart = saved_chart(arrays, "terminal", case)
    for field in GEOMETRY_FIELDS:
        require(np.array_equal(arrays[field], getattr(terminal_chart, field)),
                f"legacy terminal geometry alias mismatch: {field}")
    records, validation = history["history"], history["validation_history"]
    require([row["iteration"] for row in records] == list(range(result["n_iter"] + 1)),
            "accepted history iteration sequence mismatch")
    history_epochs = history["history_chart_epochs"]
    require(len(history_epochs) == len(records), "history chart epochs do not align with accepted records")
    for t, epoch in enumerate(history_epochs):
        require(isinstance(epoch, int) and sum(event["iteration"] < t for event in events) <= epoch
                <= sum(event["iteration"] <= t for event in events), "accepted record chart epoch outside iteration")
    selected = validation_winner(validation)
    require(selected == result["selected_iteration"], "selected iteration differs from validation transcript")
    require(validation[-1]["iteration"] <= result["n_iter"], "validation lies beyond accepted endpoint")
    for row in validation:
        epoch, t = row["chart_epoch"], row["iteration"]
        require(isinstance(epoch, int) and 0 <= epoch <= len(events), "invalid validation chart epoch")
        require(sum(event["iteration"] < t for event in events) <= epoch
                <= sum(event["iteration"] <= t for event in events), "validation chart epoch outside iteration")
    domain = {key: config["margins"][key] for key in ("d_lower", "d_upper", "gap", "anchor_min")}
    design, response = data["X"] @ source["left"], data["Y"] @ source["right"]
    penalties = (result["task"]["penalty_u"], result["task"]["penalty_v"])
    errors = dict(orthogonality=0., validation_mse=0.)
    retained_factors = {}
    count = 0

    def check_state(name, chart_prefix, chart_epoch, *, iteration=None, selected_iteration=None,
                    saved_factors=False, endpoint_record=None):
        nonlocal count
        chart = saved_chart(arrays, chart_prefix, case)
        require(isinstance(chart_epoch, int) and 0 <= chart_epoch < len(epochs), f"invalid saved chart epoch: {name}")
        check_chart_epoch(chart, epochs[chart_epoch], name=name)
        free = fixed_free_rows(chart, physical_rows)
        state = arrays[name]
        require(state.ndim == 1 and np.isfinite(state).all(), f"invalid state: {name}")
        reason = chart.domain_reason(state, **domain)
        require(reason is None, f"infeasible {name}: {reason}")
        P, d, Q = chart.reconstruct(state)
        physical_iteration = iteration if iteration is not None else selected_iteration
        require(physical_iteration is not None, f"saved state has no accepted iteration: {name}")
        if physical_iteration in retained_factors:
            for label, factor, other in zip(("P", "d", "Q"), (P, d, Q), retained_factors[physical_iteration]):
                close(factor, other, f"physical factors differ at iteration {physical_iteration}: {label}")
        else:
            retained_factors[physical_iteration] = (P.copy(), d.copy(), Q.copy())
        for side, factor in (("u", P), ("v", Q)):
            err = float(np.linalg.norm(factor.T @ factor - np.eye(case["rank"])))
            errors["orthogonality"] = max(errors["orthogonality"], err)
            close(factor.T @ factor, np.eye(case["rank"]), f"{name} orthogonality {side}")
        penalty = 0.
        for side, factor, lam, cap in zip(("u", "v"), (P, Q), penalties, case["support_limits"]):
            mask = getattr(free, f"penalized_{side}")
            block = state[getattr(chart, f"z_{side}_slice")].reshape(mask.shape)[mask]
            require(np.count_nonzero(block) <= cap, f"support cap exceeded: {name}/{side}")
            outside = np.setdiff1d(np.arange(len(factor)), getattr(free, f"rows_{side}"))
            physical = (factor[outside] * d).ravel()
            close(block, physical, f"{name} physical weighted outside entries {side}")
            require(np.array_equal(block == 0, physical == 0), f"{name} outside zeros changed {side}")
            penalty += lam * np.sum(np.abs(block), dtype=np.longdouble)
        if saved_factors:
            prefix = name.removesuffix("_state")
            for label, factor in (("P", P), ("d", d), ("Q", Q)):
                close(factor, arrays[f"{prefix}_{label}"], f"{name} stored factor {label}")
        coefficient = ((source["left"] @ P) * d) @ (source["right"] @ Q).T
        metrics = _metrics(coefficient, data)
        if iteration is not None:
            record = records[iteration]
            smooth = chart.loss(state, design, response)
            close(smooth, record["smooth_loss"], f"{name} smooth loss")
            close(float(penalty), record["penalty_value"], f"{name} penalty")
            close(smooth + float(penalty), record["objective"], f"{name} objective")
        if endpoint_record is not None:
            require(endpoint_record["iteration"] == physical_iteration, f"endpoint record iteration mismatch: {name}")
            smooth, gradient_norm = smooth_gradient_norm(chart, state, design, response, solver)
            for key, value in (("smooth_loss", smooth), ("penalty_value", float(penalty)),
                               ("objective", smooth + float(penalty)),
                               ("raw_gradient_norm", gradient_norm)):
                close(value, endpoint_record[key], f"{name} own-chart endpoint {key}")
            for side, factor in (("u", P), ("v", Q)):
                margin = np.linalg.svd(factor[getattr(chart, f"anchors_{side}")], compute_uv=False)[-1]
                close(margin, endpoint_record[f"anchor_min_{side}"], f"{name} own-chart anchor {side}")
                mask = getattr(free, f"penalized_{side}")
                support = np.count_nonzero(state[getattr(chart, f"z_{side}_slice")].reshape(mask.shape)[mask])
                require(support == endpoint_record[f"support_{side}"], f"{name} endpoint support count {side}")
        if selected_iteration is not None:
            row = next(row for row in validation if row["iteration"] == selected_iteration)
            close(metrics["validation_mse"], row["loss"], f"{name} selected validation loss")
            errors["validation_mse"] = max(errors["validation_mse"], abs(metrics["validation_mse"] - row["loss"]))
        count += 1
        return metrics

    for prefix, iteration, selected_iteration in (("selected", selected, selected), ("terminal", result["n_iter"], None)):
        epoch = (len(events) if prefix == "terminal" else
                 next(row["chart_epoch"] for row in validation if row["iteration"] == selected))
        endpoint_record = history["terminal_record"] if prefix == "terminal" else None
        require(prefix != "terminal" or isinstance(endpoint_record, dict), "missing own-chart terminal record")
        metrics = check_state(f"{prefix}_state", prefix, epoch, iteration=iteration, selected_iteration=selected_iteration,
                              saved_factors=True, endpoint_record=endpoint_record)
        for side in ("u", "v"):
            require(result["metadata"][f"{prefix}_anchors"][side] == epochs[epoch][side]["anchors"],
                    f"{prefix} anchors differ from metadata: {side}")
        require(set(metrics) == set(result[f"{prefix}_metrics"]), f"{prefix} metric names mismatch")
        for key, value in metrics.items():
            close(value, result[f"{prefix}_metrics"][key], f"{prefix} reported {key}")
    close(result["metadata"]["validation_mse"], result["selected_metrics"]["validation_mse"], "metadata best MSE")
    if result["success"]:
        close(result["avg_err"], result["selected_metrics"]["coefficient_rmse"], "eligible avg_err")
    else:
        require(result["avg_err"] is None, "failed parent unexpectedly eligible for paper avg_err")
    state_names = {"selected_state", "terminal_state"}
    for key, checkpoint in history["checkpoints"].items():
        t = int(key)
        require(checkpoint["iteration"] == t and 0 <= t <= result["n_iter"], "checkpoint iteration mismatch")
        require(checkpoint["history_length"] == t + 1, "checkpoint history length mismatch")
        nv = checkpoint["validation_history_length"]
        require(0 < nv <= len(validation) and validation[nv-1]["iteration"] <= t, "checkpoint validation extent mismatch")
        require(nv == sum(row["iteration"] <= t for row in validation), "checkpoint omits available validation rows")
        selected_t = validation_winner(validation[:nv])
        require(checkpoint["selected_iteration"] == selected_t, "checkpoint selection transcript mismatch")
        nc = checkpoint["chart_epoch"]
        require(isinstance(nc, int) and 0 <= nc <= len(events) and checkpoint["anchor_switches"] == events[:nc],
                "checkpoint switch transcript is not an exact prefix")
        require(sum(event["iteration"] < t for event in events) <= nc
                <= sum(event["iteration"] <= t for event in events), "checkpoint epoch outside iteration")
        selected_epoch = next(row["chart_epoch"] for row in validation[:nv] if row["iteration"] == selected_t)
        require(checkpoint["selected_chart_epoch"] == selected_epoch, "checkpoint selected chart epoch mismatch")
        endpoint, chosen = f"checkpoint_{t}_state", f"checkpoint_{t}_selected_state"
        require(isinstance(checkpoint["endpoint_record"], dict), "missing own-chart checkpoint endpoint record")
        check_state(endpoint, f"checkpoint_{t}_terminal", nc, iteration=t, endpoint_record=checkpoint["endpoint_record"])
        metrics = check_state(chosen, f"checkpoint_{t}_selected", selected_epoch, iteration=selected_t, selected_iteration=selected_t)
        close(metrics["validation_mse"], checkpoint["best_validation_loss"], "checkpoint best validation MSE")
        state_names.update((endpoint, chosen))
    require({name for name in arrays if name.endswith("_state")} == state_names,
            "unregistered saved state array")
    return dict(report, success=True, accepted_states_checked=count,
                checkpoints_checked=len(history["checkpoints"]), distinct_retained_iterations=len(retained_factors),
                own_charts_checked=count, anchor_switches_checked=len(events), max_errors=errors)


def audit(results_root, cases_root, task_ids=None):
    report = dict(schema_version=2, checked_utc=datetime.now(timezone.utc).isoformat(),
                  results_root=str(results_root), cases_root=str(cases_root), success=False,
                  tasks=[], errors=[], limitations=[
                      "Checks each retained state with its own chart and the reported validation selection transcript; unsaved intermediate states, validation predictions and switch endpoints are not independently reconstructed.",
                      "At a switch iteration, an accepted history record may describe the old chart; only chart-invariant loss and penalty are matched to that record.",
                      "Artifact validity does not override original tuning eligibility, certify convergence, establish complete grid/budget coverage, or certify statistical theorem assumptions.",
                      "Metric/factor comparisons allow atol=rtol=2e-9 for cross-platform arithmetic; hashes and chart-domain predicates are exact."])
    try:
        plan, prep, source, source_count = load_provenance(results_root)
        report.update(source_root_checked=str(source), source_files_verified=source_count,
                      plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"])
        if task_ids is None:
            folders = sorted(folder for folder in (results_root / "tasks").iterdir() if folder.is_dir())
        else:
            require(bool(task_ids) and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0
                                           for t in task_ids), "task IDs must be nonnegative integers")
            require(len(set(task_ids)) == len(task_ids), "task IDs must be distinct")
            task_ids = sorted(task_ids)
            report["requested_task_ids"] = task_ids
            require(all(t < plan["n_tasks"] for t in task_ids), "requested task ID is outside the campaign plan")
            folders = [results_root / "tasks" / f"{t:05d}" for t in task_ids]
            missing = [t for t, folder in zip(task_ids, folders)
                       if not folder.is_dir() or not (folder / "result.json").is_file()]
            require(not missing, f"requested task results are missing: {missing}")
        require(folders, "no downloaded task directories")
        prep_cases = {row["case_id"]: row for row in prep["cases"]}
        needed = {read(folder / "result.json")["case"]["case_id"] for folder in folders}
        cache = {case["case_id"]: load_case(results_root, cases_root, case, prep_cases[case["case_id"]], plan)
                 for case in plan["cases"] if case["case_id"] in needed}
        require(set(cache) == needed, "downloaded result refers to unplanned case")
        for folder in folders:
            try:
                report["tasks"].append(audit_task(folder, plan, cache))
            except Exception as error:
                message = f"task {folder.name}: {type(error).__name__}: {error}"
                report["errors"].append(message)
                report["tasks"].append(dict(task_directory=folder.name, success=False, error=message))
        report["cases_checked"] = len(cache)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["tasks_checked"] = len(report["tasks"])
    report["original_status_counts"] = dict(Counter(row.get("original_fit_status", "audit_failed") for row in report["tasks"]))
    report["eligible_under_original_policy"] = sum(row.get("eligible_under_original_policy", False) for row in report["tasks"])
    report["accepted_states_checked"] = sum(row.get("accepted_states_checked", 0) for row in report["tasks"])
    report["success"] = bool(report["tasks"]) and not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", help="Distinct task IDs to audit; default: all task directories")
    args = parser.parse_args()
    report = audit(args.results_root.resolve(), args.cases_root.resolve(), args.task_ids)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
