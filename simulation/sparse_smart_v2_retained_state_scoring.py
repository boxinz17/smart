"""Inventory and training-BIC scoring of every completed, saved v2 state.

Only the frozen writer's schema-2 own-chart states and schema-3 physical RRR
states are supported. This module does not fit, initialize, regenerate inputs,
or reconstruct unsaved updates. Its candidates inherit validation stopping;
incumbents were retained by validation. BIC itself sees observed training data.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import run_sparse_smart_v2_retrospective_bic as old

require, audit, legacy = old.legacy.require, old.audit, old.legacy
LIMITATION = ("Saved trajectories inherit validation stopping; checkpoint incumbents were retained "
              "using validation. Unsaved accepted updates cannot be reconstructed. Retrospective "
              "BIC over these states is not a training-only fitting/stopping experiment.")
GEOMETRY = audit.GEOMETRY_FIELDS


class RetainedStateError(ValueError):
    """A missing/corrupt expected artifact cannot become a smaller tuning grid."""
    def __init__(self, message, inventory):
        super().__init__(message)
        self.inventory = inventory


def state_key(row):
    """Deterministic within/across-trajectory BIC ties; never use evaluation."""
    return row["bic"]["score"], row["iteration"], row["task_id"], row["state_id"]


def _integer(value, name, *, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _geometry_keys(prefix):
    return [f"{prefix}_{field}" for field in GEOMETRY]


def _expected(result, configuration, direct):
    require(result.get("status") == "complete" and result.get("execution_success") is True
            and result.get("success") is True, "only originally eligible completed trajectories may be scored")
    terminal = _integer(result["n_iter"], "terminal iteration")
    cap = _integer(configuration["iterations"], "iteration budget")
    require(terminal <= cap, "terminal iteration exceeds original budget")
    if direct:
        require(terminal == result["selected_iteration"] == 0, "RRR cannot have iterative checkpoints")
        require(result.get("termination_reason") == "target_rrr_closed_form" and
                result.get("optimization_converged") is True, "RRR completion flags differ")
        return [0], [0], []
    reason = result.get("termination_reason")
    require(reason in ("max_iterations", "stationarity", "validation_stop"),
            "successful trajectory has unsupported stopping reason")
    require((result.get("optimization_converged") is True) == (reason == "stationarity"),
            "trajectory convergence flag differs from stopping reason")
    require(reason != "max_iterations" or terminal == cap, "budget completion did not reach original update cap")
    interval = _integer(configuration["checkpoint_interval"], "checkpoint interval", minimum=1)
    scheduled = [0, *range(interval, cap + 1, interval)]
    reached = sorted({t for t in scheduled if t <= terminal} | {terminal})
    return scheduled, reached, [t for t in scheduled if t > terminal]


def _load(folder, result):
    require(legacy.read(folder / "result.json") == result, "supplied result differs from completed result.json")
    require(result.get("status") == "complete" and result.get("execution_success") is True
            and result.get("success") is True, "only originally eligible completed trajectories may be scored")
    for name in ("states.npz", "history.json.gz"):
        require(name in result.get("files", {}), f"missing authenticated {name}")
        legacy.check_hash(folder / name, result["files"][name])
    arrays = audit.load_npz(folder / "states.npz")
    history = json.loads(gzip.decompress((folder / "history.json.gz").read_bytes()))
    require(isinstance(history, dict) and isinstance(history.get("checkpoints"), dict),
            "missing checkpoint history inventory")
    return arrays, history


def _fingerprint(arrays, names, metadata):
    value = hashlib.sha256()
    value.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    # Role labels, not storage prefixes, allow exact duplicated snapshots to
    # collapse without merging distinct charts, actual iterations or scores.
    for role, name in names:
        array = np.ascontiguousarray(arrays[name])
        value.update(json.dumps((role, array.shape, array.dtype.str)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def _state(arrays, state_array, chart_prefix, case, source, configuration, *, direct):
    if direct:
        coefficient = arrays[state_array]
        require(coefficient.shape == (case["p"], case["q"]), "invalid RRR checkpoint coefficient shape")
        return coefficient, dict(direct_rrr=True)
    chart = audit.saved_chart(arrays, chart_prefix, case)
    state = arrays[state_array]
    require(state.ndim == 1 and np.isfinite(state).all(), "invalid retained chart state")
    domain = {key: configuration["margins"][key] for key in ("d_lower", "d_upper", "gap", "anchor_min")}
    require(chart.domain_reason(state, **domain) is None, "retained state violates its own chart domain")
    P, d, Q = chart.reconstruct(state)
    for side, factor in (("u", P), ("v", Q)):
        audit.close(factor.T @ factor, np.eye(case["rank"]), f"{state_array} factor orthogonality {side}")
    physical = [arrays[f"free_rows_{side}"] for side in ("u", "v")]
    for rows, count, dimension in zip(physical, case["free_directions"], (case["p"], case["q"])):
        require(rows.ndim == 1 and rows.dtype.kind in "iu" and len(rows) == count and
                len(np.unique(rows)) == count and np.all((rows >= 0) & (rows < dimension)),
                "invalid fixed physical free rows")
    free = audit.fixed_free_rows(chart, physical)
    zu, zv = chart.unpack(state)[-2:]
    for side, block, factor, cap in zip(("u", "v"), (zu, zv), (P, Q), case["support_limits"]):
        mask = getattr(free, f"penalized_{side}")
        require(np.count_nonzero(block[mask]) <= cap, "retained state exceeds support cap")
        outside = np.setdiff1d(np.arange(len(factor)), getattr(free, f"rows_{side}"))
        audit.close(block[mask], (factor[outside] * d).ravel(), f"{state_array} physical masked coordinates {side}")
    coefficient = ((source["left"] @ P) * d) @ (source["right"] @ Q).T
    return coefficient, dict(free_directions=case["free_directions"], weighted_u=zu, weighted_v=zv,
                            penalized_u=free.penalized_u, penalized_v=free.penalized_v)


def _deduplicate(rows):
    chosen = {}
    for row in rows:
        key = row["task_id"], row["state_id"]
        if key not in chosen:
            chosen[key] = deepcopy(row)
        else:
            incumbent = chosen[key]
            require(incumbent["bic"] == row["bic"] and incumbent["iteration"] == row["iteration"],
                    "identical state fingerprint has inconsistent BIC metadata")
            for origin in row["origins"]:
                if origin not in incumbent["origins"]:
                    incumbent["origins"].append(deepcopy(origin))
    return sorted(chosen.values(), key=state_key)


def _bind_chart_epochs(arrays, history, result, expected, case):
    """Verify geometry belongs to its declared historical chart epoch.

    Only saved event identities are needed here. This does not replay or claim
    to verify the unsaved coefficient at the chart transition itself.
    """
    initial = audit.saved_chart(arrays, "checkpoint_0_terminal", case)
    epochs = [{side: dict(anchors=getattr(initial, f"anchors_{side}").tolist(),
                        center_sha256=hashlib.sha256(np.asarray(getattr(initial, f"center_{side}"),
                                                    dtype="<f8").tobytes(order="C")).hexdigest())
               for side in ("u", "v")}]
    events = result.get("anchor_switches", [])
    previous = -1
    for event in events:
        t = _integer(event["iteration"], "chart event iteration")
        require(previous <= t <= result["n_iter"], "chart event sequence differs from trajectory")
        fresh = {}
        for side in ("u", "v"):
            detail = event["sides"][side]
            require(detail["old_anchors"] == epochs[-1][side]["anchors"] and
                    detail["old_center_sha256"] == epochs[-1][side]["center_sha256"],
                    "chart event geometry chain differs")
            fresh[side] = dict(anchors=detail["new_anchors"], center_sha256=detail["new_center_sha256"])
        epochs.append(fresh)
        previous = t
    validation = history["validation_history"]
    for row in validation:
        t, epoch = row["iteration"], _integer(row["chart_epoch"], "validation chart epoch")
        require(sum(e["iteration"] < t for e in events) <= epoch <= sum(e["iteration"] <= t for e in events),
                "validation chart epoch outside iteration")
    for t, epoch in enumerate(history["history_chart_epochs"]):
        _integer(epoch, "accepted history chart epoch")
        require(sum(e["iteration"] < t for e in events) <= epoch <= sum(e["iteration"] <= t for e in events),
                "accepted chart epoch outside iteration")
    selected_epoch = next(row["chart_epoch"] for row in validation if row["iteration"] == result["selected_iteration"])
    identities = [("terminal", len(events)), ("selected", selected_epoch)]
    for t in expected:
        cp = history["checkpoints"][str(t)]
        identities.extend([(f"checkpoint_{t}_terminal", cp["chart_epoch"]),
                           (f"checkpoint_{t}_selected", cp["selected_chart_epoch"])])
    for prefix, epoch in identities:
        require(type(epoch) is int and 0 <= epoch < len(epochs), "saved chart epoch has no event identity")
        audit.check_chart_epoch(audit.saved_chart(arrays, prefix, case), epochs[epoch], name=prefix)


def _chart_inventory(arrays, history, result, expected, inventory, case):
    checkpoints = history["checkpoints"]
    require(all(str(int(key)) == key and int(key) >= 0 for key in checkpoints), "noncanonical checkpoint keys")
    registered = sorted(int(key) for key in checkpoints)
    inventory["registered_checkpoint_iterations"] = registered
    available, incumbent_available, expected_arrays = [], [], {"states_schema_version", "free_rows_u", "free_rows_v"}
    expected_arrays.update(GEOMETRY)
    for endpoint in ("selected", "terminal"):
        expected_arrays.update([f"{endpoint}_{field}" for field in ("state", "P", "d", "Q")])
        expected_arrays.update(_geometry_keys(endpoint))
    for t in expected:
        current = [f"checkpoint_{t}_state", *_geometry_keys(f"checkpoint_{t}_terminal")]
        selected = [f"checkpoint_{t}_selected_state", *_geometry_keys(f"checkpoint_{t}_selected")]
        expected_arrays.update(current + selected)
        if t in registered and all(key in arrays for key in current):
            available.append(t)
        if t in registered and all(key in arrays for key in selected):
            incumbent_available.append(t)
    inventory.update(available_checkpoint_iterations=available,
        available_incumbent_checkpoint_iterations=incumbent_available,
        missing_checkpoint_metadata=sorted(set(expected) - set(registered)),
        unexpected_checkpoint_metadata=sorted(set(registered) - set(expected)),
        missing_expected_arrays=sorted(expected_arrays - set(arrays)),
        unexpected_arrays=sorted(set(arrays) - expected_arrays))
    inventory["current_coverage_complete"] = available == expected and all(
        key in arrays for key in ["terminal_state", *_geometry_keys("terminal")])
    require(registered == expected, "missing or unexpected checkpoint metadata relative to frozen writer schedule")
    require(set(arrays) == expected_arrays, "missing or unexpected saved-state arrays relative to frozen writer inventory")
    for field in GEOMETRY:
        require(np.array_equal(arrays[field], arrays[f"terminal_{field}"]), "terminal compatibility chart alias differs")
    validation = history.get("validation_history")
    require(isinstance(validation, list) and bool(validation), "missing validation selection transcript")
    selected = audit.validation_winner(validation)
    require(selected == result["selected_iteration"] and validation[-1]["iteration"] <= result["n_iter"],
            "final selected iteration differs from validation transcript")
    records = history.get("history")
    require(isinstance(records, list) and [r["iteration"] for r in records] == list(range(result["n_iter"] + 1)),
            "accepted history does not cover the reported trajectory")
    epochs = history.get("history_chart_epochs")
    require(isinstance(epochs, list) and len(epochs) == len(records), "missing accepted history chart epochs")
    events = result.get("anchor_switches", [])
    require(isinstance(events, list), "invalid chart transition history")
    for t in expected:
        checkpoint = checkpoints[str(t)]
        require(checkpoint["iteration"] == t and checkpoint["history_length"] == t + 1,
                "checkpoint iteration/history length mismatch")
        nv = checkpoint["validation_history_length"]
        require(type(nv) is int and 0 < nv <= len(validation) and
                nv == sum(row["iteration"] <= t for row in validation), "checkpoint validation extent mismatch")
        actual = _integer(checkpoint["selected_iteration"], "checkpoint incumbent iteration")
        require(actual == audit.validation_winner(validation[:nv]), "checkpoint incumbent iteration differs from transcript")
        epoch = _integer(checkpoint["chart_epoch"], "checkpoint chart epoch")
        require(epoch <= len(events) and checkpoint["anchor_switches"] == events[:epoch],
                "checkpoint chart transition transcript differs")
        require(sum(event["iteration"] < t for event in events) <= epoch <=
                sum(event["iteration"] <= t for event in events), "checkpoint chart epoch outside iteration")
        actual_epoch = next(row["chart_epoch"] for row in validation[:nv] if row["iteration"] == actual)
        require(checkpoint["selected_chart_epoch"] == actual_epoch, "incumbent chart epoch differs from transcript")
    _bind_chart_epochs(arrays, history, result, expected, case)
    inventory["all_retained_coverage_complete"] = True


def _rrr_inventory(arrays, history, result, inventory):
    expected = {"states_schema_version", "checkpoint_0_coefficient"}
    expected.update(f"{endpoint}_{field}" for endpoint in ("selected", "terminal")
                    for field in ("coefficient", "P", "d", "Q"))
    inventory.update(registered_checkpoint_iterations=sorted(int(t) for t in history["checkpoints"]),
        available_checkpoint_iterations=[0] if "checkpoint_0_coefficient" in arrays else [],
        available_incumbent_checkpoint_iterations=[], missing_checkpoint_metadata=[] if "0" in history["checkpoints"] else [0],
        unexpected_checkpoint_metadata=sorted(int(t) for t in history["checkpoints"] if t != "0"),
        missing_expected_arrays=sorted(expected-set(arrays)), unexpected_arrays=sorted(set(arrays)-expected))
    require(set(arrays) == expected and set(history["checkpoints"]) == {"0"}, "missing or unexpected RRR saved-state inventory")
    certificate = result.get("rrr_certificate")
    require(isinstance(certificate, dict) and certificate.get("certified") is True and
            certificate.get("scope") == "numerical_design_range_at_recorded_svd_tolerance" and
            certificate.get("coefficient_convention") == "minimum_norm_lift_of_selected_fitted_response",
            "RRR endpoint lacks original optimality certificate")
    require(result.get("metadata", {}).get("rrr_certificate") == certificate,
            "RRR result/metadata certificate differs")
    checkpoint = history["checkpoints"]["0"]
    require(checkpoint["iteration"] == checkpoint["selected_iteration"] == 0 and
            checkpoint["termination_reason"] == "target_rrr_closed_form" and checkpoint["certificate"] == certificate,
            "RRR checkpoint metadata/certificate differs")
    audit.close(arrays["checkpoint_0_coefficient"], arrays["terminal_coefficient"], "RRR checkpoint coefficient differs")
    inventory.update(current_coverage_complete=True, all_retained_coverage_complete=True,
                     rrr_certificate_verified=True,
                     rrr_certificate_scope="authenticated original certificate and endpoint reconstruction; no refit")


def score_retained_states(folder, result, data, source, case, configuration, *, design_rank=None):
    """Return exact old endpoints, all available state scores, and coverage.

    The caller verifies campaign/task and input provenance. This function also
    verifies the completed result and its bound state/history hashes. Failure
    raises RetainedStateError with a structured ``inventory``; it never quietly
    substitutes an incomplete candidate library. Results contain no arrays.
    """
    folder = Path(folder)
    inventory = dict(schema_version=1, current_coverage_complete=False, all_retained_coverage_complete=False,
        expected_checkpoint_iterations=[], available_checkpoint_iterations=[], unreached_checkpoint_iterations=[],
        unsaved_validation_iterations=[], errors=[], authority="completed hash-bound states.npz and history.json.gz",
        live_checkpoint_policy="ignored when completed authoritative state archive exists", limitation=LIMITATION)
    try:
        direct = result.get("fit_method") == "target_rrr"
        scheduled, expected, unreached = _expected(result, configuration, direct)
        inventory.update(scheduled_checkpoint_iterations=scheduled, expected_checkpoint_iterations=expected,
                         unreached_checkpoint_iterations=unreached)
        arrays, history = _load(folder, result)
        schema = arrays.get("states_schema_version")
        require(schema is not None and schema.shape == () and schema.dtype.kind in "iu" and
                int(schema.item()) == (3 if direct else 2), "unsupported retained-state schema")
        if direct:
            _rrr_inventory(arrays, history, result, inventory)
        else:
            _chart_inventory(arrays, history, result, expected, inventory, case)
        # This exact existing routine is retained as the endpoint control.
        endpoints = old.score_endpoints(folder, result, data, source, case, configuration, design_rank=design_rank)
        definitions = []
        for t in expected:
            cp = history["checkpoints"][str(t)]
            if direct:
                definitions.append(("rrr", 0, 0, "checkpoint_0_coefficient", None, "checkpoint_0_rrr"))
            else:
                definitions.extend([
                    ("checkpoint_current", t, t, f"checkpoint_{t}_state", f"checkpoint_{t}_terminal", f"checkpoint_{t}_current"),
                    ("checkpoint_incumbent", t, cp["selected_iteration"], f"checkpoint_{t}_selected_state", f"checkpoint_{t}_selected", f"checkpoint_{t}_incumbent")])
        definitions.extend((endpoint, result["n_iter"], endpoints[endpoint]["iteration"],
                            f"{endpoint}_coefficient" if direct else f"{endpoint}_state",
                            None if direct else endpoint, endpoint) for endpoint in ("terminal", "selected"))
        rows = []
        result_hash = legacy.sha(folder / "result.json")
        def state_names(state_array, chart_prefix):
            names = [("coefficient" if direct else "state", state_array)]
            if not direct:
                names += [(field, f"{chart_prefix}_{field}") for field in GEOMETRY]
                names += [(f"free_rows_{side}", f"free_rows_{side}") for side in ("u", "v")]
            return names

        def identity_metadata(actual):
            return dict(iteration=actual, rank=case["rank"], free_directions=case["free_directions"],
                support_tolerance=0., fit_method=result.get("fit_method", "sparse_smart_v2"),
                rrr_certificate=result.get("rrr_certificate") if direct else None)

        # The already-audited endpoint scores also seed the within-trajectory
        # cache. Repeated incumbents need no repeated large matrix products.
        cache = {}
        for endpoint, record in endpoints.items():
            field = f"{endpoint}_coefficient" if direct else f"{endpoint}_state"
            names = state_names(field, None if direct else endpoint)
            cache[_fingerprint(arrays, names, identity_metadata(record["iteration"]))] = record["bic"], record["metrics"]
        for kind, saved_at, actual, state_array, chart_prefix, origin_name in definitions:
            names = state_names(state_array, chart_prefix)
            identity = identity_metadata(actual)
            cache_key = _fingerprint(arrays, names, identity)
            if cache_key in cache:
                score, metrics = cache[cache_key]
            else:
                coefficient, args = _state(arrays, state_array, chart_prefix, case, source, configuration, direct=direct)
                if direct and design_rank is not None:
                    args["design_rank"] = design_rank
                score = old.bic_score(data["X"], data["Y"], coefficient, rank=case["rank"], **args).as_dict()
                # Scoring and the deterministic ordering have no evaluation inputs.
                metrics = audit._metrics(coefficient, data)
                cache[cache_key] = score, metrics
            if kind in ("selected", "terminal"):
                require(score == endpoints[kind]["bic"], "new state score differs from exact old endpoint score")
                metrics = endpoints[kind]["metrics"]
            if kind == "checkpoint_incumbent":
                audit.close(metrics["validation_mse"], history["checkpoints"][str(saved_at)]["best_validation_loss"],
                            "incumbent validation metric differs from saved transcript")
            fingerprint = _fingerprint(arrays, names, dict(identity, bic=score))
            rows.append(dict(task_id=result["task"]["task_id"], task=result["task"], endpoint=origin_name,
                iteration=actual, fit_method=result.get("fit_method", "sparse_smart_v2"),
                termination_reason=result.get("termination_reason"),
                optimization_converged=result.get("optimization_converged") is True,
                bic=score, metrics=metrics, states_sha256=result["files"]["states.npz"], result_sha256=result_hash,
                history_sha256=result["files"]["history.json.gz"],
                state_id=f"task_{result['task']['task_id']}_state_{fingerprint}",
                state_pointer=dict(state_array=state_array, chart_prefix=chart_prefix, archive="states.npz", history="history.json.gz"),
                origins=[dict(kind=kind, saved_at_iteration=saved_at, actual_iteration=actual)]))
        current = _deduplicate([row for row in rows if row["origins"][0]["kind"] in ("checkpoint_current", "terminal", "rrr")])
        all_states = _deduplicate(rows)
        represented = {row["iteration"] for row in all_states}
        validation_iterations = [row["iteration"] for row in history["validation_history"]]
        inventory.update(raw_saved_state_count=len(rows), distinct_current_state_count=len(current),
            distinct_all_retained_state_count=len(all_states), validation_evaluation_iterations=validation_iterations,
            unsaved_validation_iterations=sorted(set(validation_iterations)-represented),
            saved_actual_iterations=sorted(represented),
            recorded_hashes=dict(states_sha256=result["files"]["states.npz"],
                                 history_sha256=result["files"]["history.json.gz"], result_sha256=result_hash))
        return dict(endpoints=endpoints, current_states=current, all_states=all_states, inventory=inventory)
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, EOFError, zipfile.BadZipFile) as error:
        inventory["all_retained_coverage_complete"] = False
        inventory["errors"].append(str(error))
        raise RetainedStateError(str(error), inventory) from error
