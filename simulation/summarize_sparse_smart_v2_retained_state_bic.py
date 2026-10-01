#!/usr/bin/env python3
"""Compare five selection rules on existing validation-stopped retained states.

Only compact scoring records are read: no models are fitted and no coefficient
archives are copied. Every arm inherits validation stopping. The all-retained
arm additionally searches validation-filtered incumbents; it is not training-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

import summarize_sparse_smart_v2_retrospective_bic as historical

METHOD = "SparseSMARTv2RetainedStateBIC"
ARMS = ("original_validation", "bic_validation_selected", "bic_validation_terminal",
        "bic_saved_current", "bic_all_retained")
LABELS = dict(zip(ARMS, ("Original validation", "BIC: validation-selected", "BIC: terminal",
                        "BIC: saved current", "BIC: all retained")))
METRICS = ("coefficient_rmse", "training_prediction_mse", "validation_mse")
PAIRS = tuple((later, earlier) for earlier, later in combinations(ARMS, 2))
INVENTORY_COUNTS = (
    "expected_checkpoint_count", "available_checkpoint_count", "available_incumbent_checkpoint_count",
    "unreached_checkpoint_count", "validation_evaluation_count", "unsaved_validation_evaluation_count",
    "raw_saved_state_count", "distinct_current_state_count", "distinct_all_retained_state_count",
)
LIMITATION = (
    "All five arms reuse trajectories stopped using validation observations. "
    "BIC on validation-selected states and BIC on all retained states additionally "
    "search validation-filtered incumbents. Neither is a training-only procedure. "
    "The all-retained arm searches only saved artifacts, not every accepted or "
    "unsaved iteration. No models are refitted. Validation MSE is reported only; "
    "it is not independent test error. Training-design signal MSE is "
    "||X(C_hat-C_true)||_F^2/(n*q), not observed residual RSS."
)
require, read, sha, digest = historical.require, historical.read, historical.sha, historical.digest
check_hash, file_inside = historical.check_hash, historical.file_inside
write_json, write_csv, finite, xvalue = historical.write_json, historical.write_csv, historical.finite, historical.xvalue


def stats(values):
    """Student-t intervals across independent seed blocks, never pooled states."""
    values = list(values)
    n = len(values)
    mean = statistics.mean(values) if n else None
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    if se is None:
        low = high = None
    else:
        from scipy.stats import t
        width = float(t.ppf(.975, n - 1)) * se
        low, high = mean - width, mean + width
    return dict(n=n, mean=mean, se=se, lower95=low, upper95=high)


def state_changed(winner, reference):
    if winner is None or reference is None:
        return None, None
    candidate = winner["task_id"] != reference["task_id"]
    # Iteration/task identity survives aliases such as terminal and saved current.
    return candidate, candidate or winner["iteration"] != reference["iteration"]


def winner_features(record, arm):
    if arm in record.get("winner_features", {}):
        return record["winner_features"][arm]
    winner = record["arms"][arm]
    if winner is None:
        return {}
    candidate = next((c for c in record.get("candidates", []) if c["selected"]["task_id"] == winner["task_id"]), None)
    states = candidate["states"] if candidate is not None else []
    state = winner if "origins" in winner else next((s for s in states if s["iteration"] == winner["iteration"] and
             any(o["kind"] in (winner["endpoint"], "rrr") for o in s["origins"])), None)
    origins = state["origins"] if state else []
    checkpoints = [o["actual_iteration"] for s in states for o in s["origins"] if o["kind"] == "checkpoint_current"]
    last = max(checkpoints) if checkpoints else None
    return dict(state_id=state["state_id"] if state else None,
                origin_kinds=sorted({o["kind"] for o in origins}),
                saved_at_iterations=sorted({o["saved_at_iteration"] for o in origins}),
                state_pointer=state["state_pointer"] if state else None,
                last_available_current_checkpoint_iteration=last,
                last_available_current_checkpoint=bool(winner["fit_method"] != "target_rrr" and last is not None and
                    any(o["kind"] == "checkpoint_current" and o["actual_iteration"] == last for o in origins)))


def build_tables(records):
    rows, curves, differences, blocks = [], [], [], []
    grouped = defaultdict(list)
    for record in records:
        case = record["case"]
        grouped[tuple(case[k] for k in ("model_id", "experiment_id", "setting_index"))].append(record)
        for arm in ARMS:
            winner = record["arms"][arm]
            changed, changed_state = state_changed(winner, record["arms"][ARMS[0]])
            row = dict(case_id=case["case_id"], model_id=case["model_id"], experiment_id=case["experiment_id"],
                       setting_index=case["setting_index"], seed_id=case["seed_id"], x=xvalue(case), arm=arm,
                       available=winner is not None, scoring_status=record["classification"],
                       complete_execution_coverage=record["complete_execution_coverage"],
                       complete_tuning_coverage=record["complete_tuning_coverage"],
                       candidate_changed_from_validation=changed, state_changed_from_validation=changed_state)
            if winner is not None:
                row.update(winner["metrics"])
                row.update(task_id=winner["task_id"], endpoint=winner["endpoint"], iteration=winner["iteration"],
                           state_id=winner.get("state_id"), origin=winner.get("origin", winner["endpoint"]),
                           fit_method=winner["fit_method"], bic=winner["bic"]["score"], rss=winner["bic"]["rss"],
                           model_dimension=winner["bic"]["model_dimension"], rank=case["rank"],
                           free_u=case["free_directions"][0], free_v=case["free_directions"][1],
                           termination_reason=winner.get("termination_reason"),
                           optimization_converged=winner.get("optimization_converged"),
                           last_available_current_checkpoint=winner.get("last_available_current_checkpoint", False))
                row.update({k: winner["task"].get(k) for k in ("init_penalty", "penalty_u", "penalty_v")})
                row.update(winner_features(record, arm))
            rows.append(row)
    for key, members in sorted(grouped.items()):
        common = dict(model_id=key[0], experiment_id=key[1], setting_index=key[2], x=xvalue(members[0]["case"]),
                      requested_seeds=len(members), requested_seed_ids=sorted(r["case"]["seed_id"] for r in members),
                      complete_execution_coverage=all(r["complete_execution_coverage"] for r in members),
                      complete_tuning_coverage=all(r["complete_tuning_coverage"] for r in members))
        for arm in ARMS:
            winners = [r["arms"][arm] for r in members if r["arms"][arm] is not None]
            curve = dict(common, arm=arm, available_seeds=len(winners),
                         missing_seed_ids=sorted(r["case"]["seed_id"] for r in members if r["arms"][arm] is None),
                         rrr_selected=sum(w["fit_method"] == "target_rrr" for w in winners),
                         initializer_selected=sum(w["fit_method"] != "target_rrr" and w["iteration"] == 0 for w in winners))
            for metric in (*METRICS, "iteration", "model_dimension"):
                vals = [w["iteration"] if metric == "iteration" else w["bic"][metric] if metric == "model_dimension"
                        else w["metrics"][metric] for w in winners]
                curve.update({metric + "_" + k: v for k, v in stats(vals).items() if k != "n"})
            curves.append(curve)
        for arm, reference in PAIRS:
            pairs = [r for r in members if r["arms"][arm] is not None and r["arms"][reference] is not None]
            for metric in METRICS:
                d = stats(r["arms"][arm]["metrics"][metric] - r["arms"][reference]["metrics"][metric] for r in pairs)
                differences.append(dict(common, arm=arm, reference=reference, metric=metric,
                                        paired_seed_ids=sorted(r["case"]["seed_id"] for r in pairs),
                                        **d, interpretation="negative favors arm; paired within case"))
    # A repeated setting is not an independent replication. Require a complete
    # within-seed block over the scope's planned settings before averaging it.
    scopes = defaultdict(list)
    for record in records:
        c = record["case"]
        scopes[(c["model_id"], c["experiment_id"])].append(record)
        scopes[(c["model_id"], "all_experiments")].append(record)
    for (model, experiment), members in sorted(scopes.items(), key=lambda item: str(item[0])):
        expected = {(r["case"]["experiment_id"], r["case"]["setting_index"]) for r in members}
        seeds = sorted({r["case"]["seed_id"] for r in members})
        by_seed = {seed: [r for r in members if r["case"]["seed_id"] == seed] for seed in seeds}
        for arm, reference in PAIRS:
            for metric in METRICS[:2]:
                deltas = {}
                for seed, group in by_seed.items():
                    coords = {(r["case"]["experiment_id"], r["case"]["setting_index"]) for r in group}
                    if coords != expected or len(group) != len(expected) or any(r["arms"][arm] is None or r["arms"][reference] is None for r in group):
                        continue
                    deltas[seed] = statistics.mean(r["arms"][arm]["metrics"][metric] - r["arms"][reference]["metrics"][metric] for r in group)
                for cohort, test in (("all", lambda s: True), ("design_seeds_0_9", lambda s: s < 10),
                                     ("seeds_10_99", lambda s: 10 <= s <= 99)):
                    selected = [s for s in seeds if test(s)]
                    included = [s for s in selected if s in deltas]
                    blocks.append(dict(model_id=model, experiment_id=experiment, arm=arm, reference=reference,
                                       metric=metric, cohort=cohort, requested_seed_blocks=len(selected),
                                       settings_per_seed=len(expected), seed_ids=included,
                                       omitted_seed_ids=[s for s in selected if s not in deltas],
                                       **stats(deltas[s] for s in included),
                                       interpretation="descriptive paired seed means; repeated settings share one seed block"))
    return rows, curves, differences, blocks


def load_plan(root):
    plan = read(root / "plan.json")
    require(plan.get("schema_version") == 1 and plan.get("method") == METHOD, "unsupported retained-state plan")
    require(plan.get("selection_arms") == list(ARMS), "five-arm plan declaration differs")
    require(plan.get("root") == str(root), "plan root identity mismatch")
    require(digest({k: v for k, v in plan.items() if k != "plan_fingerprint"}) == plan.get("plan_fingerprint"),
            "plan fingerprint mismatch")
    cases = plan["cases"]
    require(isinstance(cases, list) and len(cases) == plan["n_cases"] > 0, "invalid planned case count")
    identities = set()
    for entry in cases:
        case, cid = entry["case"], entry["case_id"]
        coords = tuple(case[k] for k in ("model_id", "experiment_id", "setting_index", "seed_id"))
        require(all(type(c) is int and c >= 0 for c in coords) and coords[0] < 3 and coords[1] < 4,
                "invalid case coordinates")
        require(cid == case["case_id"] == "m%d_e%d_k%d_s%d" % coords and coords not in identities,
                "invalid or duplicate case identity")
        require(Path(entry["source_root"]).is_absolute(), "reference source root must be absolute")
        identities.add(coords)
    check_hash(root / "source-manifest.json", plan["source_manifest_sha256"])
    manifest = read(root / "source-manifest.json")
    require(bool(manifest.get("files")), "empty source manifest")
    for name, expected in manifest["files"].items():
        check_hash(file_inside(root / "source", name), expected)
    prior = Path(plan["previous_retrospective_root"])
    require(prior.is_absolute() and prior != root, "invalid previous retrospective root")
    check_hash(prior / "plan.json", plan["previous_plan_sha256"])
    check_hash(prior / "source-manifest.json", plan["previous_source_manifest_sha256"])
    preparation = read(root / "preparation.json")
    require(preparation.get("status") == "complete" and preparation.get("success") is True, "preparation incomplete")
    for key, expected in (("schema_version", 1), ("method", METHOD), ("plan_sha256", sha(root / "plan.json")),
                          ("plan_fingerprint", plan["plan_fingerprint"]), ("n_cases", plan["n_cases"]),
                          ("source_manifest_sha256", plan["source_manifest_sha256"])):
        require(preparation.get(key) == expected, f"preparation identity mismatch: {key}")
    prepared = {row["case_id"]: row for row in preparation["cases"]}
    require(len(prepared) == len(preparation["cases"]) == len(cases) and
            set(prepared) == {e["case_id"] for e in cases}, "prepared case coverage mismatch")
    return plan, prepared


def validate_state(row):
    # The numerical runner validates chart and physical metrics. The compact
    # summary binds its score to a literal task/archive/state pointer.
    endpoint = row.get("endpoint")
    clone = dict(row, endpoint="selected")
    historical.validate_winner(clone, historical.ARMS[0])
    require(isinstance(endpoint, str) and endpoint, "missing state endpoint")
    require(isinstance(row.get("state_id"), str) and row["state_id"], "missing stable state ID")
    require(isinstance(row.get("history_sha256"), str) and re.fullmatch("[0-9a-f]{64}", row["history_sha256"]),
            "missing saved-history provenance")
    pointer = row.get("state_pointer")
    require(isinstance(pointer, dict) and set(pointer) >= {"state_array", "chart_prefix", "archive", "history"},
            "missing literal state pointer")
    require(pointer["archive"] == "states.npz" and pointer["history"] == "history.json.gz", "unexpected state artifact")
    require(isinstance(pointer["state_array"], str) and pointer["state_array"], "missing array pointer")
    require(pointer["chart_prefix"] is None or isinstance(pointer["chart_prefix"], str), "invalid chart pointer")
    origins = row.get("origins")
    require(isinstance(origins, list) and origins, "missing state origins")
    for origin in origins:
        require(origin.get("kind") in ("checkpoint_current", "checkpoint_incumbent", "terminal", "selected", "rrr"),
                "unknown state origin")
        require(type(origin.get("saved_at_iteration")) is int and origin["saved_at_iteration"] >= 0 and
                type(origin.get("actual_iteration")) is int and origin["actual_iteration"] == row["iteration"] and
                origin["saved_at_iteration"] >= origin["actual_iteration"], "invalid actual/saved state iteration")


def new_key(row):
    return row["bic"]["score"], row["iteration"], row["task_id"], row["state_id"]


def validate_candidates(result):
    candidates, inventory = result["candidates"], result["inventory"]
    for key in INVENTORY_COUNTS:
        require(type(inventory.get(key)) is int and inventory[key] >= 0, f"invalid inventory count: {key}")
    require(inventory["available_checkpoint_count"] <= inventory["expected_checkpoint_count"],
            "available checkpoints exceed expected reached checkpoints")
    require(inventory["available_incumbent_checkpoint_count"] <= inventory["available_checkpoint_count"],
            "incumbent checkpoints exceed available checkpoints")
    require(inventory["unsaved_validation_evaluation_count"] <= inventory["validation_evaluation_count"],
            "unsaved validation iterations exceed evaluation inventory")
    require(inventory["distinct_current_state_count"] <= inventory["distinct_all_retained_state_count"] <= inventory["raw_saved_state_count"],
            "distinct state counts exceed retained inventory")
    require(isinstance(candidates, list) and len(candidates) == result["eligible_tasks"], "candidate task coverage differs")
    ids = [c["selected"]["task_id"] for c in candidates]
    require(len(ids) == len(set(ids)) and sorted(ids) == inventory["eligible_task_ids"], "eligible task inventory differs")
    all_states, current, origins = [], [], Counter()
    for candidate in candidates:
        historical.validate_winner(candidate["selected"], historical.ARMS[1])
        historical.validate_winner(candidate["terminal"], historical.ARMS[2])
        tid = candidate["selected"]["task_id"]
        require(candidate["terminal"]["task_id"] == tid, "endpoint task identity differs")
        require(isinstance(candidate.get("inventory"), dict), "missing task inventory")
        states = candidate["states"]
        require(isinstance(states, list) and states, "empty eligible task state inventory")
        state_ids = [s["state_id"] for s in states]
        require(len(state_ids) == len(set(state_ids)), "duplicate stable state ID within task")
        selected_ids = candidate["current_state_ids"]
        require(isinstance(selected_ids, list) and len(set(selected_ids)) == len(selected_ids) and
                bool(selected_ids) and set(selected_ids) <= set(state_ids), "invalid current-state inventory")
        for row in states:
            validate_state(row)
            require(row["task_id"] == tid and row["task"] == candidate["selected"]["task"], "retained state tuning identity differs")
            require(all(row[k] == candidate["selected"][k] for k in ("result_sha256", "states_sha256", "fit_method")),
                    "state provenance differs within task")
            origins.update({o["kind"] for o in row["origins"]})
        actual_current = {s["state_id"] for s in states if any(o["kind"] in ("checkpoint_current", "terminal", "rrr") for o in s["origins"])}
        require(set(selected_ids) == actual_current, "current arm contains incumbents or omits saved current states")
        for endpoint in ("selected", "terminal"):
            old = candidate[endpoint]
            represented = [s for s in states if s["iteration"] == old["iteration"] and any(o["kind"] == endpoint for o in s["origins"])]
            require(represented and any(s["bic"] == old["bic"] and s["metrics"] == old["metrics"] for s in represented),
                    f"retained pool does not represent exact historical {endpoint} endpoint")
        if result["success"]:
            require(candidate["inventory"].get("current_coverage_complete") is True and
                    candidate["inventory"].get("all_retained_coverage_complete") is True,
                    "successful task has incomplete retained-state inventory")
        all_states.extend(states)
        current.extend(s for s in states if s["state_id"] in selected_ids)
    require(inventory["states_total"] == len(all_states), "retained state count differs")
    require(inventory["distinct_all_retained_state_count"] == len(all_states) and
            inventory["distinct_current_state_count"] == len(current), "distinct state inventory count differs")
    require(inventory["states_by_origin"] == dict(origins), "state-origin inventory differs")
    require(inventory["candidate_states_fingerprint"] == digest([(c["selected"]["task_id"], c["states"]) for c in candidates]),
            "candidate-state score fingerprint differs")
    for key in ("current_coverage_complete", "all_retained_coverage_complete"):
        require(type(inventory.get(key)) is bool, f"missing inventory coverage: {key}")
    if not result["success"]:
        require(not any(result["arms"].values()), "incomplete scoring cannot expose a winner")
        return []
    require(inventory["current_coverage_complete"] and inventory["all_retained_coverage_complete"], "successful case has incomplete state coverage")
    old_selected = [c["selected"] for c in candidates]
    old_terminal = [c["terminal"] for c in candidates]
    require(result["arms"][ARMS[0]] == min(old_selected, key=lambda s: (s["metrics"]["validation_mse"], s["task_id"])),
            "historical validation tie rule differs")
    require(result["arms"][ARMS[1]] == min(old_selected, key=lambda s: (s["bic"]["score"], s["task_id"])),
            "historical selected-state BIC tie rule differs")
    require(result["arms"][ARMS[2]] == min(old_terminal, key=lambda s: (s["bic"]["score"], s["task_id"])),
            "historical terminal BIC tie rule differs")
    require(result["arms"][ARMS[3]] == min(current, key=new_key), "saved-current BIC tie rule differs")
    require(result["arms"][ARMS[4]] == min(all_states, key=new_key), "all-retained BIC tie rule differs")
    checks = []
    for superset, subset in ((ARMS[3], ARMS[2]), (ARMS[4], ARMS[1]), (ARMS[4], ARMS[2]), (ARMS[4], ARMS[3])):
        a, b = result["arms"][superset]["bic"]["score"], result["arms"][subset]["bic"]["score"]
        require(a <= b, f"minimum BIC increases for superset {superset} versus {subset}")
        checks.append(dict(superset=superset, subset=subset, difference=a-b, passed=True,
                           eligible_task_ids=sorted(ids)))
    return checks


def load_case(root, plan, entry, prepared):
    cid = entry["case_id"]
    path, status_path = root / "cases" / f"{cid}.json", root / "cases" / f"{cid}.status.json"
    check_hash(root / "inputs" / f"{cid}.json", prepared["input_sha256"])
    frozen_input = read(root / "inputs" / f"{cid}.json")
    result, status = read(path), read(status_path)
    check_hash(path, status.get("result_sha256"))
    for key, expected in (("schema_version", 1), ("method", METHOD), ("case_id", cid),
                          ("plan_fingerprint", plan["plan_fingerprint"]), ("source_manifest_sha256", plan["source_manifest_sha256"]),
                          ("input_sha256", prepared["input_sha256"])):
        require(result.get(key) == status.get(key) == expected, f"case result/status identity mismatch: {key}")
    require(result.get("case") == entry["case"] and result.get("reference_subroot") == entry["source_root"], "case/reference identity mismatch")
    require(status.get("status") == "finished" and type(result.get("success")) is bool and
            status.get("success") == result["success"], "case status mismatch")
    for key in ("complete_execution_coverage", "complete_tuning_coverage"):
        require(type(result.get(key)) is bool, f"missing coverage: {key}")
    require(isinstance(result.get("errors"), list), "missing case errors")
    for key in ("elapsed_seconds", "process_cpu_seconds"):
        require(finite(result.get(key)) and result[key] >= 0, f"invalid worker timing: {key}")
    counts = result.get("counts")
    require(isinstance(counts, dict) and all(type(v) is int and v >= 0 for v in counts.values()), "invalid task counts")
    require(type(result.get("planned_tasks")) is int and result["planned_tasks"] > 0 and
            sum(counts.values()) == result["planned_tasks"], "planned task outcome coverage differs")
    require(type(result.get("eligible_tasks")) is int and result["eligible_tasks"] == counts.get("eligible", 0), "eligible count differs")
    require(isinstance(result.get("candidate_inputs_fingerprint"), str) and
            re.fullmatch("[0-9a-f]{64}", result["candidate_inputs_fingerprint"]) and
            result["candidate_inputs_fingerprint"] == status.get("candidate_inputs_fingerprint"), "candidate input fingerprint differs")
    require(set(result["arms"]) == set(ARMS), "five-arm coverage differs")
    if result["success"]:
        require(result["complete_execution_coverage"] and not result["errors"] and all(result["arms"].values()),
                "successful case lacks execution coverage or winners")
    checks = validate_candidates(result)
    prior_root = Path(plan["previous_retrospective_root"])
    prior_path, prior_status_path = prior_root / "cases" / f"{cid}.json", prior_root / "cases" / f"{cid}.status.json"
    comparison = result["historical_comparison"]
    check_hash(prior_path, frozen_input["previous_case_sha256"])
    check_hash(prior_status_path, frozen_input["previous_status_sha256"])
    prior, prior_status = read(prior_path), read(prior_status_path)
    check_hash(prior_path, prior_status["result_sha256"])
    require(prior.get("case") == entry["case"] and prior.get("method") == historical.METHOD and
            prior.get("success") is True, "previous retrospective case identity/completion differs")
    if result["success"]:
        require(comparison.get("passed") is True, "historical endpoint verification failed")
        require(all(comparison[k] == frozen_input[k] for k in ("previous_case_sha256", "previous_status_sha256")),
                "historical comparison hashes differ from frozen case input")
        require(prior["candidate_inputs_fingerprint"] == result["candidate_inputs_fingerprint"],
                "source candidate artifacts differ from historical comparison")
        prior_candidates = {c["selected"]["task_id"]: c for c in prior["candidates"]}
        require(len(prior_candidates) == len(prior["candidates"]) and
                sorted(prior_candidates) == result["inventory"]["eligible_task_ids"],
                "historical and retained-state eligible task coverage differs")
        for candidate in result["candidates"]:
            old = prior_candidates[candidate["selected"]["task_id"]]
            require(all(candidate[endpoint] == old[endpoint] for endpoint in ("selected", "terminal")),
                    "historical candidate endpoint scores changed")
        for arm in ARMS[:3]:
            require(result["arms"][arm] == prior["arms"][arm], f"historical endpoint arm changed: {arm}")
    hashes = dict(case_id=cid, result_sha256=sha(path), status_sha256=sha(status_path), input_sha256=prepared["input_sha256"],
                  candidate_inputs_fingerprint=result["candidate_inputs_fingerprint"],
                  candidate_states_fingerprint=result["inventory"]["candidate_states_fingerprint"],
                  previous_case_sha256=sha(prior_path), previous_status_sha256=sha(prior_status_path))
    return result, hashes, checks


def plot_curves(output, curves):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors = dict(zip(ARMS, ("#64748B", "#009E73", "#D55E00", "#0072B2", "#CC79A7")))
    markers = dict(zip(ARMS, ("o", "s", "^", "D", "v")))
    xlabels = ("Training observations", "Fitted rank", "Unpenalized/source directions", "Source-noise standard deviation")
    outputs = []
    temporary_book = output / f"comparison.{os.getpid()}.tmp.pdf"
    with PdfPages(temporary_book) as book:
        for metric, ylabel in (("coefficient_rmse", "Coefficient RMSE"), ("training_prediction_mse", "Training-design signal MSE")):
            for model in sorted({r["model_id"] for r in curves}):
                fig, axes = plt.subplots(2, 2, figsize=(13, 9.4))
                fig.subplots_adjust(top=.79, bottom=.17, hspace=.40, wspace=.30)
                fig.suptitle(f"Retained-state selection — Model {('I', 'II', 'III')[model]}", fontsize=18, y=.975)
                handles = []
                for experiment, ax in enumerate(axes.flat):
                    subset = [r for r in curves if r["model_id"] == model and r["experiment_id"] == experiment]
                    ticks = sorted({r["x"] for r in subset})
                    for arm in ARMS:
                        selected = sorted((r for r in subset if r["arm"] == arm and r[metric + "_mean"] is not None), key=lambda r: r["x"])
                        xs = [ticks.index(r["x"]) if experiment == 3 else r["x"] for r in selected]
                        ys = [r[metric + "_mean"] for r in selected]
                        err = [r[metric + "_upper95"] - r[metric + "_mean"] if r[metric + "_upper95"] is not None else 0. for r in selected]
                        handle = ax.errorbar(xs, ys, yerr=err, marker=markers[arm], color=colors[arm], markersize=4,
                                             lw=1.5, capsize=2, linestyle="--" if arm == ARMS[0] else "-", label=LABELS[arm])
                        if experiment == 0:
                            handles.append(handle)
                    ns = [r["available_seeds"] for r in subset]
                    label = f"n={min(ns)}" if ns and min(ns) == max(ns) else f"n={min(ns)}–{max(ns)}" if ns else "no data"
                    ax.set_title(f"Experiment {experiment + 1} · {label} seeds", loc="left", fontsize=12)
                    ax.set_xlabel(xlabels[experiment]); ax.set_ylabel(ylabel)
                    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.20)
                    if experiment == 3:
                        ax.set_xticks(range(len(ticks)), [f"{x:g}" for x in ticks])
                    else:
                        ax.set_xticks(ticks)
                    positive = [r[metric + "_mean"] for r in subset if r[metric + "_mean"] is not None]
                    if experiment == 1 and positive and min(positive) > 0:
                        ax.set_yscale("log")
                fig.legend(handles=handles, labels=[LABELS[a] for a in ARMS], loc="upper center",
                           bbox_to_anchor=(.5, .93), ncol=2, fontsize=10, frameon=False,
                           columnspacing=2.3, handlelength=2.6)
                fig.text(.055, .077, "Means and 95% t intervals over seeds; lower is better. Incomplete cases never supply arm winners.\n"
                         "All arms inherit validation stopping; the all-retained arm also searches validation-filtered incumbents.\n"
                         "Only saved states are searched. Source-noise ticks are equally spaced; fitted-rank panels use a log y-axis.", fontsize=9)
                for suffix in ("png", "pdf"):
                    name = f"model_{model + 1}_{metric}.{suffix}"
                    temporary = output / f"{name}.{os.getpid()}.tmp"
                    fig.savefig(temporary, format=suffix, dpi=180, bbox_inches="tight")
                    temporary.replace(output / name); outputs.append(name)
                book.savefig(fig, bbox_inches="tight"); plt.close(fig)
    temporary_book.replace(output / "comparison.pdf")
    return outputs + ["comparison.pdf"]


def selection_counts(records, rows):
    counts = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm and r["available"]]
        counts[arm] = dict(available_cases=len(selected),
            rrr_selected=sum(r["fit_method"] == "target_rrr" for r in selected),
            initializer_selected=sum(r["fit_method"] != "target_rrr" and r["iteration"] == 0 for r in selected),
            refinement_selected=sum(r["fit_method"] != "target_rrr" and r["iteration"] > 0 for r in selected),
            candidate_changed_from_validation=sum(bool(r["candidate_changed_from_validation"]) for r in selected),
            state_changed_from_validation=sum(bool(r["state_changed_from_validation"]) for r in selected),
            last_available_current_checkpoint_selected=sum(r["last_available_current_checkpoint"] for r in selected),
            selected_origin_presence=dict(Counter(kind for r in selected for kind in r["origin_kinds"])),
            iteration_counts=dict(Counter(str(r["iteration"]) for r in selected)),
            tuning_counts={key: dict(Counter(str(r.get(key)) for r in selected if r["fit_method"] != "target_rrr"))
                           for key in ("init_penalty", "penalty_u", "penalty_v")},
            model_dimension=stats(r["model_dimension"] for r in selected))
    return counts


def write_report(output, report):
    lines = ["# Five-arm retained-state BIC comparison", "", LIMITATION, "",
             f"Completed cases: {report['completed_cases']}/{report.get('planned_cases', 0)}; "
             f"settings: {report.get('planned_settings', 0)}; issues: {len(report['errors'])}.", "",
             "| Selection | Available | Initializer | RRR | Refinement | Last saved current checkpoint |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, count in report["selection_counts"].items():
        lines.append(f"| {LABELS[arm]} | {count['available_cases']} | {count['initializer_selected']} | "
                     f"{count['rrr_selected']} | {count['refinement_selected']} | {count['last_available_current_checkpoint_selected']} |")
    inv = report["inventory_totals"]
    lines += ["", f"Retained-state inventory: {inv['available_checkpoint_count']} available current checkpoints out of "
              f"{inv['expected_checkpoint_count']} expected reached checkpoints, with {inv['available_incumbent_checkpoint_count']} "
              f"available incumbent checkpoints. There are {inv['unreached_checkpoint_count']} scheduled checkpoints beyond "
              f"trajectory termination and {inv['unsaved_validation_evaluation_count']} unsaved validation-evaluation iterations "
              f"out of {inv['validation_evaluation_count']} evaluations. Unreached checkpoints and unsaved evaluations are "
              "not corruption; missing expected retained states are audit failures.", "",
              f"Saved representations: {inv['raw_saved_state_count']} before alias removal, "
              f"{inv['distinct_all_retained_state_count']} distinct retained states, "
              f"{inv['distinct_current_state_count']} distinct current states.", "",
              f"Measured worker process CPU: {report['worker_process_cpu_seconds']:.3f} seconds; "
              f"summed worker elapsed time: {report['worker_elapsed_seconds_sum']:.3f} seconds. "
              "These are measured process/task totals, not Slurm reserved CPU time or campaign wall-clock duration."]
    lines += ["", "The three historical arms are checked exactly against the previous retrospective case outputs. "
              "Their original tie rules remain unchanged. New BIC arms break ties by actual iteration, frozen task ID, "
              "and stable state ID. Minimum-BIC monotonicity is checked only on the same eligible task coverage.", "",
              "Missing, corrupt or incomplete case scoring produces no winner in any arm. Initializer exclusions are "
              "reported separately from execution problems and do not alone invalidate a completed case.", "",
              "Per-setting intervals use paired seed values. Cross-setting comparisons first average within each seed "
              "and require its complete planned setting block, then calculate 95% t intervals across seeds. "
              "Repeated settings are never treated as independent observations. The all-seed and seeds 10–99 analyses "
              "are descriptive; excluding design seeds does not make the retained trajectories validation-independent.", "",
              "BIC uses the existing support/model-dimension approximation; this is not a proved adaptive degrees-of-freedom formula. "
              "Origin-presence counts may overlap when aliases refer to the same state. A last-checkpoint selection means a "
              "winner has a checkpoint-current origin at that task's latest saved current checkpoint; a terminal alias alone is not counted.", "",
              "Files: `per-seed-results.csv`, `performance-curves.csv`, `paired-differences.csv`, `paired-seed-blocks.csv`, "
              "`coverage.csv`, `state-inventory.csv`, `case-runtime.csv`, `minimum-bic-checks.csv`, `summary.json`, and model figures. "
              "Only compact case records, scores, pointers and provenance hashes are copied into the summary."]
    if report["errors"]:
        lines += ["", "## Issues", ""] + ["- " + json.dumps(e, sort_keys=True) for e in report["errors"][:30]]
    path = output / "REPORT.md"
    temporary = output / f"REPORT.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n"); temporary.replace(path)
    return path.name


def summarize(root, output_root=None, *, make_plots=True):
    started, cpu_started = time.monotonic(), time.process_time()
    root = Path(root).resolve()
    output = Path(output_root).resolve() if output_root else root / "summary"
    require(output != root and output.is_relative_to(root), "summary output must be a subdirectory of its run")
    require(output.relative_to(root).parts[0] not in ("source", "cases", "inputs"), "summary cannot overwrite source/cases/inputs")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, method=METHOD, result_root=str(root), output_root=str(output),
                  checked_utc=datetime.now(timezone.utc).isoformat(), success=False, errors=[], case_results=[],
                  input_hashes=[], completed_cases=0, limitation=LIMITATION, no_refitting=True,
                  validation_used_for_original_trajectories=True, all_retained_includes_validation_filtered_incumbents=True,
                  truth_used_for_selection=False, complete_execution_coverage=False, complete_tuning_coverage=False,
                  historical_endpoint_comparisons=0, minimum_bic_checks=[])
    totals, case_counts, inventories = Counter(), Counter(), []
    report["started_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        plan, prepared = load_plan(root)
        report.update(plan_fingerprint=plan["plan_fingerprint"], plan_sha256=sha(root / "plan.json"),
                      preparation_sha256=sha(root / "preparation.json"), source_manifest_sha256=plan["source_manifest_sha256"],
                      previous_retrospective_root=plan["previous_retrospective_root"],
                      previous_plan_sha256=plan["previous_plan_sha256"],
                      previous_source_manifest_sha256=plan["previous_source_manifest_sha256"],
                      planned_cases=plan["n_cases"], planned_settings=len({(e["case"]["model_id"], e["case"]["experiment_id"], e["case"]["setting_index"]) for e in plan["cases"]}),
                      models=sorted({e["case"]["model_id"] for e in plan["cases"]}),
                      experiments=sorted({e["case"]["experiment_id"] for e in plan["cases"]}),
                      seed_ids=sorted({e["case"]["seed_id"] for e in plan["cases"]}))
        for entry in plan["cases"]:
            record = dict(case=entry["case"], case_id=entry["case_id"], classification="missing", counts={},
                          complete_execution_coverage=False, complete_tuning_coverage=False,
                          planned_tasks=None, eligible_tasks=0, candidates=[], arms=dict.fromkeys(ARMS))
            record.update(elapsed_seconds=None, process_cpu_seconds=None)
            try:
                result, hashes, checks = load_case(root, plan, entry, prepared[entry["case_id"]])
                record.update({key: result[key] for key in ("counts", "planned_tasks", "eligible_tasks", "complete_execution_coverage", "complete_tuning_coverage", "elapsed_seconds", "process_cpu_seconds")})
                record["classification"] = "complete" if result["success"] else "incomplete"
                report["input_hashes"].append(hashes)
                totals.update(result["counts"])
                inventory = dict(case_id=entry["case_id"], **result["inventory"])
                inventories.append(inventory)
                record["inventory"] = result["inventory"]
                if result["success"]:
                    # Validate one case's full compact score inventory, then
                    # retain only five selected pointers and scalar metadata.
                    record.update(arms=result["arms"])
                    record["winner_features"] = {arm: winner_features(result, arm) for arm in ARMS}
                    report["historical_endpoint_comparisons"] += 3
                    report["minimum_bic_checks"].extend(dict(case_id=entry["case_id"], **check) for check in checks)
                else:
                    report["errors"].append(dict(case_id=entry["case_id"], classification="incomplete", errors=result["errors"]))
            except FileNotFoundError as error:
                report["errors"].append(dict(case_id=entry["case_id"], classification="missing", error=str(error)))
            except (OSError, ValueError, KeyError, TypeError) as error:
                record["classification"] = "corrupt"
                report["errors"].append(dict(case_id=entry["case_id"], classification="corrupt", error=str(error)))
            case_counts[record["classification"]] += 1
            report["case_results"].append(record)
        report["completed_cases"] = case_counts["complete"]
        report["complete_execution_coverage"] = all(r["complete_execution_coverage"] and r["classification"] == "complete" for r in report["case_results"])
        report["complete_tuning_coverage"] = all(r["complete_tuning_coverage"] and r["classification"] == "complete" for r in report["case_results"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(dict(classification="campaign_provenance", error=str(error)))
    report.update(task_counts=dict(totals), case_counts=dict(case_counts))
    report["inventory_totals"] = {key: sum(row[key] for row in inventories) for key in INVENTORY_COUNTS}
    report["worker_process_cpu_seconds"] = sum(r["process_cpu_seconds"] for r in report["case_results"] if r["process_cpu_seconds"] is not None)
    report["worker_elapsed_seconds_sum"] = sum(r["elapsed_seconds"] for r in report["case_results"] if r["elapsed_seconds"] is not None)
    report["worker_timings_available_cases"] = sum(r["elapsed_seconds"] is not None for r in report["case_results"])
    rows, curves, differences, blocks = build_tables(report["case_results"])
    coverage = [{key: r[key] for key in ("case_id", "classification", "planned_tasks", "eligible_tasks", "counts", "complete_execution_coverage", "complete_tuning_coverage")} for r in report["case_results"]]
    runtimes = [{key: r[key] for key in ("case_id", "classification", "elapsed_seconds", "process_cpu_seconds")} for r in report["case_results"]]
    outputs = []
    for name, table in (("per-seed-results.csv", rows), ("performance-curves.csv", curves),
                        ("paired-differences.csv", differences), ("paired-seed-blocks.csv", blocks),
                        ("coverage.csv", coverage), ("state-inventory.csv", inventories), ("case-runtime.csv", runtimes),
                        ("minimum-bic-checks.csv", report["minimum_bic_checks"])):
        write_csv(output / name, table); outputs.append(name)
    report.update(setting_results=curves, paired_differences=differences, paired_seed_blocks=blocks,
                  selection_counts=selection_counts(report["case_results"], rows))
    # Candidate score/state lists stay in hash-bound case files. Keep selected
    # pointers and per-case inventories here, avoiding a second giant score dump.
    for record in report["case_results"]:
        record.pop("candidates", None)
    if make_plots and curves:
        try:
            outputs.extend(plot_curves(output, curves))
        except (ImportError, OSError, ValueError) as error:
            report["errors"].append(dict(classification="plotting", error=str(error)))
    report["success"] = not report["errors"] and report["complete_execution_coverage"]
    outputs.append(write_report(output, report))
    report["output_hashes"] = {name: sha(output / name) for name in outputs}
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                  summary_elapsed_seconds=time.monotonic() - started,
                  summary_process_cpu_seconds=time.process_time() - cpu_started)
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", "--root", dest="root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = summarize(args.root, args.output_root, make_plots=not args.no_plots)
    print(json.dumps({k: report.get(k) for k in ("success", "planned_cases", "completed_cases", "case_counts", "task_counts")}, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
