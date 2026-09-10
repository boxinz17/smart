"""Independent synthetic tuning shards; no optimization is performed."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_shard_schedule_fixture", Path(__file__).with_name("test_budget_validation_schedule.py"))
schedule_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(schedule_fixture)
fixtures = schedule_fixture.fixtures
runner, summary = fixtures.runner, fixtures.summary

import merge_sparse_smart_budget_shards as merger
from sparse_smart_selection import PAIRWISE_RULE, prediction_selection_score, pairwise_loss_difference
from sparse_smart_provenance import FINGERPRINT_SCHEME, _manifest_digest


CONFIG = runner.RunnerConfig(iteration_budgets=(4, 8), checkpoint_interval=4,
    validation_iterations=(1, 2, 3), init_penalties=(.01, .03),
    penalties_u=(.0025, .01), penalties_v=(.0025, .01))


def _synthetic(config=CONFIG, *, tied=False, failed_ids=()):
    base = schedule_fixture.early_validation_record()
    setting = runner.SimulationSetting(**base["setting"])
    data = fixtures.generated(0)
    full = fixtures._json_value(runner.resolved_configuration(setting, CONFIG))
    resolved = fixtures._json_value(runner.resolved_configuration(setting, config))
    template = deepcopy(base["trajectories"][0])
    base.update(configuration=resolved, trajectories=[], selection_history=[], fit_time_sec=.25,
                elapsed_time_sec=.5, tuning_diagnostics={}, failure_reason=None)
    all_predictions = {}
    for j, params in enumerate(resolved["candidate_grid"]):
        global_id = full["candidate_grid"].index(params)
        multiplier = 1. if tied else 1.-global_id*.03
        last = 6 if global_id in failed_ids else 8
        trajectory = deepcopy(template)
        trajectory.update(grid_candidate_id=j, params=params, factor_states={}, checkpoints=[],
            validation_history=[], history=[], n_iter=last,
            success=last == 8, status="completed" if last == 8 else "numerical_stagnation",
            termination_reason="max_iterations" if last == 8 else "numerical_stagnation",
            checkpoint_iterations=[0, 4, 8] if last == 8 else [0, 4])
        for t, scale in ((0, 1.), (1, .9), (2, .8), (3, 1.1), (4, .84), (8, .68)):
            if t > last:
                continue
            factor = dict(iteration=t, left=np.eye(setting.p, setting.target_rank).tolist(),
                right=np.eye(setting.q, setting.target_rank).tolist(),
                singular_values=(np.arange(5, 0, -1)*scale*multiplier).tolist())
            all_predictions[j, t] = runner._factor_prediction(factor, data["X"])
            if t != 3:
                trajectory["factor_states"][str(t)] = factor
                trajectory["history"].append(fixtures.diagnostic(t))
        if last != 8:
            trajectory["history"].append(fixtures.diagnostic(last))
        base["trajectories"].append(trajectory)
    reference = all_predictions[0, 0]
    base.update(validation_reference_prediction=reference.tolist(), selection_reference=dict(
        kind="first_evaluated_initializer", grid_candidate_id=0, iteration=0,
        reference_validation_mse=float(np.mean((reference-data["Y"])**2))))
    for j, trajectory in enumerate(base["trajectories"]):
        best = None
        for t in (0, 1, 2, 3, 4, 8):
            if (j, t) not in all_predictions:
                continue
            prediction = all_predictions[j, t]
            difference = None if best is None else pairwise_loss_difference(prediction, data["Y"], all_predictions[j, best])
            trajectory["validation_history"].append(dict(iteration=t,
                loss=float(np.mean((prediction-data["Y"])**2)), selection_rule=PAIRWISE_RULE,
                selection_score=prediction_selection_score(prediction, data["Y"], reference),
                selection_comparison=dict(incumbent_iteration=best, loss_difference=difference)))
            if best is None or difference < 0:
                best = t
        by_t = {row["iteration"]: row for row in trajectory["validation_history"]}
        for t in trajectory["checkpoint_iterations"]:
            selected = 0 if t == 0 else 2 if t == 4 else 8
            point = deepcopy(next(cp for cp in template["checkpoints"] if cp["checkpoint_iteration"] == t))
            point.update(selected_iteration=selected, terminal_factor_key=str(t), selected_factor_key=str(selected),
                selection_score=by_t[selected]["selection_score"], terminal_selection_score=by_t[t]["selection_score"])
            for label, at in (("terminal", t), ("selected", selected)):
                factor = trajectory["factor_states"][str(at)]
                coefficient = (np.asarray(factor["left"])*factor["singular_values"]) @ np.asarray(factor["right"]).T
                point[f"{label}_validation_mse"] = by_t[at]["loss"]
                point[f"{label}_coefficient_error"] = float(np.linalg.norm(coefficient-data["truth"])/np.sqrt(setting.p*setting.q))
            trajectory["checkpoints"].append(point)
    for budget in config.iteration_budgets:
        for j, trajectory in enumerate(base["trajectories"]):
            point = max((cp for cp in trajectory["checkpoints"] if cp["checkpoint_iteration"] <= budget),
                        key=lambda cp: cp["checkpoint_iteration"])
            base["selection_history"].append(dict(candidate_id=len(base["selection_history"]), grid_candidate_id=j,
                iteration_budget=budget, params=trajectory["params"], success=True,
                budget_reached=point["checkpoint_iteration"] == budget,
                trajectory_checkpoint_iteration=point["checkpoint_iteration"], n_iter=point["checkpoint_iteration"],
                selected_iteration=point["selected_iteration"], validation_mse=point["selected_validation_mse"],
                selection_score=point["selection_score"], selection_rule=PAIRWISE_RULE,
                status="completed", termination_reason="checkpoint", trajectory_status=trajectory["status"],
                trajectory_termination_reason=trajectory["termination_reason"]))
    # Independently write the two incumbent chains in canonical order.
    winner, budget_winners = None, {}
    for row in base["selection_history"]:
        for key, incumbent in (("selection_comparison", winner),
                               ("budget_selection_comparison", budget_winners.get(row["iteration_budget"]))):
            difference = None if incumbent is None else pairwise_loss_difference(
                all_predictions[row["grid_candidate_id"], row["selected_iteration"]], data["Y"],
                all_predictions[incumbent["grid_candidate_id"], incumbent["selected_iteration"]])
            row[key] = dict(incumbent_candidate_id=None if incumbent is None else incumbent["candidate_id"],
                            loss_difference=difference)
            if incumbent is None or difference < 0:
                if key == "selection_comparison":
                    winner = row
                else:
                    budget_winners[row["iteration_budget"]] = row
    base["cap_outcomes"] = runner._cap_outcomes(base["selection_history"], base["trajectories"],
        resolved, config, dict(X_validation=data["X"], Y_validation=data["Y"]))
    final = base["cap_outcomes"][-1]
    base.update(selection_score=final["selection_score"], success=final["success"],
        status="complete" if final["coverage_complete"] else "partial",
        selected_budget=final["winner_origin_budget"], selected_candidate_id=final["winner_candidate_id"],
        selected_iteration=final["selected_iteration"], validation_loss=final["validation_mse"], avg_err=final["coefficient_error"])
    schedule_fixture._refresh_identity(base)
    return base


def _shards(**options):
    return [_synthetic(replace(CONFIG, init_penalties=(li,), penalties_u=(pu,)), **options)
            for li in CONFIG.init_penalties for pu in CONFIG.penalties_u]


def _audit(record):
    _, cell = merger._cell(record)
    return summary.validate_record(record, data_cache={}, generate_data_fn=fixtures.generator, **cell)


def _initial_failures(record, failed):
    """Convert specified local grid positions to failures before initialization."""
    record = deepcopy(record)
    for j in failed:
        record["trajectories"][j].update(status="initialization_failed", success=False, n_iter=0,
            termination_reason="initialization_failed", history=[], validation_history=[],
            checkpoint_iterations=[], checkpoints=[], factor_states={})
        for row in record["selection_history"]:
            if row["grid_candidate_id"] == j:
                row.update(success=False, budget_reached=False, trajectory_checkpoint_iteration=None,
                    n_iter=0, selected_iteration=None, validation_mse=None, selection_score=None,
                    status="initialization_failed", termination_reason="initialization_failed",
                    trajectory_status="initialization_failed", trajectory_termination_reason="initialization_failed")
    data = fixtures.generated(0)
    # Rebuild fixture global comparisons after removing failed candidates.
    winner, per_budget = None, {}
    for row in record["selection_history"]:
        row.pop("selection_comparison", None)
        row.pop("budget_selection_comparison", None)
        if not row["success"]:
            continue
        def prediction(item):
            factor = record["trajectories"][item["grid_candidate_id"]]["factor_states"][str(item["selected_iteration"])]
            return runner._factor_prediction(factor, data["X"])
        for key, incumbent in (("selection_comparison", winner),
                               ("budget_selection_comparison", per_budget.get(row["iteration_budget"]))):
            difference = None if incumbent is None else pairwise_loss_difference(prediction(row), data["Y"], prediction(incumbent))
            row[key] = dict(incumbent_candidate_id=None if incumbent is None else incumbent["candidate_id"], loss_difference=difference)
            if incumbent is None or difference < 0:
                if key == "selection_comparison":
                    winner = row
                else:
                    per_budget[row["iteration_budget"]] = row
    setting = runner.SimulationSetting(**record["setting"])
    config = summary._configuration(record, setting)
    record["cap_outcomes"] = runner._cap_outcomes(record["selection_history"], record["trajectories"],
        record["configuration"], config, dict(X_validation=data["X"], Y_validation=data["Y"]))
    final = record["cap_outcomes"][-1]
    record.update(selection_score=final["selection_score"], success=final["success"],
        status="complete" if final["coverage_complete"] else "partial" if final["success"] else "all_candidates_failed",
        selected_budget=final["winner_origin_budget"], selected_candidate_id=final["winner_candidate_id"],
        selected_iteration=final["selected_iteration"], validation_loss=final["validation_mse"], avg_err=final["coefficient_error"])
    return record


@pytest.mark.parametrize("options", [{}, {"tied": True}, {"failed_ids": (6, 7)}],
                         ids=["different-initializers", "exact-ties", "failed-continuations"])
def test_shards_match_unsplit_selection_caps_and_early_validation_audit(options):
    baseline = _synthetic(**options)
    expected = _audit(baseline)
    shards = _shards(**options)
    before = deepcopy(shards)
    merged = merger.merge_records(list(reversed(shards)), config=CONFIG, generate_data_fn=fixtures.generator)
    assert shards == before
    actual = _audit(merged)
    assert actual["caps"] == [{**cap, "selection_score": actual["caps"][i]["selection_score"]}
                              for i, cap in enumerate(expected["caps"])]
    for key in ("status", "selected_budget", "selected_candidate_id", "selected_iteration", "validation_loss", "avg_err"):
        assert merged[key] == baseline[key]
    assert merged["configuration"] == baseline["configuration"]
    assert merged["configuration_fingerprint"] == baseline["configuration_fingerprint"]
    assert merged["fit_time_sec"] == 1. and merged["elapsed_time_sec"] == 2.
    assert actual["validation_audit"] == expected["validation_audit"]
    assert actual["validation_audit"]["metadata_only_points"] == 8
    assert merged["tuning_diagnostics"]["derived_from"] == "audited_tuning_shard_merge"
    assert merged["validation_reference_scope"] == "trajectory"
    assert len(merged["trajectory_validation_references"]) == 8
    for artifact in merged["tuning_shard_merge"]["source_artifacts"]:
        shard = next(item for item in shards if item["configuration_fingerprint"] == artifact["configuration_fingerprint"])
        for local_id, global_id in artifact["local_to_global_grid_ids"].items():
            assert merged["trajectories"][global_id]["validation_history"] == shard["trajectories"][int(local_id)]["validation_history"]
            assert merged["trajectory_validation_references"][str(global_id)]["validation_reference_prediction"] == shard["validation_reference_prediction"]
    json.dumps(merged, allow_nan=False)
    if options.get("tied"):
        assert merged["selected_candidate_id"] == 8


def test_shard_artifact_hashes_and_order_are_preserved():
    shards = _shards()
    sources = [dict(path=f"task-{i}.json", task_id=f"task-{i}", sha256=str(i)*64,
                    grid_candidate_ids=[2*i, 2*i+1]) for i in range(4)]
    merged = merger.merge_records(shards, config=CONFIG, source_artifacts=sources,
                                  generate_data_fn=fixtures.generator)
    assert [item["sha256"] for item in merged["tuning_shard_merge"]["source_artifacts"]] == [str(i)*64 for i in range(4)]
    assert len(merged["tuning_shard_merge"]["merge_source_sha256"]) == 64
    assert merged["trajectory_validation_references"]["0"] != merged["trajectory_validation_references"]["2"]


def test_identical_source_contents_from_different_checkouts_preserve_each_location():
    shards = _shards()
    manifest = dict(api="injected-api", generator="fixture", files=[
        dict(name="simulation/source.py", sha256="b"*64)])
    for index, record in enumerate(shards):
        record.update(implementation_manifest=deepcopy(manifest),
            implementation_fingerprint_scheme=FINGERPRINT_SCHEME,
            implementation_fingerprint=_manifest_digest(manifest),
            implementation_source_locations={"simulation/source.py": f"/checkout-{index}/simulation/source.py"})
    merged = merger.merge_records(shards, config=CONFIG, generate_data_fn=fixtures.generator)
    assert merged["implementation_manifest"] == manifest
    sources = merged["tuning_shard_merge"]["source_artifacts"]
    assert [item["implementation_source_locations"] for item in sources] == [
        record["implementation_source_locations"] for record in shards]
    assert len({item["implementation_fingerprint"] for item in sources}) == 1


def test_reversed_local_v_order_uses_full_grid_ties_and_regenerates_data_once():
    shards = [_synthetic(replace(CONFIG, init_penalties=(li,), penalties_u=(pu,),
                                penalties_v=tuple(reversed(CONFIG.penalties_v))), tied=True)
              for li in CONFIG.init_penalties for pu in CONFIG.penalties_u]
    calls = []
    def generator(**kwargs):
        calls.append(kwargs)
        return fixtures.generator(**kwargs)
    merged = merger.merge_records(list(reversed(shards)), config=CONFIG, generate_data_fn=generator)
    assert len(calls) == 1
    assert merged["selected_candidate_id"] == 8
    assert [row["grid_candidate_id"] for row in merged["selection_history"]] == list(range(8))*2
    assert merged["tuning_shard_merge"]["source_artifacts"][0]["local_to_global_grid_ids"] == {"0": 1, "1": 0}


@pytest.mark.parametrize("all_failed", [False, True])
def test_initial_failures_remain_ineligible_and_cannot_resolve_coverage(all_failed):
    shards = _shards()
    shards = [_initial_failures(record, (0, 1) if all_failed or i == 3 else ())
              for i, record in enumerate(shards)]
    merged = merger.merge_records(shards, config=CONFIG, generate_data_fn=fixtures.generator)
    _audit(merged)
    expected_failed = set(range(8)) if all_failed else {6, 7}
    assert merged["status"] == ("all_candidates_failed" if all_failed else "partial")
    assert merged["success"] is (not all_failed)
    for cap in merged["cap_outcomes"]:
        assert not cap["coverage_complete"] and set(cap["unresolved_grid_candidate_ids"]) == expected_failed
    for row in merged["selection_history"]:
        if row["grid_candidate_id"] in expected_failed:
            assert not row["success"] and not row["budget_reached"]
            assert row["selection_score"] is None and "selection_comparison" not in row


@pytest.mark.parametrize("mutation,match", [
    (lambda shards: shards.pop(), "Missing tuning shard"),
    (lambda shards: shards.append(deepcopy(shards[0])), "Duplicate tuning shard"),
    (lambda shards: shards[1].update(training_observed_input_fingerprint="b"*64), "fingerprint mismatch"),
    (lambda shards: shards[1].update(implementation_fingerprint="b"*64), "implementation_fingerprint mismatch"),
    (lambda shards: shards[1]["generator_arguments"].update(sigma=1.), "generator_arguments mismatch"),
    (lambda shards: shards[1]["trajectories"][0]["validation_history"][1].update(selection_score=999.), "Factor prediction/selection history"),
])
def test_missing_duplicate_mismatched_or_unverified_shards_are_rejected(mutation, match):
    shards = _shards()
    mutation(shards)
    with pytest.raises(ValueError, match=match):
        merger.merge_records(shards, config=CONFIG, generate_data_fn=fixtures.generator)


@pytest.mark.parametrize("mutation", [
    lambda record: record.pop("trajectory_validation_references"),
    lambda record: record["trajectory_validation_references"].pop("0"),
    lambda record: record["trajectory_validation_references"].update(extra={}),
    lambda record: record["trajectory_validation_references"]["0"].update(unexpected=True),
    lambda record: record["trajectory_validation_references"]["0"].update(validation_reference_prediction=None),
    lambda record: record["trajectory_validation_references"]["2"].update(
        validation_reference_prediction=record["trajectory_validation_references"]["0"]["validation_reference_prediction"]),
    lambda record: record.pop("tuning_shard_merge"),
])
def test_merged_reference_mappings_are_complete_explicit_and_factor_verified(mutation):
    merged = merger.merge_records(_shards(), config=CONFIG, generate_data_fn=fixtures.generator)
    mutation(merged)
    with pytest.raises(ValueError):
        _audit(merged)
