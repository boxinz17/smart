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


CONFIG = runner.RunnerConfig(iteration_budgets=(4, 8), checkpoint_interval=4, validation_interval=4, validation_patience=None, n_validation=100,
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


@pytest.mark.parametrize("options", [{}, {"tied": True}, {"failed_ids": (6, 7)}],
                         ids=["complete", "exact-ties", "partial-coverage"])
def test_merge_with_audit_matches_existing_api_without_repeating_final_validation(monkeypatch, options):
    shards = list(reversed(_shards(**options)))
    sources = [dict(path=f"scratch/task-{index}.json", task_id=f"task-{index}",
                    sha256=str(index)*64, provenance_label="original-fit")
               for index in range(len(shards))]
    before = deepcopy((shards, sources))
    expected = merger.merge_records(shards, config=CONFIG, source_artifacts=sources,
                                    generate_data_fn=fixtures.generator)
    expected_audit = _audit(expected)
    validation_calls, generated_calls = [], []
    original_validate = summary.validate_record

    def validate(record, **kwargs):
        audit = original_validate(record, **kwargs)
        validation_calls.append((record, kwargs["data_cache"], audit))
        return audit

    def generator(**kwargs):
        generated_calls.append(kwargs)
        return fixtures.generator(**kwargs)

    def no_fitting(*args, **kwargs):
        pytest.fail("Merging saved records must not load or run an estimator")

    monkeypatch.setattr(summary, "validate_record", validate)
    monkeypatch.setattr(runner, "run_setting", no_fitting)
    monkeypatch.setattr(runner.external_runner.old_runner, "_load_sparse_api", no_fitting)
    merged, audit = merger.merge_records_with_audit(shards, config=CONFIG,
        source_artifacts=sources, generate_data_fn=generator)

    assert merged == expected
    assert audit == expected_audit
    assert (shards, sources) == before
    assert len(validation_calls) == len(shards) + 1
    assert all(call[0] is shard for call, shard in zip(validation_calls[:-1], shards))
    assert validation_calls[-1][0] is merged
    assert validation_calls[-1][2] is audit
    assert len({id(call[1]) for call in validation_calls}) == 1
    assert len(generated_calls) == 1
    assert all(source["provenance_label"] == "original-fit"
               for source in merged["tuning_shard_merge"]["source_artifacts"])
    assert "trajectories" not in audit and "factor_states" not in audit
    json.dumps((merged, audit), allow_nan=False)


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


def _without_shard(index=3, *, tied=False):
    records = _shards(tied=tied)
    declarations = [dict(grid_candidate_id=candidate, task_id=f"task-{index}", reason="user_canceled")
                    for candidate in range(2*index, 2*index+2)]
    records.pop(index)
    sources = [dict(task_id=f"task-{j}", path=f"task-{j}.json", sha256=str(j)*64)
               for j in range(4) if j != index]
    return records, sources, declarations


def test_explicit_missing_winning_shard_preserves_full_grid_and_audits_available_winner(monkeypatch):
    records, sources, declarations = _without_shard()
    before = deepcopy((records, sources, declarations))
    calls, generated_calls = [], []
    original_validate = summary.validate_record

    def validate(record, **kwargs):
        calls.append(record)
        return original_validate(record, **kwargs)

    def generator(**kwargs):
        generated_calls.append(kwargs)
        return fixtures.generator(**kwargs)

    def no_fit(*args, **kwargs):
        pytest.fail("Missing candidates must not trigger estimator fitting")

    monkeypatch.setattr(summary, "validate_record", validate)
    monkeypatch.setattr(runner, "run_setting", no_fit)
    monkeypatch.setattr(runner.external_runner.old_runner, "_load_sparse_api", no_fit)
    merged, audit = merger.merge_records_with_audit(records, config=CONFIG,
        source_artifacts=sources, unavailable_candidates=declarations, generate_data_fn=generator)
    assert (records, sources, declarations) == before
    assert len(calls) == len(records)+1
    assert calls[-1] is merged and len(generated_calls) == 1
    baseline = _synthetic()
    assert baseline["cap_outcomes"][-1]["winner_grid_candidate_id"] == 7
    assert merged["configuration"] == baseline["configuration"]
    assert merged["configuration_fingerprint"] == baseline["configuration_fingerprint"]
    assert [t["grid_candidate_id"] for t in merged["trajectories"]] == list(range(8))
    assert [r["candidate_id"] for r in merged["selection_history"]] == list(range(16))
    assert [r["grid_candidate_id"] for r in merged["selection_history"]] == list(range(8))*2
    assert merged["status"] == audit["status"] == "partial"
    assert merged["success"] is True
    assert audit["unavailable_grid_candidate_ids"] == [6, 7]
    assert audit["available_candidate_count"] == 6 and audit["planned_candidate_count"] == 8
    assert audit["failed_trajectories"] == 0
    assert audit["trajectory_status_counts"] == {"completed": 6, "unavailable": 2}
    assert audit["verified_factor_states"] == 30
    assert merged["tuning_diagnostics"]["trajectory_fits"] == 6
    assert merged["tuning_diagnostics"]["failed_trajectories"] == 0
    assert merged["tuning_diagnostics"]["failed_candidate_fits"] == 0
    assert merged["fit_time_sec"] == .75 and merged["elapsed_time_sec"] == 1.5
    for cap, audited_cap in zip(merged["cap_outcomes"], audit["caps"], strict=True):
        assert cap["coverage_complete"] is audited_cap["coverage_complete"] is False
        assert cap["unresolved_grid_candidate_ids"] == audited_cap["unresolved_candidate_ids"] == [6, 7]
        assert cap["budget_reached_count"] == 6
        available = [row for row in baseline["selection_history"]
                     if row["grid_candidate_id"] < 6 and row["iteration_budget"] <= cap["iteration_budget"]]
        expected = min(available, key=lambda row: (row["validation_mse"], row["candidate_id"]))
        assert cap["winner_candidate_id"] == expected["candidate_id"]
        assert cap["winner_grid_candidate_id"] == expected["grid_candidate_id"] == 5
        assert cap["validation_mse"] == expected["validation_mse"]
    for candidate in (6, 7):
        trajectory = merged["trajectories"][candidate]
        assert trajectory["status"] == "unavailable" and trajectory["success"] is False
        assert trajectory["n_iter"] is None and "elapsed_time_sec" not in trajectory
        assert trajectory["termination_reason"] == "allowed_missing_shard"
        assert trajectory["unavailable_origin"] == declarations[candidate-6]
        assert trajectory["history"] == trajectory["validation_history"] == trajectory["checkpoints"] == []
        assert trajectory["factor_states"] == {}
        for row in merged["selection_history"]:
            if row["grid_candidate_id"] == candidate:
                assert not row["success"] and not row["budget_reached"]
                assert all(row[key] is None for key in ("n_iter", "selected_iteration", "validation_mse",
                    "selection_score", "trajectory_checkpoint_iteration"))
                assert "selection_comparison" not in row and "budget_selection_comparison" not in row
    assert merged["tuning_shard_merge"]["schema_version"] == 2
    assert merged["tuning_shard_merge"]["unavailable_candidates"] == declarations
    json.dumps((merged, audit), allow_nan=False)


def test_missing_first_shard_keeps_original_tie_order_and_global_candidate_ids():
    records, sources, declarations = _without_shard(index=0, tied=True)
    merged, audit = merger.merge_records_with_audit(list(reversed(records)), config=CONFIG,
        source_artifacts=list(reversed(sources)), unavailable_candidates=list(reversed(declarations)),
        generate_data_fn=fixtures.generator)
    assert audit["unavailable_grid_candidate_ids"] == [0, 1]
    assert merged["tuning_shard_merge"]["unavailable_candidates"] == declarations
    assert [cap["winner_grid_candidate_id"] for cap in merged["cap_outcomes"]] == [2, 2]
    assert [cap["winner_candidate_id"] for cap in merged["cap_outcomes"]] == [2, 10]
    assert merged["tuning_shard_merge"]["source_artifacts"][0]["local_to_global_grid_ids"] == {"0": 2, "1": 3}


@pytest.mark.parametrize("declarations", [None, []])
def test_no_unavailable_candidates_preserves_existing_merge_and_audit_contract(declarations):
    records = _shards()
    expected = merger.merge_records(records, config=CONFIG, generate_data_fn=fixtures.generator)
    merged, audit = merger.merge_records_with_audit(records, config=CONFIG,
        unavailable_candidates=declarations, generate_data_fn=fixtures.generator)
    assert merged == expected
    assert audit == _audit(expected)
    assert merged["tuning_shard_merge"]["schema_version"] == 1
    assert "unavailable_grid_candidate_ids" not in audit
    assert "unavailable_candidates" not in merged["tuning_shard_merge"]


def test_unavailable_merge_does_not_require_optional_source_artifacts():
    records, _, missing = _without_shard()
    merged, audit = merger.merge_records_with_audit(records, config=CONFIG,
        unavailable_candidates=missing, generate_data_fn=fixtures.generator)
    assert audit["unavailable_grid_candidate_ids"] == [6, 7]
    assert all("task_id" not in artifact for artifact in merged["tuning_shard_merge"]["source_artifacts"])


@pytest.mark.parametrize("mutation", [
    lambda records, sources, missing: missing.pop(),
    lambda records, sources, missing: missing.append(deepcopy(missing[0])),
    lambda records, sources, missing: missing[0].update(grid_candidate_id=8),
    lambda records, sources, missing: missing[0].update(grid_candidate_id=-1),
    lambda records, sources, missing: missing[0].update(grid_candidate_id=True),
    lambda records, sources, missing: missing[0].update(grid_candidate_id=0),
    lambda records, sources, missing: missing[0].update(task_id=""),
    lambda records, sources, missing: missing[0].update(reason=""),
    lambda records, sources, missing: missing[0].update(extra="unrecognized"),
    lambda records, sources, missing: missing[0].update(task_id="task-0"),
    lambda records, sources, missing: sources[0].update(grid_candidate_ids=[6, 7]),
    lambda records, sources, missing: sources[0].update(task_id=[]),
    lambda records, sources, missing: records.append(deepcopy(records[0])),
])
def test_unavailable_declarations_cannot_hide_coverage_or_provenance_mismatches(mutation):
    records, sources, missing = _without_shard()
    mutation(records, sources, missing)
    with pytest.raises(ValueError):
        merger.merge_records_with_audit(records, config=CONFIG, source_artifacts=sources,
            unavailable_candidates=missing, generate_data_fn=fixtures.generator)


def test_unavailable_mode_still_requires_at_least_one_available_shard():
    missing = [dict(grid_candidate_id=j, task_id=f"task-{j}", reason="user_canceled") for j in range(8)]
    with pytest.raises(ValueError, match="No tuning shards|All candidates"):
        merger.merge_records_with_audit([], config=CONFIG, unavailable_candidates=missing,
                                        generate_data_fn=fixtures.generator)
    with pytest.raises(ValueError, match="All candidates"):
        merger.merge_records_with_audit(_shards(), config=CONFIG, unavailable_candidates=missing,
                                        generate_data_fn=fixtures.generator)


def test_unavailable_mode_does_not_excuse_numerically_corrupted_available_shard():
    records, sources, missing = _without_shard()
    records[0]["trajectories"][0]["factor_states"]["2"]["singular_values"][0] += 1.
    with pytest.raises(ValueError, match="Factor prediction"):
        merger.merge_records_with_audit(records, config=CONFIG, source_artifacts=sources,
            unavailable_candidates=missing, generate_data_fn=fixtures.generator)


@pytest.mark.parametrize("mutation", [
    lambda r: r["tuning_shard_merge"].update(schema_version=1),
    lambda r: r["tuning_shard_merge"].update(schema_version=2.),
    lambda r: r["tuning_shard_merge"].pop("unavailable_candidates"),
    lambda r: r["tuning_shard_merge"].update(available_candidate_count=8),
    lambda r: r["tuning_shard_merge"]["unavailable_candidates"][0].update(task_id="wrong-task"),
    lambda r: r["tuning_shard_merge"]["source_artifacts"][0].update(task_id="task-3"),
    lambda r: r["tuning_shard_merge"]["source_artifacts"][0].update(task_id=[]),
    lambda r: r["tuning_shard_merge"]["source_artifacts"][0]["local_to_global_grid_ids"].update({"0": 6}),
    lambda r: r["trajectories"][0]["tuning_shard_origin"].update(record_digest="b"*64),
    lambda r: r["trajectories"][6].update(n_iter=0),
    lambda r: r["trajectories"][6].update(elapsed_time_sec=0.),
    lambda r: r["trajectories"][6].update(success=True),
    lambda r: r["trajectories"][6].update(success=0),
    lambda r: r["trajectories"][6].update(grid_candidate_id=6.),
    lambda r: r["trajectory_validation_references"]["6"].update(selection_reference={}),
    lambda r: r["selection_history"][6].update(validation_mse=0.),
    lambda r: r["selection_history"][6].update(selected_iteration=0),
    lambda r: r["selection_history"][6].update(budget_reached=True),
    lambda r: r["selection_history"][6].update(budget_reached=0),
    lambda r: r["cap_outcomes"][0].update(coverage_complete=True),
])
def test_missing_input_merge_validator_rejects_invented_results_and_provenance(mutation):
    records, sources, missing = _without_shard()
    merged, _ = merger.merge_records_with_audit(records, config=CONFIG, source_artifacts=sources,
        unavailable_candidates=missing, generate_data_fn=fixtures.generator)
    mutation(merged)
    with pytest.raises(ValueError):
        _audit(merged)


def test_ordinary_runner_record_cannot_smuggle_an_unavailable_trajectory():
    record = _synthetic()
    declaration = dict(grid_candidate_id=0, task_id="task-0", reason="user_canceled")
    record["trajectories"][0] = summary._unavailable_trajectory(declaration, record["configuration"]["candidate_grid"][0])
    with pytest.raises(ValueError, match="explicit merge declaration"):
        _audit(record)


def test_legacy_shards_keep_their_original_configuration_schema_when_merged():
    records = _shards()
    fields = ("validation_interval", "validation_patience", "validation_min_iterations",
              "validation_min_relative_improvement")
    for record in records:
        for key in fields:
            del record["configuration"]["runner"][key]
        for cap in record["cap_outcomes"]:
            del cap["policy_complete"], cap["validation_stopped_count"]
        schedule_fixture._refresh_identity(record)
    merged, audit = merger.merge_records_with_audit(records, config=CONFIG,
        generate_data_fn=fixtures.generator)
    assert all(key not in merged["configuration"]["runner"] for key in fields)
    assert audit["status"] == "complete"
    assert summary._configuration(merged, runner.SimulationSetting(**merged["setting"])).validation_patience is None
