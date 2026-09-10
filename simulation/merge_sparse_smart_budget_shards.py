"""Audit and merge disjoint tuning shards without fitting or rebasing scores.

Every shard has its own fixed validation reference. Local trajectory scores,
including scalar-only early evaluations, retain that reference. Cross-shard
selection is rebuilt from direct pairwise comparisons of saved predictions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

import run_sparse_smart_budget_study as runner
import summarize_sparse_smart_budget_study as summary
from run_sparse_smart import _digest_json, _json_value
from sparse_smart_selection import PAIRWISE_RULE, pairwise_loss_difference


_IDENTITY_KEYS = ("schema_version", "method", "model", "experiment", "rd_seed_id",
                  "random_seed", "setting", "configuration", "generator_arguments")
_SHARED_KEYS = ("schema_version", "method", "model", "experiment", "rd_seed_id", "random_seed",
    "setting", "generator_arguments", "training_observed_input_fingerprint",
    "validation_observed_input_fingerprint", "evaluation_truth_fingerprint", "input_fingerprint",
    "validation_seed_metadata", "implementation_fingerprint", "implementation_fingerprint_scheme",
    "implementation_manifest", "n_train", "n_validation",
    "all_training_rows_used", "training_matches_legacy", "refit_on_all_data", "theorem_certified",
    "source_check_mode", "applicable")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid_key(params):
    return _digest_json(params)


def _cell(record):
    setting = runner.SimulationSetting(**record["setting"])
    model_id = runner.MODEL_NAMES.index(record["model"])
    exp_id = runner.EXPERIMENT_NAMES.index(record["experiment"])
    path = runner.result_path(Path("."), model=record["model"], experiment=record["experiment"],
                              setting=setting, seed_id=record["rd_seed_id"])
    return setting, dict(path=path, setting=setting, model_id=model_id, exp_id=exp_id,
                        seed_id=record["rd_seed_id"], random_seed=record["random_seed"])


def _merge_selection(rows, trajectories, data):
    """Rebuild ordered global/budget incumbent chains using retained factors."""
    points = {(t["grid_candidate_id"], cp["checkpoint_iteration"]): cp
              for t in trajectories for cp in t["checkpoints"]}
    predictions = {}

    def prediction(row):
        key = (row["grid_candidate_id"], row["trajectory_checkpoint_iteration"])
        if key not in predictions:
            cp = points[key]
            factor = trajectories[key[0]]["factor_states"][cp["selected_factor_key"]]
            predictions[key] = runner._factor_prediction(factor, data["X"])
        return predictions[key]

    winner, budget_winners = None, {}
    for row in rows:
        row.pop("selection_comparison", None)
        row.pop("budget_selection_comparison", None)
        if not row["success"]:
            continue
        budget = row["iteration_budget"]
        for key, incumbent in (("selection_comparison", winner),
                               ("budget_selection_comparison", budget_winners.get(budget))):
            difference = (None if incumbent is None else pairwise_loss_difference(
                prediction(row), data["Y"], prediction(incumbent)))
            row[key] = dict(incumbent_candidate_id=None if incumbent is None else incumbent["candidate_id"],
                            loss_difference=difference)
            if incumbent is None or difference < 0:
                if key == "selection_comparison":
                    winner = row
                else:
                    budget_winners[budget] = row
    return winner, budget_winners


def merge_records(records, *, config, source_artifacts=None, generate_data_fn=None):
    """Return a fully audited, JSON-safe full-cell record from exact shard coverage.

    Shards must contain one initialization penalty, one U penalty, and one or
    more V penalties. Their disjoint union must equal ``config`` exactly.
    ``source_artifacts`` optionally supplies one metadata dictionary per input
    (for example task_id/path/sha256). Input order does not affect grid order or
    tie breaking. No estimator is called.
    """
    records = list(records)
    config.validate()
    _require(bool(records), "No tuning shards supplied")
    sources = [{} for _ in records] if source_artifacts is None else list(source_artifacts)
    _require(len(sources) == len(records) and all(isinstance(item, dict) for item in sources),
             "source_artifacts must contain one metadata mapping per shard")
    first = records[0]
    setting, cell = _cell(first)
    _require(setting.inapplicability_reason() is None, "Inapplicable cells do not have fitted tuning shards")
    resolved = _json_value(runner.resolved_configuration(setting, config))
    full_grid = resolved["candidate_grid"]
    grid_lookup = {_grid_key(params): j for j, params in enumerate(full_grid)}
    _require(len(grid_lookup) == len(full_grid), "Full tuning grid contains duplicates")
    target_options = _json_value(asdict(config))
    fixed_options = {k: v for k, v in target_options.items()
                     if k not in ("init_penalties", "penalties_u", "penalties_v")}
    generator = generate_data_fn or runner.external_runner.old_runner._load_generator()
    cache = summary._RegeneratedDataCache()
    trajectory_by_grid, rows_by_grid, references, source_records = {}, {}, {}, []
    for record, supplied_source in zip(records, sources):
        _require("tuning_shard_merge" not in record, "A tuning shard must be an ordinary runner record")
        _require(record.get("selection_rule") == PAIRWISE_RULE and "selection_score" in record,
                 "Tuning shards require the current pairwise validation selection rule")
        for name in _SHARED_KEYS:
            _require(record.get(name) == first.get(name) and (name in record) == (name in first),
                     f"Tuning shard {name} mismatch")
        local_options = record["configuration"]["runner"]
        _require({k: v for k, v in local_options.items()
                  if k not in ("init_penalties", "penalties_u", "penalties_v")} == fixed_options,
                 "Tuning shard fixed configuration mismatch")
        _require(len(local_options["init_penalties"]) == len(local_options["penalties_u"]) == 1
                 and len(local_options["penalties_v"]) >= 1,
                 "Each tuning shard must contain one initialization and U penalty")
        summary.validate_record(record, data_cache=cache, generate_data_fn=generator, **cell)
        local_grid = record["configuration"]["candidate_grid"]
        record_digest = _digest_json(record)
        mapping = {}
        for local_id, params in enumerate(local_grid):
            key = _grid_key(params)
            _require(key in grid_lookup, "Tuning shard contains a candidate outside the full grid")
            global_id = grid_lookup[key]
            _require(global_id not in trajectory_by_grid, "Duplicate tuning shard candidate coverage")
            mapping[local_id] = global_id
            trajectory = deepcopy(record["trajectories"][local_id])
            trajectory["grid_candidate_id"] = global_id
            # Diagnostics describe the local fit reference; its IDs deliberately
            # stay local and are bound by this explicit provenance mapping.
            trajectory["tuning_shard_origin"] = dict(local_grid_candidate_id=local_id,
                configuration_fingerprint=record["configuration_fingerprint"],
                record_digest=record_digest)
            trajectory_by_grid[global_id] = trajectory
            rows_by_grid[global_id] = {
                row["iteration_budget"]: deepcopy(row) for row in record["selection_history"]
                if row["grid_candidate_id"] == local_id}
            references[str(global_id)] = dict(
                validation_reference_prediction=deepcopy(record.get("validation_reference_prediction")),
                selection_reference=deepcopy(record.get("selection_reference")))
        artifact = deepcopy(supplied_source)
        if "sha256" in artifact:
            summary._fingerprint(artifact["sha256"], "source artifact sha256")
        declared = artifact.get("grid_candidate_ids")
        _require(declared is None or sorted(declared) == sorted(mapping.values()),
                 "Source artifact grid candidate mapping mismatch")
        artifact.update(record_digest=record_digest,
            configuration_fingerprint=record["configuration_fingerprint"],
            implementation_fingerprint=record["implementation_fingerprint"],
            implementation_source_locations=deepcopy(record.get("implementation_source_locations", {})),
            local_to_global_grid_ids={str(k): v for k, v in mapping.items()},
            grid_candidate_ids=sorted(mapping.values()))
        source_records.append(artifact)
    _require(set(trajectory_by_grid) == set(range(len(full_grid))), "Missing tuning shard candidate coverage")
    trajectories = [trajectory_by_grid[j] for j in range(len(full_grid))]
    rows = []
    for budget in config.iteration_budgets:
        for j in range(len(full_grid)):
            row = rows_by_grid[j][budget]
            row.update(candidate_id=len(rows), grid_candidate_id=j, params=deepcopy(full_grid[j]))
            rows.append(row)
    data = summary._data(setting, first["random_seed"], config, cache, generator)
    winner, budget_winners = _merge_selection(rows, trajectories, data)
    cap_data = dict(X_validation=data["X"], Y_validation=data["Y"])
    outcomes = runner._cap_outcomes(rows, trajectories, resolved, config, cap_data)
    final = outcomes[-1]
    value = deepcopy(first)
    value.update(configuration=resolved, trajectories=trajectories, selection_history=rows,
        cap_outcomes=outcomes, validation_reference_prediction=None,
        validation_reference_scope="trajectory",
        selection_reference=dict(kind="per_trajectory", mapping="trajectory_validation_references"),
        trajectory_validation_references={str(j): references[str(j)] for j in range(len(full_grid))},
        selection_score=None if winner is None else winner["selection_score"],
        selected_budget=final["winner_origin_budget"], selected_candidate_id=final["winner_candidate_id"],
        selected_iteration=final["selected_iteration"], validation_loss=final["validation_mse"],
        avg_err=final["coefficient_error"], success=final["success"],
        status="complete" if final["coverage_complete"] else "partial" if final["success"] else "all_candidates_failed",
        failure_reason=None if final["coverage_complete"] else "unresolved_maximum_budget",
        tuner_status="selected" if final["success"] else "no_successful_candidate",
        fit_time_sec=sum(float(record.get("fit_time_sec", sum(t["elapsed_time_sec"] for t in record["trajectories"])))
                         for record in records),
        elapsed_time_sec=sum(float(record.get("elapsed_time_sec") or record.get("fit_time_sec") or 0.) for record in records))
    value["configuration_fingerprint"] = _digest_json({key: value[key] for key in _IDENTITY_KEYS})
    budget_statuses = []
    for budget in config.iteration_budgets:
        current = [row for row in rows if row["iteration_budget"] == budget]
        best = budget_winners.get(budget)
        successful = sum(row["success"] for row in current)
        reached = sum(row["budget_reached"] for row in current)
        budget_statuses.append(dict(iteration_budget=budget, success=best is not None,
            successful_candidates=successful, failed_candidates=len(current)-successful,
            best_candidate_id=None if best is None else best["candidate_id"],
            best_validation_mse=None if best is None else best["validation_mse"],
            best_selection_score=None if best is None else best["selection_score"],
            budget_reached_candidates=reached, unreached_candidates=len(current)-reached,
            budget_fully_covered=reached == len(current)))
    value["tuning_diagnostics"] = dict(
        derived_from="audited_tuning_shard_merge", theorem_certified=False,
        selection_metric="pairwise_validation_loss_difference", selection_rule=PAIRWISE_RULE,
        validation_reference_scope="trajectory", selection_scores_comparable_across_trajectories=False,
        selection_reference=value["selection_reference"],
        refit_on_all_data=False, split_mode="explicit_validation", training_rows=setting.n,
        validation_rows=config.n_validation, validation_score_is_independent_test_estimate=False,
        checkpoint_execution="continuous_trajectory", checkpoint_interval=config.checkpoint_interval,
        checkpoint_iterations=sorted({0, *config.iteration_budgets,
            *range(config.checkpoint_interval, config.iterations+1, config.checkpoint_interval)}),
        validation_iterations=list(config.validation_iterations), validation_schedule=runner.validation_schedule(config),
        candidate_records_are_checkpoint_prefixes=True, elapsed_time_scope="sum_of_shard_work_not_wall_time",
        trajectory_fits=len(trajectories), successful_trajectories=sum(t["success"] for t in trajectories),
        failed_trajectories=sum(not t["success"] for t in trajectories),
        successful_candidate_fits=sum(row["success"] for row in rows),
        failed_candidate_fits=sum(not row["success"] for row in rows),
        selected_budget=value["selected_budget"], selected_candidate_id=value["selected_candidate_id"],
        selected_checkpoint_iteration=final["winner_checkpoint_iteration"],
        selected_budget_reached=False if winner is None else winner["budget_reached"],
        selected_checkpoint_retained_after_failure=bool(winner and not trajectories[winner["grid_candidate_id"]]["success"]),
        best_selection_score=value["selection_score"], budget_statuses=budget_statuses)
    value["tuning_shard_merge"] = dict(schema_version=1,
        source_artifacts=sorted(source_records, key=lambda item: item["grid_candidate_ids"]),
        source_count=len(records), trajectory_count=len(trajectories),
        candidate_order="budget-major then full configured initialization/U/V order",
        selection_score_scope="original trajectory fit reference; never rebased",
        fit_time_scope="sum_of_shard_fit_work_not_wall_time",
        elapsed_time_scope="sum_of_shard_elapsed_work_not_wall_time",
        merge_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    value = _json_value(value)
    summary.validate_record(value, data_cache=cache, generate_data_fn=generator, **cell)
    return value
