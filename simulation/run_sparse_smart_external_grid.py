"""Run the saved three-model, four-experiment grid with SparseSMART only.

Each setting retains its original training size and adds 100 independent tuning
observations. Existing per-cell checkpoints are verified by the external runner.
Processes isolate the legacy generator's global random state. Structural
inapplicability and solver failures remain explicit records, never zero errors.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import multiprocessing
import os
from pathlib import Path
import time

from run_restricted_rrr import (DEFAULT_SEED_FILE, EXPERIMENT_NAMES, MODEL_NAMES,
    experiment_settings, load_experiment_seeds)
from run_sparse_smart import _atomic_json_dump
from run_sparse_smart_external import RunnerConfig, result_path, run_setting
from batch_manifest import BatchManifest

HERE = Path(__file__).resolve().parent


def fit_cell(task):
    model_id, exp_id, setting, seed_id, random_seed, output_root, config, reuse_root = task
    model, experiment = MODEL_NAMES[model_id], EXPERIMENT_NAMES[exp_id]
    destination = result_path(output_root, model=model, experiment=experiment,
                              setting=setting, seed_id=seed_id)
    kwargs = dict(setting=setting, model=model, experiment=experiment,
                  seed_id=seed_id, random_seed=random_seed, config=config)
    reused_from = None
    prior = (result_path(reuse_root, model=model, experiment=experiment,
                        setting=setting, seed_id=seed_id) if reuse_root is not None else None)
    if not destination.exists() and prior is not None and prior.exists():
        # Verify the source checkpoint through the ordinary runner before
        # copying it. Configuration, implementation, data and seeds must match.
        outcome, result = run_setting(destination=prior, **kwargs)
        if outcome != "skipped":
            raise RuntimeError("Existing reuse checkpoint was unexpectedly fitted")
        _atomic_json_dump(result, destination)
        outcome, reused_from = "reused", str(prior)
    else:
        outcome, result = run_setting(destination=destination, **kwargs)
    return dict(model=model, experiment=experiment, setting=setting.suffix,
        seed_id=seed_id, outcome=outcome, status=result["status"], avg_err=result["avg_err"],
        n_train=result["n_train"], n_validation=result["n_validation"],
        failure_reason=result["failure_reason"], selected_iteration=result["selected_iteration"],
        selected_budget=result.get("selected_budget"),
        selected_candidate_id=result.get("selected_candidate_id"),
        termination_reason=result["termination_reason"],
        candidate_count=len(result["selection_history"]), failed_candidates=len(result["fit_errors"]),
        fit_time_sec=result["fit_time_sec"], path=str(destination), reused_from=reused_from)


def settings_for_profile(model_id, exp_id, profile):
    settings = experiment_settings(model_id, exp_id)
    if profile == "full":
        return settings
    if profile == "difficult":
        return tuple(s for s in settings if
            (exp_id == 2 and s.source_rank in (5, 7)) or (exp_id == 3 and s.sigma0 == .5))
    raise ValueError("profile must be full or difficult")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=int, nargs="+", choices=range(3), default=[0, 1, 2])
    parser.add_argument("--experiments", type=int, nargs="+", choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--output-root", type=Path, default=HERE / "result" / "sparse_smart_external")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--iteration-budgets", type=int, nargs="+",
                        help="Increasing checkpoint budgets ending at --iterations; default uses one budget")
    parser.add_argument("--profile", choices=("full", "difficult"), default="full")
    parser.add_argument("--initialization-spectrum", choices=("auto", "projected", "reject"), default="auto")
    parser.add_argument("--refinement-solver", choices=("auto", "chart", "anchor_projected"), default="auto")
    parser.add_argument("--reuse-root", type=Path,
                        help="Reuse existing cells only after exact runner checkpoint verification")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.seed_count <= 100 or args.workers < 1:
        parser.error("seed-count must be 1–100 and workers must be positive")
    if len(set(args.models)) != len(args.models) or len(set(args.experiments)) != len(args.experiments):
        parser.error("model and experiment lists must not contain duplicates")
    config = RunnerConfig(iterations=args.iterations,
        iteration_budgets=tuple(args.iteration_budgets) if args.iteration_budgets else None,
        initialization_spectrum=args.initialization_spectrum, refinement_solver=args.refinement_solver)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    output_root = args.output_root.resolve()
    reuse_root = args.reuse_root.resolve() if args.reuse_root is not None else None
    if reuse_root is not None and (reuse_root == output_root or reuse_root in output_root.parents
                                  or output_root in reuse_root.parents):
        parser.error("Reuse and output roots must be separate, non-nested directories")
    seeds = load_experiment_seeds(args.seed_file)
    if args.seed_count > len(seeds):
        parser.error("Not enough saved seeds")
    # Finish each smaller model before scheduling the larger one. Within a
    # model, sweep all settings for each seed so failures surface early.
    tasks = [(model, exp, setting, seed, int(seeds[seed]), output_root, config, reuse_root)
             for model in args.models for seed in range(args.seed_count)
             for exp in args.experiments for setting in settings_for_profile(model, exp, args.profile)]
    if not tasks:
        parser.error("Selected models/experiments/profile contain no settings")
    manifest = dict(started=datetime.now(timezone.utc).isoformat(),
        models=args.models, experiments=args.experiments, seed_ids=list(range(args.seed_count)),
        seed_file=str(args.seed_file.resolve()), expected_cells=len(tasks),
        expected_applicable=sum(task[2].inapplicability_reason() is None for task in tasks),
        expected_inapplicable=sum(task[2].inapplicability_reason() is not None for task in tasks),
        workers=args.workers, configuration=asdict(config), profile=args.profile,
        output_root=str(output_root), reuse_root=str(reuse_root) if reuse_root is not None else None,
        process_id=os.getpid(), process_group_id=os.getpgrp(),
        batch_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cells=[], errors=[])
    import json
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("cells", "errors")}), flush=True)
    if args.dry_run:
        return 0
    manifest_path = args.output_root / "expanded_pilot_manifest.json"
    started = time.perf_counter()
    with BatchManifest(manifest_path, manifest) as attempt:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(fit_cell, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    cell = future.result()
                    manifest["cells"].append(cell)
                    print(json.dumps(dict(done=len(manifest["cells"]), total=len(tasks), **cell)), flush=True)
                except Exception as error:
                    model, exp, setting, seed, *_ = futures[future]
                    failure = dict(model=MODEL_NAMES[model], experiment=EXPERIMENT_NAMES[exp],
                        setting=setting.suffix, seed_id=seed, exception=type(error).__name__, message=str(error))
                    manifest["errors"].append(failure)
                    print(json.dumps(failure), flush=True)
                manifest["status_counts"] = dict(Counter(c["status"] for c in manifest["cells"]))
                attempt.update()
        manifest.update(finished=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter() - started,
                        outcome_counts=dict(Counter(c["outcome"] for c in manifest["cells"])))
        attempt.finish(not manifest["errors"] and len(manifest["cells"]) == len(tasks))
    print(json.dumps({k: manifest[k] for k in ("wall_seconds", "status_counts", "outcome_counts", "errors")}), flush=True)
    # Recorded numerical failures are experimental outcomes. Exceptions or
    # missing records indicate an incomplete batch and produce a nonzero exit.
    return int(bool(manifest["errors"]) or len(manifest["cells"]) != len(tasks))


if __name__ == "__main__":
    raise SystemExit(main())
