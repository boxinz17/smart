"""Run one task: one leave-one-donor-out fold under one protocol variant.

    code/.venv/bin/python code/single_cell_v2/run_fold.py --counts <extract.npz> --folds folds.json \
        --out <run dir> (--task-index I | --fold donor-15078 --variant main) --mode smoke|production

Smoke mode fits a few SparseSMART candidates with short trajectories and small
Park et al. grids, and never touches test cells. Production mode fits the full
protocol and then, after every method is selected, scores the test cells.
Outputs go to <run dir>/<variant>/<fold>/; a completed task is not refitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--fold")
    parser.add_argument("--variant")
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if (args.task_index is None) == (args.fold is None or args.variant is None):
        parser.error("give either --task-index or both --fold and --variant")
    return args


def jsonable(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def score(method, X, Y):
    import numpy as np

    prediction = X @ method["coefficient"] + method.get("intercept", 0.0)
    residual = Y - prediction
    correlations = []
    for j in range(Y.shape[1]):
        if np.std(Y[:, j]) > 0 and np.std(prediction[:, j]) > 0:
            correlations.append(float(np.corrcoef(prediction[:, j], Y[:, j])[0, 1]))
    # Y is centered by the target training mean, so mean(Y**2) is the error of predicting that mean.
    return prediction, dict(rmse=float(np.sqrt(np.mean(residual ** 2))), mse=float(np.mean(residual ** 2)),
                            r2=float(1 - np.mean(residual ** 2) / np.mean(Y ** 2)),
                            median_protein_correlation=float(np.median(correlations)) if correlations else None,
                            n_proteins_correlated=len(correlations))


def run_task(extract, folds, fold_name, variant, mode, config, out_root, transformed=None, isotypes=None):
    import numpy as np

    import diagnostics
    import modeling
    from config import fingerprint
    from preprocess import fit_fold

    fold = next(f for f in folds["folds"] if f["fold"] == fold_name)
    directory = Path(out_root) / variant / fold_name
    result_path = directory / "result.json"
    config_sha = fingerprint(config)
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("complete") and previous.get("config_sha256") == config_sha and previous.get("mode") == mode:
            print(f"{variant}/{fold_name}: already complete", flush=True)
            return previous
        raise RuntimeError(f"{result_path} exists but does not match this task; move it before rerunning")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    settings = config["variants"][variant]
    result = dict(schema_version=1, fold=fold, variant=variant, variant_settings=settings, mode=mode,
                  config_sha256=config_sha, folds_config_sha256=folds["config_sha256"],
                  extract_provenance=extract.provenance, complete=False)

    fd = fit_fold(extract, fold, config, transformed, isotypes)
    p, q = fd.X_train.shape[1], fd.Y_train.shape[1]
    result["preprocessing"] = dict(fd.stats, n_genes=p, n_proteins=q, genes=fd.genes.tolist(),
                                   proteins=fd.proteins.tolist())

    main_source, main_info = modeling.fit_source(fd, "reduced_rank_ridge", config)
    if settings["source_estimator"] == "reduced_rank_ridge":
        C0, source_info = main_source, main_info
    else:
        C0, source_info = modeling.fit_source(fd, settings["source_estimator"], config)
    rank_hat, rsc_target = modeling.rsc(fd.X_train, fd.Y_train)
    rank = int(min(max(1, rank_hat + settings["rank_offset"]), p, q))
    rank0 = int(min(max(main_info["selected_rank"], rank), p, q))
    scales = modeling.target_scales(fd, rank)
    bases = modeling.prepare_bases(C0, rank0)
    result.update(source_fit=source_info, reduced_rank_ridge_source_rank=main_info["selected_rank"],
                  ranks=dict(rsc_target=rank_hat, rank=rank, rank0=rank0), scales=scales)

    methods = {}
    began = time.perf_counter()
    winners, candidates, sparse_summary = modeling.run_sparse_smart(fd, bases, rank, rank0, scales, config, mode)
    for label in ("sparse_smart", "sparse_smart_transfer_only"):
        winner = winners.get(label)
        methods[label] = (dict(success=True, coefficient=winner["coefficient"],
                               record={k: v for k, v in winner.items() if k not in ("coefficient", "prediction")},
                               elapsed_seconds=time.perf_counter() - began)
                          if winner else dict(success=False, error="no eligible candidate"))
    (directory / "sparse_smart_candidates.json").write_text(json.dumps(jsonable(candidates)) + "\n")
    result["sparse_smart"] = sparse_summary
    print(f"sparse_smart done {time.perf_counter() - began:.0f}s", flush=True)
    methods.update(modeling.run_comparators(fd, C0, bases, rank, rank0, scales, config))
    print("comparators " + " ".join(f"{k}={m.get('elapsed_seconds', 0):.0f}s" for k, m in methods.items()
                                    if not k.startswith("sparse_smart")), flush=True)
    methods["lasso"] = modeling.run_lasso(fd, config)
    print(f"lasso done {methods['lasso'].get('elapsed_seconds', 0):.0f}s", flush=True)
    methods["source_only"] = dict(success=True, coefficient=C0, elapsed_seconds=0.0,
                                  record=dict(note="source fit applied with the target training intercept"))
    if variant == "main":  # Park et al. uses raw source cells and no rank, so no variant changes it
        methods["park"] = modeling.run_park(fd, config, mode)
        print(f"park done {methods['park'].get('elapsed_seconds', 0):.0f}s", flush=True)

    for name, method in methods.items():
        if method.get("success"):
            _, method["validation"] = score(method, fd.X_val, fd.Y_val)

    if variant == "main":
        _, rsc_source = modeling.rsc(fd.X_source, fd.Y_source)
        result["diagnostics"] = diagnostics.run(fd, bases, rank, rank0, rsc_target, rsc_source, config)

    if mode == "production":  # test cells are scored only after every method has been selected
        predictions = dict(Y_test=fd.Y_test, test_cells=fd.test_cells)
        for name, method in methods.items():
            if method.get("success"):
                predictions[name], method["test"] = score(method, fd.X_test, fd.Y_test)
        np.savez_compressed(directory / "predictions.npz", **predictions)

    coefficients = {name: m["coefficient"] for name, m in methods.items() if m.get("success")}
    coefficients.update({f"{name}__intercept": m["intercept"] for name, m in methods.items()
                         if m.get("success") and "intercept" in m})
    np.savez_compressed(directory / "coefficients.npz", genes=fd.genes, proteins=fd.proteins, **coefficients)
    result["methods"] = {name: {k: v for k, v in m.items() if k not in ("coefficient", "intercept")}
                         for name, m in methods.items()}
    result.update(complete=True, elapsed_seconds=time.perf_counter() - started)
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(jsonable(result), indent=1) + "\n")
    temporary.replace(result_path)
    return result


def main(argv=None):
    args = parse_args(argv)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(args.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import CONFIG
    from data import load_extract
    from folds import load_folds, tasks

    folds = load_folds(args.folds)
    if args.task_index is not None:
        task = tasks(folds)[args.task_index]
        fold_name, variant = task["fold"], task["variant"]
    else:
        fold_name, variant = args.fold, args.variant
    print(f"task {fold_name} / {variant} / {args.mode}", flush=True)
    result = run_task(load_extract(args.counts), folds, fold_name, variant, args.mode, CONFIG, args.out)
    print(json.dumps({name: m.get("success") for name, m in result["methods"].items()}), flush=True)


if __name__ == "__main__":
    main()
