"""Frozen Discovery campaign: prepare shared data, then fit one grid point.

Production prepare/fit commands require Slurm and an absolute /scratch2 root.
No command aggregates results or archives workers. Unpenalized v2 candidates
use the explicitly recorded target-only RRR endpoint when the plan enables it.
Without selectors, plan retains the original 40-case Experiment 3/4 pilot.
Explicit experiments select all their canonical paper settings by default.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True


SCHEMA_VERSION = 1
METHOD = "SparseSMARTv2Pilot"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path, value):
    _atomic(path, lambda stream: stream.write(_canonical(value) + b"\n"))


def _npz(path, values):
    _atomic(path, lambda stream: np.savez_compressed(stream, **values))


def _history(path, value):
    _atomic(path, lambda stream: stream.write(gzip.compress(_canonical(value), compresslevel=5, mtime=0)))


def _array_fingerprint(data, names):
    # Identical names, ordering, dtype/shape encoding and bytes to the legacy
    # run_sparse_smart_tuned._array_fingerprint; no import of its fitting code.
    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(data[name])
        digest.update(json.dumps((name, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _configuration():
    _paths()
    from sparse_smart_v2.anchors import CHART_CONTINUATION_RULE
    return dict(init_penalties=[.003, .03, .1, .3, 1., 3.],
        penalties_u=[0., .001, .0025, .01, .04], penalties_v=[0., .001, .0025, .01],
        rank=5, iterations=2000, checkpoint_interval=250, validation_interval=50,
        validation_iterations=[1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200],
        validation_patience=300, validation_min_iterations=500,
        validation_min_relative_improvement=.001, n_validation=200,
        inverse_step=20., stationarity_tol=1e-6, max_backtracks=60,
        margins=dict(d_lower=.05, d_upper=12., gap=.01, anchor_min=.04, trial_radius=1.),
        generator_target_rank=5, generator_source_rank=10, target_noise_std=.5,
        validation_seed_tag=1397970481, source_basis="full_observed_svd",
        free_directions="per_case_free_directions", support_limits="full_outside_entry_capacity",
        initialization_spectrum="reject", refinement_solver="masked_chart_spectral_soft_hard", rrr_shortcut=True,
        adaptive_anchors=True, max_anchor_switches=16,
        chart_continuation_rule=CHART_CONTINUATION_RULE,
        tuning_uses_truth=False)


def _paths():
    code = Path(__file__).resolve().parents[1]
    for path in reversed((code / "simulation", code / "smart", code / "sparse-smart" / "src",
                          code / "sparse-smart-v2" / "src")):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return code


def _selection(values, name, default, allowed):
    if values is None:
        return tuple(default)
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence of distinct integer indices") from error
    if (not values or any(isinstance(value, (bool, np.bool_))
                          or not isinstance(value, (int, np.integer)) or value not in allowed for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be distinct indices in {list(allowed)}")
    return tuple(sorted(int(value) for value in values))


def _case_specs(*, models=None, experiments=None, seed_ids=None, setting_indices=None):
    code = _paths()
    # Import only the canonical setting/seed helpers. Their module loads no
    # estimator unless its separate fitting entry point is explicitly called.
    from run_restricted_rrr import experiment_settings, load_experiment_seeds
    seed_file = code / "simulation/data/random_seeds/experiment_seeds.csv"
    seeds = load_experiment_seeds(seed_file)
    selected_models = _selection(models, "models", (0, 1), range(3))
    selected_experiments = _selection(experiments, "experiments", (2, 3), range(4))
    selected_seeds = _selection(seed_ids, "seed_ids", range(3, 8), range(len(seeds)))
    pilot_settings = {2: (2, 3), 3: (4, 5)}
    cases = []
    for model in selected_models:
        for experiment in selected_experiments:
            settings = experiment_settings(model, experiment)
            defaults = range(len(settings)) if experiments is not None else pilot_settings[experiment]
            indices = _selection(setting_indices, "setting_indices", defaults, range(len(settings)))
            for setting_index in indices:
                setting = settings[setting_index]
                for seed_id in selected_seeds:
                    cases.append(dict(case_id=f"m{model}_e{experiment}_k{setting_index}_s{seed_id}",
                        model_id=model, model=f"model{model + 1}", experiment_id=experiment,
                        experiment=f"exp{experiment + 1}", setting_index=setting_index,
                        seed_id=seed_id, random_seed=int(seeds[seed_id]), n_train=setting.n, p=setting.p, q=setting.q,
                        sigma0=setting.sigma0, rank=setting.target_rank, source_rank=setting.source_rank,
                        free_directions=[setting.source_rank, setting.source_rank],
                        support_limits=[(setting.p-setting.source_rank)*setting.target_rank,
                                        (setting.q-setting.source_rank)*setting.target_rank]))
    return cases, _sha(seed_file)


def _configure_cases(cases, initializer_source_rank, rank11_policy):
    if rank11_policy not in ("error", "expand", "omit"):
        raise ValueError("rank11_policy must be error, expand, or omit")
    if initializer_source_rank is not None and (
            isinstance(initializer_source_rank, (bool, np.bool_))
            or not isinstance(initializer_source_rank, (int, np.integer)) or initializer_source_rank < 1):
        raise ValueError("initializer_source_rank must be a positive integer")
    result, omitted = [], []
    for original in cases:
        case = dict(original, free_directions=list(original["free_directions"]),
                    support_limits=list(original["support_limits"]))
        is_rank11 = case["experiment_id"] == 1 and case["rank"] == 11
        if is_rank11 and rank11_policy == "omit":
            omitted.append(dict(case_id=case["case_id"], reason="explicit_rank11_omission"))
            continue
        case["initializer_source_rank"] = int(case["source_rank"] if initializer_source_rank is None
                                               else initializer_source_rank)
        if is_rank11 and rank11_policy == "expand":
            case["initializer_source_rank"] = max(11, case["initializer_source_rank"])
            case["free_directions"] = [max(11, count) for count in case["free_directions"]]
            case["support_limits"] = [(dimension-free)*case["rank"] for dimension, free in
                                      zip((case["p"], case["q"]), case["free_directions"])]
        if not case["rank"] <= case["initializer_source_rank"] <= min(case["p"], case["q"]):
            hint = "; select --rank11-policy expand or omit explicitly" if is_rank11 else ""
            raise ValueError(f"initializer rank is incompatible with case {case['case_id']}{hint}")
        for side, dimension, free in zip(("u", "v"), (case["p"], case["q"]), case["free_directions"]):
            if not case["rank"] <= free <= dimension:
                hint = "; select --rank11-policy expand or omit explicitly" if is_rank11 else ""
                raise ValueError(f"free directions on side {side} are incompatible with case {case['case_id']}{hint}")
        result.append(case)
    if not result:
        raise ValueError("case selection contains no admissible requested cases")
    return result, omitted


def _spectral_configuration(config, cases, *, d_lower=None, spectral_gap=None, anchor_min=None):
    """Validate practical spectrum/anchor bounds before freezing a campaign plan."""
    config = dict(config, margins=dict(config["margins"]))
    for name, key, override in (("d_lower", "d_lower", d_lower), ("spectral_gap", "gap", spectral_gap),
                                ("anchor_min", "anchor_min", anchor_min)):
        value = config["margins"][key] if override is None else override
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be finite and strictly positive")
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{name} must be finite and strictly positive") from error
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
        config["margins"][key] = value
    if config["margins"]["anchor_min"] >= 1 / 16:
        raise ValueError("anchor_min must be less than 1/16, as required by Margins")
    lower, upper, gap = (np.longdouble(config["margins"][key]) for key in ("d_lower", "d_upper", "gap"))
    if not np.isfinite(upper) or lower >= upper:
        raise ValueError("d_lower must be less than the finite d_upper")
    # Preserve the package's existing Margins contract; do not defer a known
    # invalid calibration to the expensive preparation/allocation stage.
    if lower >= upper / 4:
        raise ValueError("d_upper must exceed 4 * d_lower, as required by Margins")
    for case in cases:
        if upper - lower < (case["rank"] - 1) * gap:
            raise ValueError(f"spectral bounds have no rank-{case['rank']} feasible spectrum for case {case['case_id']}")
    return config


def _initialization_configuration(config, init_penalties=None):
    if init_penalties is None:
        return config
    message = "init_penalties must be a nonempty sequence of distinct finite nonnegative numbers"
    try:
        if isinstance(init_penalties, (dict, set, frozenset)):
            raise TypeError("the initializer grid must be ordered")
        values = tuple(init_penalties)
        if not values or any(isinstance(value, (bool, np.bool_))
                             or not isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            raise ValueError(message)
        values = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if any(not np.isfinite(value) or value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError(message)
    return dict(config, init_penalties=values)


def _read(path):
    return json.loads(Path(path).read_text())


def _manifest(root, *, full=False, expected=None):
    path = root / "source-manifest.json"
    actual = _sha(path)
    if expected is not None and actual != expected:
        raise ValueError("source manifest hash differs from frozen plan")
    manifest = _read(path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported source manifest schema")
    source = Path(manifest["source_root"])
    if not source.is_absolute() or source.resolve() != (root / "source").resolve():
        raise ValueError("manifest source_root must be ROOT/source")
    current = Path(__file__).resolve()
    try:
        runner_relative = current.relative_to(source.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("run the frozen campaign source/simulation runner") from error
    selected = manifest["files"] if full else {runner_relative: manifest["files"].get(runner_relative)}
    for relative, digest in selected.items():
        item = source / relative
        if not item.resolve().is_relative_to(source.resolve()) or _sha(item) != digest:
            raise ValueError(f"source provenance mismatch: {relative}")
    return manifest, actual


def _verify_imports(manifest):
    source = Path(manifest["source_root"]).resolve()
    checked = set()
    for name, module in list(sys.modules.items()):
        if not (name in ("smart", "sparse_smart", "sparse_smart_v2", "external_validation_data")
                or name.startswith(("smart.", "sparse_smart.", "sparse_smart_v2."))):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path in checked:
            continue
        checked.add(path)
        if not path.is_relative_to(source):
            raise ValueError(f"imported {name} outside frozen source: {path}")
        relative = path.relative_to(source).as_posix()
        if manifest["files"].get(relative) != _sha(path):
            raise ValueError(f"imported module hash mismatch: {relative}")


def _load_api():
    _paths()
    import sparse_smart_v2 as api
    return api


def _require_slurm(root):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("prepare and fit must run inside a Slurm allocation")
    if not root.is_relative_to(Path("/scratch2")):
        raise ValueError("production campaign output must be under /scratch2")


def plan(root, initializer_source_rank=None, *, models=None, experiments=None, seed_ids=None,
         setting_indices=None, rank11_policy="error", d_lower=None, spectral_gap=None,
         anchor_min=None, init_penalties=None, refinement_solver=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _, manifest_hash = _manifest(root)
    selectors = dict(models=models, experiments=experiments, seed_ids=seed_ids, setting_indices=setting_indices)
    explicit_selection = any(value is not None for value in selectors.values())
    cases, seeds_hash = _case_specs(**selectors) if explicit_selection else _case_specs()
    cases, omitted = _configure_cases(cases, initializer_source_rank, rank11_policy)
    config = _spectral_configuration(_configuration(), cases, d_lower=d_lower, spectral_gap=spectral_gap,
                                     anchor_min=anchor_min)
    config = _initialization_configuration(config, init_penalties)
    if refinement_solver is not None:
        if refinement_solver not in ("masked_chart_spectral_soft_hard", "masked_anchor_projected"):
            raise ValueError("unsupported refinement_solver")
        config["refinement_solver"] = refinement_solver
    ranks = sorted({case["rank"] for case in cases})
    if ranks != [config["rank"]]:
        config["rank"] = ranks[0] if len(ranks) == 1 else "per_case"
        config["fitted_ranks"] = ranks
    tasks = []
    grid = list(product(config["init_penalties"], config["penalties_u"], config["penalties_v"]))
    for case in cases:
        for grid_index, (initial, left, right) in enumerate(grid):
            tasks.append(dict(task_id=len(tasks), case_id=case["case_id"], grid_index=grid_index,
                              init_penalty=initial, penalty_u=left, penalty_v=right))
    value = dict(schema_version=SCHEMA_VERSION, method=METHOD, root=str(root),
        source_manifest_sha256=manifest_hash, seed_file_sha256=seeds_hash,
        initializer_source_rank_override=initializer_source_rank,
        reference_cases_sha256=_sha(root / "reference-cases.json") if (root / "reference-cases.json").exists() else None,
        configuration=config, cases=cases, tasks=tasks, n_cases=len(cases), n_tasks=len(tasks))
    if explicit_selection or rank11_policy != "error":
        value["case_selection"] = dict(
            models=sorted({case["model_id"] for case in cases}),
            experiments=sorted({case["experiment_id"] for case in cases}),
            seed_ids=sorted({case["seed_id"] for case in cases}),
            setting_indices_by_experiment={str(experiment): sorted({case["setting_index"] for case in cases
                if case["experiment_id"] == experiment}) for experiment in sorted({case["experiment_id"] for case in cases})},
            rank11_policy=rank11_policy, omitted_cases=omitted,
            canonical_settings="run_restricted_rrr.experiment_settings",
            rank_semantics="case.rank is fitted rank; generator_target_rank remains the fixed truth rank")
    value["plan_fingerprint"] = _digest(value)
    target = root / "plan.json"
    if target.exists() and _read(target) != value:
        raise ValueError("existing plan differs; use a fresh campaign root")
    _json(target, value)
    _atomic(root / "work-items.tsv", lambda stream: stream.write(
        "".join(f"{row['task_id']}\n" for row in tasks).encode()))
    return value


def _load_plan(root, *, full_source=False):
    value = _read(root / "plan.json")
    fingerprint = value.pop("plan_fingerprint", None)
    if fingerprint != _digest(value) or value.get("root") != str(root):
        raise ValueError("plan identity/hash mismatch")
    value["plan_fingerprint"] = fingerprint
    if value.get("schema_version") != SCHEMA_VERSION or value.get("method") != METHOD:
        raise ValueError("unsupported pilot plan")
    manifest, _ = _manifest(root, full=full_source, expected=value["source_manifest_sha256"])
    if value["n_cases"] != len(value["cases"]) or value["n_tasks"] != len(value["tasks"]):
        raise ValueError("plan counts are inconsistent")
    return value, manifest


def _fingerprints(data):
    return dict(training_observed_input_fingerprint=_array_fingerprint(data, ("X", "Y", "C0")),
                validation_observed_input_fingerprint=_array_fingerprint(data, ("X_validation", "Y_validation")),
                evaluation_truth_fingerprint=_array_fingerprint(data, ("C_star",)))


def _read_case(root, case, plan_value):
    path = root / "cases" / case["case_id"]
    meta = _read(path / "case.json")
    if meta.get("case") != case or meta.get("plan_fingerprint") != plan_value["plan_fingerprint"]:
        raise ValueError("prepared case identity mismatch")
    for name in meta["files"]:
        if _sha(path / name) != meta["files"][name]:
            raise ValueError(f"prepared case file hash mismatch: {case['case_id']}/{name}")
    with np.load(path / "data.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    shapes = {"X": (case["n_train"], case["p"]), "Y": (case["n_train"], case["q"]),
              "C0": (case["p"], case["q"]), "C_star": (case["p"], case["q"]),
              "X_validation": (plan_value["configuration"]["n_validation"], case["p"]),
              "Y_validation": (plan_value["configuration"]["n_validation"], case["q"])}
    for name, shape in shapes.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all() or np.iscomplexobj(data[name]):
            raise ValueError(f"invalid saved {name} for {case['case_id']}")
    if _fingerprints(data) != meta["fingerprints"]:
        raise ValueError("prepared case input fingerprint mismatch")
    return data, meta


def _model_options(api, case, config, initial, left=0., right=0., *, iterations=None):
    return dict(rank=case["rank"], source_rank=case["initializer_source_rank"],
        margins=api.Margins(**config["margins"]),
        calibration=api.PracticalCalibration(initial, (left, right), config["inverse_step"], tuple(case["support_limits"])),
        free_directions=tuple(case["free_directions"]),
        iterations=config["iterations"] if iterations is None else iterations,
        max_backtracks=config["max_backtracks"], stationarity_tol=config["stationarity_tol"],
        adaptive_anchors=config["adaptive_anchors"], max_anchor_switches=config["max_anchor_switches"],
        refinement_solver=config["refinement_solver"], rrr_shortcut=config.get("rrr_shortcut", False))


def _rrr_reason(case, config, left, right):
    # Missing policy is intentionally legacy: existing frozen campaigns keep
    # their original estimator, preflight exclusion and artifact semantics.
    if not config.get("rrr_shortcut", False):
        return None
    _paths()
    from sparse_smart_v2.rrr import shortcut_reason
    return shortcut_reason(rank=case["rank"], p=case["p"], q=case["q"],
        free_directions=tuple(case["free_directions"]), penalties=(left, right),
        support_limits=tuple(case["support_limits"]))


def _preflight_initializers(api, case, config, source, data):
    records, arrays, cache = [], {}, {}
    fields = ("P", "d", "Q", "coefficient", "dual_gaps", "n_iter")
    for index, initial_penalty in enumerate(config["init_penalties"]):
        options = _model_options(api, case, config, initial_penalty, iterations=0)
        options["rrr_shortcut"] = False
        model = api.SparseSMARTv2(**options)
        model.fit(data["X"], data["Y"], source=source, _initialization_cache=cache)
        record = dict(index=index, init_penalty=initial_penalty, eligible=bool(model.success_),
                      status=model.status_, message=model.message_, metadata=model.metadata_)
        initial = getattr(model, "initialization_", None)
        if initial is not None:
            record.update(kkt_residual=float(initial.kkt_residual), converged=bool(initial.converged))
            for name in fields:
                arrays[f"i{index}_{name}"] = getattr(initial, name)
        records.append(record)
    return records, arrays


def prepare(root):
    root = Path(root).resolve()
    _require_slurm(root)
    value, manifest = _load_plan(root, full_source=True)
    api = _load_api()
    import external_validation_data
    config = value["configuration"]
    reference = None
    if value.get("reference_cases_sha256") is not None:
        if _sha(root / "reference-cases.json") != value["reference_cases_sha256"]:
            raise ValueError("reference cases hash mismatch")
        reference = _read(root / "reference-cases.json")["cases"]
    records = []
    for case in value["cases"]:
        case_root = root / "cases" / case["case_id"]
        if (case_root / "case.json").exists():
            _, meta = _read_case(root, case, value)
        else:
            data = external_validation_data.generate_external_validation(
                n_train=case["n_train"], p=case["p"], q=case["q"], sigma0=case["sigma0"],
                random_seed=case["random_seed"], n_validation=config["n_validation"],
                sigma=config["target_noise_std"], r_star=config["generator_target_rank"],
                r0_star=config["generator_source_rank"], seed_tag=config["validation_seed_tag"])
            if reference is not None:
                key = f"m{case['model_id']}_e{case['experiment_id']}_s{case['seed_id']}_k{case['setting_index']}"
                expected = reference[key]["fingerprints"]
                if _fingerprints(data) != expected:
                    raise ValueError(f"generated data differ from the prior pilot reference for {key}")
            source = api.prepare_source(api.ObservedSource(data["C0"]), p=case["p"], q=case["q"],
                                        source_rank=case["initializer_source_rank"])
            initializers, initializer_arrays = _preflight_initializers(api, case, config, source, data)
            selected = {name: np.asarray(data[name]) for name in
                        ("X", "Y", "C0", "C_star", "X_validation", "Y_validation")}
            _npz(case_root / "data.npz", selected)
            source_arrays = {name: getattr(source, name) for name in
                             ("left", "right", "leading_left", "leading_right", "source_singular_values")}
            _npz(case_root / "source.npz", source_arrays)
            _npz(case_root / "initializers.npz", initializer_arrays)
            meta = dict(schema_version=1, case=case, plan_fingerprint=value["plan_fingerprint"],
                fingerprints=_fingerprints(selected), source_fingerprint=_array_fingerprint(source_arrays, tuple(sorted(source_arrays))),
                validation_seed_metadata=data["validation_seed_metadata"],
                source_metadata=dict(mode=source.mode, noise_std=source.noise_std,
                                     gap_lower=source.gap_lower, cluster_size=source.cluster_size),
                initializers=initializers, reference_verified=reference is not None,
                files={name: _sha(case_root / name) for name in ("data.npz", "source.npz", "initializers.npz")})
            _json(case_root / "case.json", meta)
        records.append(dict(case_id=case["case_id"], case_json_sha256=_sha(case_root / "case.json"),
                            files=meta["files"], eligible_initializers=sum(row["eligible"] for row in meta["initializers"]),
                            initializers=meta["initializers"],
                            rrr_candidates=sum(_rrr_reason(case, config, left, right) is not None
                                for left, right in product(config["penalties_u"], config["penalties_v"]))
                                * len(config["init_penalties"])))
        print(json.dumps(dict(case_id=case["case_id"], prepared=True,
                              eligible_initializers=records[-1]["eligible_initializers"])), flush=True)
    _verify_imports(manifest)
    ready = dict(schema_version=1, status="complete", success=True, finished_at=_now(),
        plan_sha256=_sha(root / "plan.json"), source_manifest_sha256=value["source_manifest_sha256"],
        n_cases=value["n_cases"], n_tasks=value["n_tasks"], cases=records,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    blocked = [row["case_id"] for row in records if row["eligible_initializers"] == 0 and row["rrr_candidates"] == 0]
    _json(root / "preparation-preflight.json", dict(ready, blocked_cases=blocked,
          status="blocked" if blocked else "complete", success=not blocked))
    if blocked:
        raise RuntimeError(f"No eligible initializer in {len(blocked)} cases; preparation readiness withheld")
    _json(root / "preparation.json", ready)
    return ready


def _state_arrays(model, checkpoint=None):
    if getattr(model, "method_", None) == "target_rrr":
        if not hasattr(model, "coefficient_"):
            return {}
        # Physical coordinates require neither source frames nor a chart.
        values = {"states_schema_version": np.asarray(3, dtype=np.int64)}
        coefficient = model.coefficient_ if checkpoint is None else checkpoint.coefficient
        for prefix in ("selected", "terminal"):
            values[f"{prefix}_coefficient"] = coefficient
            for field, array in model.factors_.items():
                values[f"{prefix}_{field}"] = array
        for iteration, snap in getattr(model, "checkpoints_", {}).items():
            values[f"checkpoint_{iteration}_coefficient"] = snap.coefficient
        return values
    values = {"states_schema_version": np.asarray(2, dtype=np.int64)}
    def save_chart(prefix, chart):
        for field in ("anchors_u", "anchors_v", "center_u", "center_v"):
            values[prefix + field] = getattr(chart, field)

    if checkpoint is not None:
        states = {"terminal": checkpoint.state, "selected": checkpoint.selected_state}
        charts = {"terminal": checkpoint.chart, "selected": checkpoint.selected_chart}
    else:
        states = {name: getattr(model, attr) for name, attr in
                  (("terminal", "last_state_"), ("selected", "state_")) if hasattr(model, attr)}
        charts = {"terminal": getattr(model, "chart_", None),
                  "selected": getattr(model, "selected_chart_", getattr(model, "chart_", None))}
    for name, state in states.items():
        chart = charts[name]
        P, d, Q = chart.reconstruct(state)
        values.update({f"{name}_state": state, f"{name}_P": P, f"{name}_d": d, f"{name}_Q": Q})
        save_chart(name + "_", chart)
    for iteration, snap in getattr(model, "checkpoints_", {}).items():
        values[f"checkpoint_{iteration}_state"] = snap.state
        values[f"checkpoint_{iteration}_selected_state"] = snap.selected_state
        save_chart(f"checkpoint_{iteration}_terminal_", snap.chart)
        save_chart(f"checkpoint_{iteration}_selected_", snap.selected_chart)
    if charts["terminal"] is not None:
        # Compatibility aliases describe only this saved endpoint. Earlier
        # states and the validation incumbent always carry their own chart.
        save_chart("", charts["terminal"])
    if hasattr(model, "free_rows_"):
        values.update(free_rows_u=model.free_rows_.rows_u, free_rows_v=model.free_rows_.rows_v)
    return values


def _checkpoint_metadata(checkpoint):
    # AnchorChart is neither JSON serializable nor shared by all states after
    # a switch. Its exact arrays live beside the corresponding state in NPZ.
    return _jsonable({key: value for key, value in vars(checkpoint).items()
                      if key not in ("state", "selected_state", "chart", "selected_chart", "coefficient")})


def _live_estimator(base, destination, identity):
    # This narrow adapter observes the estimator's existing accepted-prefix
    # publication. It changes neither its trajectory nor its selection rule.
    class PublishedCheckpoints(dict):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner
            pointer = destination / "live-checkpoint.json"
            previous = _read(pointer) if pointer.exists() else {}
            # A restarted attempt also writes the inactive slot first.
            self.generation = 1 if previous.get("checkpoint_file") == "live-checkpoint-0.npz" else 0

        def __setitem__(self, iteration, checkpoint):
            super().__setitem__(iteration, checkpoint)
            owner = self.owner
            slot = self.generation % 2
            self.generation += 1
            checkpoint_name, history_name = f"live-checkpoint-{slot}.npz", f"live-history-{slot}.json.gz"
            arrays = _state_arrays(owner, checkpoint)
            _npz(destination / checkpoint_name, arrays)
            _history(destination / history_name, dict(
                history=getattr(owner, "history_", []), validation_history=getattr(owner, "validation_history_", []),
                history_chart_epochs=getattr(owner, "history_chart_epochs_", []),
                checkpoint=_checkpoint_metadata(checkpoint)))
            _json(destination / "live-checkpoint.json", dict(identity, iteration=int(iteration),
                selected_iteration=checkpoint.selected_iteration,
                best_validation_mse=checkpoint.best_validation_loss,
                checkpoint_file=checkpoint_name, history_file=history_name,
                checkpoint_sha256=_sha(destination / checkpoint_name),
                history_sha256=_sha(destination / history_name), written_at=_now()))

    class LiveEstimator(base):
        def __setattr__(self, name, value):
            if name == "checkpoints_" and isinstance(value, dict) and not isinstance(value, PublishedCheckpoints):
                value = PublishedCheckpoints(self)
            super().__setattr__(name, value)

    return LiveEstimator


def _metrics(coefficient, data):
    error = np.asarray(coefficient) - data["C_star"]
    return dict(coefficient_rmse=float(np.linalg.norm(error) / np.sqrt(error.size)),
        coefficient_frobenius_squared=float(np.sum(error * error)),
        training_prediction_mse=float(np.mean((data["X"] @ error) ** 2)),
        validation_mse=float(np.mean((data["Y_validation"] - data["X_validation"] @ coefficient) ** 2)))


class _PreflightIneligible(Exception):
    def __init__(self, record):
        self.record = record
        super().__init__(record["message"])


def fit(root, task_id):
    root = Path(root).resolve()
    _require_slurm(root)
    value, manifest = _load_plan(root)
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < len(value["tasks"]):
        raise ValueError("task index is outside the frozen plan")
    ready = _read(root / "preparation.json")
    if (ready.get("status") != "complete" or not ready.get("success")
            or ready.get("plan_sha256") != _sha(root / "plan.json")
            or ready.get("source_manifest_sha256") != value["source_manifest_sha256"]):
        raise ValueError("preparation marker is missing or belongs to a different plan")
    task = value["tasks"][task_id]
    if task["task_id"] != task_id:
        raise ValueError("task table index identity mismatch")
    case = next(case for case in value["cases"] if case["case_id"] == task["case_id"])
    identity = dict(schema_version=1, method=METHOD, task=task, case=case,
        plan_fingerprint=value["plan_fingerprint"], source_manifest_sha256=value["source_manifest_sha256"])
    destination = root / "tasks" / f"{task_id:05d}"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".fit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / "result.json").exists():
            existing = _read(destination / "result.json")
            if any(existing.get(key) != item for key, item in identity.items()):
                raise ValueError("existing result belongs to a different task or source")
            if existing.get("execution_success"):
                for name, digest in existing.get("files", {}).items():
                    if _sha(destination / name) != digest:
                        raise ValueError(f"existing result artifact is corrupt: {name}")
                return existing
        started, start_time = _now(), time.monotonic()
        _json(destination / "status.json", dict(identity, status="running", started_at=started,
              slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_step_id=os.environ.get("SLURM_STEP_ID"),
              pid=os.getpid()))
        model = None
        result = dict(identity, started_at=started, execution_success=False, success=False)
        try:
            case_record = next(row for row in ready["cases"] if row["case_id"] == case["case_id"])
            case_root = root / "cases" / case["case_id"]
            if _sha(case_root / "case.json") != case_record["case_json_sha256"]:
                raise ValueError("prepared case manifest hash mismatch")
            data, meta = _read_case(root, case, value)
            config = value["configuration"]
            initial_index = config["init_penalties"].index(task["init_penalty"])
            initial_record = meta["initializers"][initial_index]
            result.update(fingerprints=meta["fingerprints"], source_fingerprint=meta["source_fingerprint"],
                initialization_record=initial_record, initialization_cache="shared_preparation",
                avg_err=None)
            rrr_reason = _rrr_reason(case, config, task["penalty_u"], task["penalty_v"])
            if not initial_record["eligible"] and rrr_reason is None:
                raise _PreflightIneligible(initial_record)
            api = _load_api()
            _verify_imports(manifest)
            source, cache = None, None
            if rrr_reason is not None:
                result["initialization_cache"] = "not_used_target_rrr"
                result["initialization_preflight_bypassed"] = not initial_record["eligible"]
            else:
                with np.load(case_root / "source.npz", allow_pickle=False) as saved:
                    source_arrays = {name: saved[name] for name in saved.files}
                if _array_fingerprint(source_arrays, tuple(sorted(source_arrays))) != meta["source_fingerprint"]:
                    raise ValueError("saved source basis fingerprint mismatch")
                source = api.SourceBases(**source_arrays, **meta["source_metadata"])
                with np.load(case_root / "initializers.npz", allow_pickle=False) as saved:
                    initial_fields = {name: saved[f"i{initial_index}_{name}"] for name in
                                      ("P", "d", "Q", "coefficient", "dual_gaps", "n_iter")}
                initial = api.LassoInitialization(**initial_fields, kkt_residual=initial_record["kkt_residual"],
                                                 converged=initial_record["converged"])
                # The preparation record and NPZ hashes certify this exact source,
                # data and initializer. Populate the tuner's existing private cache
                # to avoid another SVD/Gram check or repeated reduced Lasso fit.
                cache = {"context": (id(data["X"]), id(data["Y"])),
                    ("source", id(source), case["initializer_source_rank"], case["p"], case["q"]): source,
                    ("initialization", id(source), case["rank"], case["initializer_source_rank"], task["init_penalty"]): initial}
            estimator = _live_estimator(api.SparseSMARTv2, destination, identity)
            model = estimator(**_model_options(api, case, config, task["init_penalty"],
                                               task["penalty_u"], task["penalty_v"]),
                checkpoint_iterations=tuple(range(config["checkpoint_interval"], config["iterations"] + 1,
                                                 config["checkpoint_interval"])),
                validation_interval=config["validation_interval"], validation_iterations=config["validation_iterations"],
                validation_patience=config["validation_patience"], validation_min_iterations=config["validation_min_iterations"],
                validation_min_relative_improvement=config["validation_min_relative_improvement"])
            model.fit(data["X"], data["Y"], source=source,
                      validation_data=(data["X_validation"], data["Y_validation"]), _initialization_cache=cache)
            result.update(execution_success=True, success=bool(model.success_), status="complete",
                fit_status=model.status_, message=model.message_, termination_reason=model.termination_reason_,
                n_iter=int(model.n_iter_), selected_iteration=model.selected_iteration_,
                optimization_converged=bool(model.optimization_converged_),
                numerical_work=getattr(model, "numerical_work_", None),
                anchor_switches=getattr(model, "anchor_switches_", []),
                last_rejection=getattr(getattr(model, "result_", None), "last_rejection", None),
                fingerprints=meta["fingerprints"], source_fingerprint=meta["source_fingerprint"],
                metadata=model.metadata_, fit_method=getattr(model, "method_", "sparse_smart_v2"),
                rrr_certificate=getattr(model, "rrr_certificate_", None), avg_err=None)
            if hasattr(model, "coefficient_"):
                result["selected_metrics"] = _metrics(model.coefficient_, data)
                result["terminal_metrics"] = _metrics(model.last_coefficient_, data)
                if model.success_:
                    result["avg_err"] = result["selected_metrics"]["coefficient_rmse"]
        except _PreflightIneligible as error:
            result.update(status="complete", execution_success=True, success=False,
                fit_status=error.record["status"], termination_reason=error.record["status"],
                message=error.record["message"], n_iter=0, selected_iteration=None,
                optimization_converged=False, metadata=error.record["metadata"],
                refinement_skipped="initializer_failed_strict_preflight")
        except (Exception, KeyboardInterrupt) as error:
            result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "execution_failed",
                error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                fit_status=getattr(model, "status_", None), n_iter=getattr(model, "n_iter_", 0))
        result["elapsed_seconds"] = time.monotonic() - start_time
        result["finished_at"] = _now()
        result["slurm"] = dict(job_id=os.environ.get("SLURM_JOB_ID"), step_id=os.environ.get("SLURM_STEP_ID"))
        result["files"] = {}
        if model is not None:
            _history(destination / "history.json.gz", dict(history=getattr(model, "history_", []),
                history_chart_epochs=getattr(model, "history_chart_epochs_", []),
                terminal_record=getattr(model, "terminal_record_", None),
                validation_history=getattr(model, "validation_history_", []),
                checkpoints={str(t): _checkpoint_metadata(cp)
                    for t, cp in getattr(model, "checkpoints_", {}).items()}))
            result["files"]["history.json.gz"] = _sha(destination / "history.json.gz")
            arrays = _state_arrays(model)
            if arrays:
                _npz(destination / "states.npz", arrays)
                result["files"]["states.npz"] = _sha(destination / "states.npz")
        _json(destination / "result.json", result)
        _json(destination / "status.json", dict(identity, status="finished", success=result["success"],
            execution_success=result["execution_success"], fit_status=result.get("fit_status"),
            result_sha256=_sha(destination / "result.json"), finished_at=result["finished_at"]))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare", "fit"):
        child = sub.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        if name == "fit":
            child.add_argument("--task", type=int, required=True)
        if name == "plan":
            child.add_argument("--initializer-source-rank", type=int)
            child.add_argument("--models", type=int, nargs="+", help="Zero-based model IDs; default: 0 1")
            child.add_argument("--experiments", type=int, nargs="+",
                               help="Zero-based experiment IDs; explicitly supplied experiments use all canonical settings")
            child.add_argument("--seed-ids", type=int, nargs="+", help="Seed-table row IDs; default: 3 4 5 6 7")
            child.add_argument("--setting-indices", type=int, nargs="+", help="Zero-based setting indices, valid for every selected experiment")
            child.add_argument("--rank11-policy", choices=("error", "expand", "omit"), default="error",
                               help="Explicitly expand initializer/free dimensions to at least11 for fitted rank11, or omit that setting")
            child.add_argument("--d-lower", type=float,
                               help="Positive lower bound on fitted singular values; default: 0.05")
            child.add_argument("--spectral-gap", type=float,
                               help="Positive minimum adjacent singular-value gap; default: 0.01")
            child.add_argument("--anchor-min", type=float,
                               help="Fixed anchor singular-value floor, strictly between 0 and 1/16; default: 0.04")
            child.add_argument("--init-penalties", type=float, nargs="+",
                               help="Distinct nonnegative initialization penalties; default: 0.003 0.03 0.1 0.3 1 3")
            child.add_argument("--refinement-solver", choices=("masked_chart_spectral_soft_hard", "masked_anchor_projected"),
                               help="Opt-in masked_anchor_projected includes the anchor constraint; requires full outside support caps")
    args = parser.parse_args(argv)
    if not args.root.is_absolute():
        parser.error("--root must be absolute")
    if args.command == "fit":
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "fit":
            result = fit(args.root, args.task)
        elif args.command == "plan":
            result = plan(args.root, args.initializer_source_rank, models=args.models, experiments=args.experiments,
                          seed_ids=args.seed_ids, setting_indices=args.setting_indices, rank11_policy=args.rank11_policy,
                          d_lower=args.d_lower, spectral_gap=args.spectral_gap,
                          anchor_min=args.anchor_min, init_penalties=args.init_penalties,
                          refinement_solver=args.refinement_solver)
        else:
            result = prepare(args.root)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in
        ("status", "success", "execution_success", "n_cases", "n_tasks", "fit_status", "n_iter") if key in result}))
    return 0 if result.get("execution_success", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
