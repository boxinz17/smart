"""Run the full-sample, full-source-span restricted-RRR pilot.

This driver intentionally implements only the smallest BI-SMART pilot requested
for the paper simulations.  For each legacy simulation cell it

1. regenerates exactly the same synthetic data as :mod:`run_SMART`,
2. fits one restricted RRR on all observations, and
3. records the normalized Frobenius coefficient error and diagnostics.

There is no screening, validation split, candidate selection, Wedin gate,
fallback estimator, or Gauss--Newton refinement in this runner.

Examples
--------
Run every sample-size setting for Model I and seed index zero::

    python run_restricted_rrr.py 0 0 0

Run only the first setting, without writing a result::

    python run_restricted_rrr.py 0 0 0 --setting-index 0 --dry-run
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import os
from pathlib import Path
import pickle
import tempfile
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED_FILE = SCRIPT_DIR / "data" / "random_seeds" / "experiment_seeds.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "result"

MODEL_NAMES = ("model1", "model2", "model3")
EXPERIMENT_NAMES = ("exp1", "exp2", "exp3", "exp4")
MODEL_DIMS = {
    "model1": (100, 50),
    "model2": (150, 100),
    "model3": (300, 200),
}
MODEL_SAMPLE_SIZES = {
    0: (200, 400, 600, 800, 1000),
    1: (300, 500, 700, 1000, 1200),
    2: (500, 700, 1000, 1200, 1500),
}
SUMMARY_DIAGNOSTIC_KEYS = (
    "source_singular_values",
    "rrr_cutoff_gap",
    "reduced_gram_condition",
)


@dataclass(frozen=True)
class SimulationSetting:
    """One cell in the existing SMART simulation grid."""

    n: int
    p: int
    q: int
    sigma0: float
    target_rank: int
    source_rank: int
    suffix: str

    def inapplicability_reason(self) -> str | None:
        """Return why direct restricted RRR is undefined, if applicable."""

        if self.target_rank < 1:
            return "target_rank_must_be_positive"
        if self.source_rank < 1:
            return "source_rank_must_be_positive"
        if self.target_rank > self.source_rank:
            return "target_rank_exceeds_source_rank"
        if self.source_rank > min(self.p, self.q):
            return "source_rank_exceeds_source_matrix_dimension"
        return None


def experiment_settings(model_id: int, exp_id: int) -> tuple[SimulationSetting, ...]:
    """Return settings in exactly the order used by ``run_SMART.py``.

    As in the legacy driver, experimental ``target_rank`` and ``source_rank``
    tune the estimator.  They are not passed to ``generate_data``; the data
    generator therefore retains its fixed true ranks five and ten.
    """

    if model_id not in range(len(MODEL_NAMES)):
        raise ValueError("model_id must be between 0 and 2.")
    if exp_id not in range(len(EXPERIMENT_NAMES)):
        raise ValueError("exp_id must be between 0 and 3.")

    model = MODEL_NAMES[model_id]
    p, q = MODEL_DIMS[model]
    sample_sizes = MODEL_SAMPLE_SIZES[model_id]
    base_n = sample_sizes[0]

    if exp_id == 0:
        return tuple(
            SimulationSetting(
                n=n,
                p=p,
                q=q,
                sigma0=0.01,
                target_rank=5,
                source_rank=10,
                suffix=f"n={n}",
            )
            for n in sample_sizes
        )
    if exp_id == 1:
        return tuple(
            SimulationSetting(
                n=base_n,
                p=p,
                q=q,
                sigma0=0.01,
                target_rank=target_rank,
                source_rank=10,
                suffix=f"r={target_rank}",
            )
            for target_rank in (1, 3, 5, 7, 9, 11)
        )
    if exp_id == 2:
        return tuple(
            SimulationSetting(
                n=base_n,
                p=p,
                q=q,
                sigma0=0.01,
                target_rank=5,
                source_rank=source_rank,
                suffix=f"rs={source_rank}",
            )
            for source_rank in (0, 3, 5, 7, 10, 15, 20)
        )

    return tuple(
        SimulationSetting(
            n=base_n,
            p=p,
            q=q,
            sigma0=sigma0,
            target_rank=5,
            source_rank=10,
            suffix=f"sigma0={sigma0}",
        )
        for sigma0 in (0.0, 0.01, 0.02, 0.05, 0.1, 0.5)
    )


def load_experiment_seeds(path: Path = DEFAULT_SEED_FILE) -> np.ndarray:
    """Load and validate the checked-in Monte-Carlo seed vector."""

    seeds = np.loadtxt(path, dtype=np.int64, delimiter=",", skiprows=1, usecols=1)
    seeds = np.atleast_1d(seeds)
    if seeds.ndim != 1 or seeds.size == 0:
        raise ValueError(f"No one-dimensional seed vector found in {path}.")
    return seeds


def result_path(
    output_root: Path,
    *,
    model: str,
    experiment: str,
    setting: SimulationSetting,
    seed_id: int,
) -> Path:
    """Construct a result path compatible with the existing directory layout."""

    filename = (
        f"RestrictedRRR_result_{model}_{experiment}_{setting.suffix}_"
        f"rd_seed_id={seed_id}.pkl"
    )
    return output_root / model / experiment / filename


def _load_implementations() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Import project packages lazily so ``--dry-run`` needs no installation."""

    from smart import generate_data
    from bi_smart import restricted_rrr

    return generate_data, restricted_rrr


def _plain_value(value: Any) -> Any:
    """Convert immutable/API diagnostic containers to pickle-stable values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_value(item) for item in value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def _atomic_pickle_dump(value: Any, destination: Path) -> None:
    """Write a complete pickle before atomically publishing its final name."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _base_result(
    *,
    setting: SimulationSetting,
    model: str,
    experiment: str,
    seed_id: int,
    random_seed: int,
    coefficient_true: np.ndarray,
) -> dict[str, Any]:
    """Create common result fields, including the four legacy keys."""

    # These are the effective defaults in smart.generate_data.  Recording them
    # makes clear that exp2/exp3 vary estimator ranks, as the legacy script did,
    # rather than changing the data-generating ranks.
    generator_arguments = {
        "n": setting.n,
        "p": setting.p,
        "q": setting.q,
        "sigma0": setting.sigma0,
        "sigma": 0.5,
        "r0_star": 10,
        "r_star": 5,
        "random_seed": random_seed,
    }
    return {
        "C_true": coefficient_true,
        "C_hat": None,
        "avg_err": None,
        "elapsed_time_sec": None,
        "method": "full_sample_restricted_rrr",
        "status": "pending",
        "success": False,
        "applicable": True,
        "failure_reason": None,
        "failure_message": None,
        "target_rank": setting.target_rank,
        "source_rank": setting.source_rank,
        "fit_time_sec": 0.0,
        "fit_diagnostics": {},
        "source_singular_values": None,
        "rrr_cutoff_gap": None,
        "reduced_gram_condition": None,
        "model": model,
        "experiment": experiment,
        "setting_suffix": setting.suffix,
        "setting": asdict(setting),
        "rd_seed_id": seed_id,
        "random_seed": random_seed,
        "generator_arguments": generator_arguments,
    }


def run_setting(
    *,
    setting: SimulationSetting,
    model: str,
    experiment: str,
    seed_id: int,
    random_seed: int,
    destination: Path,
    force: bool = False,
    generate_data_fn: Callable[..., Mapping[str, Any]] | None = None,
    restricted_rrr_fn: Callable[..., Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Generate, fit, and save one simulation cell.

    Returns ``("written", result)`` or ``("skipped", None)``.  A numerical
    failure represented by the estimator API is still a written result.
    Unexpected exceptions deliberately propagate so batch jobs exit nonzero
    instead of disguising a programming or infrastructure error as data.
    """

    if destination.exists() and not force:
        return "skipped", None

    if generate_data_fn is None or restricted_rrr_fn is None:
        default_generator, default_estimator = _load_implementations()
        if generate_data_fn is None:
            generate_data_fn = default_generator
        if restricted_rrr_fn is None:
            restricted_rrr_fn = default_estimator

    started = time.perf_counter()

    # Do not draw random numbers before this call.  The legacy generator resets
    # NumPy's global RNG from random_seed, so this reproduces run_SMART's arrays.
    data = generate_data_fn(
        n=setting.n,
        p=setting.p,
        q=setting.q,
        sigma0=setting.sigma0,
        random_seed=random_seed,
    )
    X = np.asarray(data["X"])
    Y = np.asarray(data["Y"])
    coefficient_true = np.asarray(data["C_star"])
    observed_source = np.asarray(data["C0"])
    result = _base_result(
        setting=setting,
        model=model,
        experiment=experiment,
        seed_id=seed_id,
        random_seed=random_seed,
        coefficient_true=coefficient_true,
    )

    inapplicability = setting.inapplicability_reason()
    if inapplicability is not None:
        result.update(
            status="inapplicable",
            applicable=False,
            failure_reason=inapplicability,
            failure_message=(
                "Direct restricted RRR requires 1 <= target_rank <= source_rank "
                "<= min(p, q); this legacy simulation cell is retained without "
                "clamping either rank."
            ),
        )
    else:
        fit_started = time.perf_counter()
        fit = restricted_rrr_fn(
            X,
            Y,
            observed_source,
            target_rank=setting.target_rank,
            source_rank=setting.source_rank,
        )
        result["fit_time_sec"] = time.perf_counter() - fit_started
        candidate = fit.candidate
        diagnostics = _plain_value(candidate.metadata)
        result["fit_diagnostics"] = diagnostics
        for key in SUMMARY_DIAGNOSTIC_KEYS:
            result[key] = diagnostics.get(key)

        if fit.successful:
            coefficient = fit.coefficient
            if coefficient is None:
                raise RuntimeError(
                    "restricted_rrr reported success without a coefficient matrix."
                )
            coefficient = np.asarray(coefficient)
            if coefficient.shape != coefficient_true.shape:
                raise RuntimeError(
                    "restricted_rrr returned a coefficient with shape "
                    f"{coefficient.shape}; expected {coefficient_true.shape}."
                )
            if not np.all(np.isfinite(coefficient)):
                raise RuntimeError(
                    "restricted_rrr returned a non-finite coefficient matrix."
                )

            result.update(
                C_hat=coefficient,
                avg_err=float(
                    np.linalg.norm(coefficient - coefficient_true, ord="fro")
                    / np.sqrt(setting.p * setting.q)
                ),
                status="successful",
                success=True,
            )
        else:
            failure_reason = candidate.failure_reason
            result.update(
                status="failed",
                failure_reason=(
                    None if failure_reason is None else _plain_value(failure_reason)
                ),
                failure_message=candidate.message,
            )

    result["elapsed_time_sec"] = time.perf_counter() - started
    _atomic_pickle_dump(result, destination)
    return "written", result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", type=int, choices=range(3))
    parser.add_argument("exp_id", type=int, choices=range(4))
    parser.add_argument("rd_seed_id", type=int, choices=range(100))
    parser.add_argument(
        "--setting-index",
        type=int,
        help="Run only this zero-based position in the selected experiment grid.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Result directory root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing result for the same simulation cell.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected settings and output paths without generating data.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = experiment_settings(args.model_id, args.exp_id)
    if args.setting_index is not None:
        if not 0 <= args.setting_index < len(settings):
            raise SystemExit(
                f"--setting-index must be between 0 and {len(settings) - 1}."
            )
        indexed_settings = ((args.setting_index, settings[args.setting_index]),)
    else:
        indexed_settings = tuple(enumerate(settings))

    seeds = load_experiment_seeds()
    if args.rd_seed_id >= len(seeds):
        raise SystemExit(
            f"rd_seed_id={args.rd_seed_id} is unavailable; seed file has "
            f"{len(seeds)} entries."
        )
    random_seed = int(seeds[args.rd_seed_id])
    model = MODEL_NAMES[args.model_id]
    experiment = EXPERIMENT_NAMES[args.exp_id]

    print(
        f"Restricted RRR: model={model}, experiment={experiment}, "
        f"rd_seed_id={args.rd_seed_id}, random_seed={random_seed}"
    )
    for setting_index, setting in indexed_settings:
        destination = result_path(
            args.output_root,
            model=model,
            experiment=experiment,
            setting=setting,
            seed_id=args.rd_seed_id,
        )
        applicability = setting.inapplicability_reason() or "applicable"
        print(
            f"[{setting_index}] {setting.suffix}: n={setting.n}, p={setting.p}, "
            f"q={setting.q}, sigma0={setting.sigma0}, r={setting.target_rank}, "
            f"r_s={setting.source_rank} ({applicability}) -> {destination}"
        )
        if args.dry_run:
            continue

        outcome, result = run_setting(
            setting=setting,
            model=model,
            experiment=experiment,
            seed_id=args.rd_seed_id,
            random_seed=random_seed,
            destination=destination,
            force=args.force,
        )
        if outcome == "skipped":
            print("    skipped (result exists; pass --force to replace it)")
        else:
            assert result is not None
            error = result["avg_err"]
            error_text = (
                f", avg_err={error:.6g}"
                if error is not None and np.isfinite(error)
                else ""
            )
            print(f"    wrote status={result['status']}{error_text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
