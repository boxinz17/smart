"""Bridge to pinned, unmodified authors' R code for Park et al.

The implemented comparison is the authors' ``cv.twostep`` with one supplied
source, called "Park all-source two-stage NR". It includes the original internal
cross-validation, centering, predictor standardization and target intercept.
It does not include FSD/MSD source selection. It consumes RAW source observations
and therefore has more source information than coefficient-only competitors.

``fit_park_validation`` is separately labeled: it preserves the same fixed-lambda
two-stage fits but selects the pair on the revision's independent target holdout.
It does not refit after selection or use that holdout for centering or scaling.

No local simulation is run on import. Invoke fitting on a Slurm compute node.
Only corpcor and jsonlite are needed in R; NumPy is needed in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np


UPSTREAM_COMMIT = "9c94ff38b71cded6e6c7a8b6db243e2c7a06b78d"
ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ROOT / "third_party" / "park_transfer_learning"
DEFAULT_LAMBDAS = (.01, .03, .1, .3, 1., 3., 10.)


class ParkFitError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class ParkFit:
    coefficient: np.ndarray
    intercept: np.ndarray
    diagnostics: dict

    def predict(self, design):
        x = _matrix(design, "design")
        if x.shape[1] != self.coefficient.shape[0]:
            raise ValueError("design has the wrong number of columns")
        return x @ self.coefficient + self.intercept


@dataclass(frozen=True)
class ParkValidationResult:
    fit: ParkFit
    candidate_results: tuple[dict, ...]
    selected_index: int
    validation_loss: float
    elapsed_seconds: float

    @property
    def coefficient(self):
        return self.fit.coefficient

    @property
    def intercept(self):
        return self.fit.intercept

    @property
    def diagnostics(self):
        return self.fit.diagnostics

    def predict(self, design):
        return self.fit.predict(design)


def _matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or min(array.shape) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return array


def _positive_grid(values, name):
    grid = tuple(float(value) for value in values)
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError(f"{name} must contain finite positive values")
    # Upstream cv.nuclear uses tapply's sorted numeric groups and then indexes
    # the original lambda sequence, so enforce sorted unique grids explicitly.
    if tuple(sorted(set(grid))) != grid:
        raise ValueError(f"{name} must be strictly increasing to match authors' CV indexing")
    return grid


def verify_upstream():
    """Verify every vendored upstream file before executing any author code."""
    manifest = json.loads((VENDOR_ROOT / "PROVENANCE.json").read_text())
    if manifest["commit"] != UPSTREAM_COMMIT:
        raise ValueError("unexpected Park upstream commit")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((VENDOR_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"vendored Park source hash mismatch: {name}")
    return manifest


def _fit_park_bridge(design, response, source_design, source_response, *,
                     lambda_w=DEFAULT_LAMBDAS, lambda_delta=DEFAULT_LAMBDAS,
                     eta=1., max_iter=1000, tolerance=1e-6, nfold=5,
                     rscript="Rscript", timeout_seconds=1800, allow_local=False,
                     validation_design=None, validation_response=None,
                     require_convergence=True):
    started = perf_counter()
    if not allow_local and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Park experiment fits must run inside a Slurm allocation")
    x, y = _matrix(design, "design"), _matrix(response, "response")
    x0, y0 = _matrix(source_design, "source_design"), _matrix(source_response, "source_response")
    if x.shape[0] != y.shape[0] or x0.shape[0] != y0.shape[0] or x.shape[1] != x0.shape[1] or y.shape[1] != y0.shape[1]:
        raise ValueError("source and target dimensions are incompatible")
    if min(x.shape[1], y.shape[1]) < 2:
        raise ValueError("the unmodified authors' matrix code requires p,q >= 2")
    external = validation_design is not None or validation_response is not None
    if external:
        xv, yv = _matrix(validation_design, "validation_design"), _matrix(validation_response, "validation_response")
        if xv.shape[0] != yv.shape[0] or xv.shape[1] != x.shape[1] or yv.shape[1] != y.shape[1]:
            raise ValueError("validation dimensions are incompatible")
    if not external and (isinstance(nfold, bool) or not isinstance(nfold, int) or not 2 <= nfold <= min(len(x), len(x0)) // 2):
        raise ValueError("nfold must be an integer leaving at least two observations per fold")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite([eta, tolerance, timeout_seconds]).all() or min(eta, tolerance, timeout_seconds) <= 0:
        raise ValueError("eta, tolerance and timeout_seconds must be finite and positive")
    lambda_w, lambda_delta = _positive_grid(lambda_w, "lambda_w"), _positive_grid(lambda_delta, "lambda_delta")
    provenance = verify_upstream()
    options = dict(lambda_w=lambda_w, lambda_delta=lambda_delta, eta=float(eta),
                   max_iter=max_iter, tolerance=float(tolerance), nfold=None if external else nfold,
                   mode="external_validation" if external else "authors_internal_cv",
                   require_convergence=bool(require_convergence))
    diagnostics = {
        "method": "park_all_source_two_stage_nr", "upstream_function": "cv.twostep",
        "upstream_commit": UPSTREAM_COMMIT, "upstream_repository": provenance["repository"],
        "source_information": "raw_source_design_and_responses",
        "source_selection": "one_supplied_source_always_included_no_FSD_or_MSD",
        "tuning": "unmodified_authors_internal_cross_validation",
        "options": options, "n_train": len(x), "n_source": len(x0),
        "n_predictors": x.shape[1], "n_responses": y.shape[1],
    }
    if external:
        diagnostics.update(method="park_all_source_two_stage_nr_external_validation",
                           upstream_function="ADMM.nuclear_with_authors_cv_twostep_final_fit_transformations",
                           tuning="revision_independent_target_holdout_pair_selection",
                           n_validation=len(xv), refit=False,
                           departure_from_authors="external_validation_replaces_internal_CV_only")
    with tempfile.TemporaryDirectory(prefix="smart-park-") as temporary:
        directory = Path(temporary)
        for name, matrix in (("X", x), ("Y", y), ("X0", x0), ("Y0", y0)):
            np.savetxt(directory / f"{name}.csv", matrix, delimiter=",", fmt="%.17g")
        if external:
            np.savetxt(directory / "Xv.csv", xv, delimiter=",", fmt="%.17g")
            np.savetxt(directory / "Yv.csv", yv, delimiter=",", fmt="%.17g")
        (directory / "options.json").write_text(json.dumps(options))
        command = [str(rscript), str(ROOT / "published_park.R"), str(directory), str(VENDOR_ROOT)]
        try:
            completed = subprocess.run(command, check=False, capture_output=True,
                                       text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            diagnostics.update(status="timeout", elapsed_seconds=perf_counter() - started,
                               timeout_seconds=timeout_seconds)
            raise ParkFitError("authors' Park fit exceeded the declared timeout", diagnostics) from error
        except OSError as error:
            diagnostics.update(status="runtime_unavailable", error=str(error))
            raise ParkFitError("could not start Rscript for authors' Park code", diagnostics) from error
        diagnostics.update(returncode=completed.returncode, stdout=completed.stdout[-4000:], stderr=completed.stderr[-4000:])
        result_path = directory / "diagnostics.json"
        if result_path.exists():
            diagnostics.update(json.loads(result_path.read_text()))
        if completed.returncode != 0:
            diagnostics.update(status="failed", elapsed_seconds=perf_counter() - started)
            raise ParkFitError("authors' Park code failed; inspect diagnostics", diagnostics)
        coefficient = np.loadtxt(directory / "coefficient.csv", delimiter=",", ndmin=2)
        intercept = np.loadtxt(directory / "intercept.csv", delimiter=",", ndmin=1).reshape(-1)
    if coefficient.shape != (x.shape[1], y.shape[1]) or intercept.shape != (y.shape[1],) or not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise ParkFitError("invalid Park coefficient/intercept output", diagnostics)
    coefficient.flags.writeable = False
    intercept.flags.writeable = False
    diagnostics.update(status="ok", elapsed_seconds=perf_counter() - started,
                       training_loss=float(np.sum((y - x @ coefficient - intercept)**2) / (2 * len(x))),
                       effective_rank=int(np.linalg.matrix_rank(coefficient)),
                       upstream_source_sha256=provenance["files"]["Function/Functions_naiveapproaches.R"])
    return ParkFit(coefficient, intercept, diagnostics)


def fit_park(design, response, source_design, source_response, *,
             lambda_w=DEFAULT_LAMBDAS, lambda_delta=DEFAULT_LAMBDAS,
             eta=1., max_iter=1000, tolerance=1e-6, nfold=5,
             rscript="Rscript", timeout_seconds=1800, allow_local=False):
    """Run exact authors' all-source two-stage NR with one source.

    All supplied target rows enter the authors' original internal CV. Evaluation
    rows must be withheld. Only diagnostic observations are added to the solver.
    An explicit Slurm environment is required; allow_local is for unit fixtures.
    """
    return _fit_park_bridge(design, response, source_design, source_response,
                            lambda_w=lambda_w, lambda_delta=lambda_delta, eta=eta,
                            max_iter=max_iter, tolerance=tolerance, nfold=nfold,
                            rscript=rscript, timeout_seconds=timeout_seconds,
                            allow_local=allow_local)


def fit_park_validation(design, response, design_validation, response_validation,
                        source_design, source_response, *,
                        lambda_w=DEFAULT_LAMBDAS, lambda_delta=DEFAULT_LAMBDAS,
                        eta=1., max_iter=1000, tolerance=1e-6,
                        rscript="Rscript", timeout_seconds=1800,
                        require_convergence=True, allow_local=False):
    """Select fixed-lambda authors' two-stage fits on a target validation set.

    Each pooled nuclear fit is reused across correction penalties. For any fixed
    pair, the coefficient matches the final fit from authors' cv.twostep with
    singleton grids. Only lambda selection changes. Validation responses do not
    enter fitting, centering or scaling; there is no refit after selection.
    Ties choose the first pair in increasing lambda_w, then lambda_delta order.
    Capped fits remain visible and are excluded by default. This is not FSD/MSD.
    """
    fit = _fit_park_bridge(design, response, source_design, source_response,
                           lambda_w=lambda_w, lambda_delta=lambda_delta, eta=eta,
                           max_iter=max_iter, tolerance=tolerance, rscript=rscript,
                           timeout_seconds=timeout_seconds, allow_local=allow_local,
                           validation_design=design_validation,
                           validation_response=response_validation,
                           require_convergence=require_convergence)
    return ParkValidationResult(fit, tuple(fit.diagnostics["candidate_results"]),
                                 int(fit.diagnostics["selected_index"]),
                                 float(fit.diagnostics["validation_loss"]),
                                 float(fit.diagnostics["elapsed_seconds"]))


def _main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity", action="store_true", help="small algebraic author-code check inside Slurm")
    parser.add_argument("--sanity-validation", action="store_true", help="compare fixed-lambda fits with authors' singleton CV on Slurm")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.sanity and not args.sanity_validation:
        parser.error("use --sanity or --sanity-validation for the Slurm author-code check")
    # A deterministic no-noise fixture; this checks the bridge and source-code
    # convergence diagnostics, not statistical superiority.
    x = np.array([[1., 0., 1.], [-1., 0., -1.], [0., 1., 1.], [0., -1., -1.],
                  [1., 1., 0.], [-1., -1., 0.], [1., -1., 1.], [-1., 1., -1.]])
    coefficient = np.array([[1., .2], [0., .5], [.3, -.1]])
    y = x @ coefficient + np.array([.4, -.2])
    if args.sanity_validation:
        # Nonzero target/source means test the original separate-domain
        # centering, two different standardizations and final target intercept.
        x = x + np.array([.1, .3, -.2])
        y = x @ coefficient + np.array([.4, -.2])
        x0 = np.vstack((x, x)) + np.array([.2, -.3, .1])
        y0 = x0 @ (coefficient * 1.4) + np.array([-.1, .5])
        options = dict(lambda_w=(.01,), lambda_delta=(.03,), nfold=2,
                       max_iter=2000, tolerance=1e-10)
        internal = fit_park(x, y, x0, y0, **options)
        options.pop("nfold")
        external = fit_park_validation(x, y, x, y, x0, y0, **options)
        coefficient_difference = float(np.max(np.abs(internal.coefficient - external.coefficient)))
        intercept_difference = float(np.max(np.abs(internal.intercept - external.intercept)))
        report = {
            "status": "ok" if max(coefficient_difference, intercept_difference) <= 1e-10 else "failed",
            "coefficient_max_absolute_difference": coefficient_difference,
            "intercept_max_absolute_difference": intercept_difference,
            "internal_diagnostics": internal.diagnostics,
            "external_diagnostics": external.diagnostics,
            "scope": "fixed_lambda_equivalence_to_authors_singleton_CV_with_nonzero_domain_means",
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if report["status"] != "ok":
            raise SystemExit("Park external-validation singleton equivalence failed")
        return
    result = fit_park(x, y, np.vstack((x, x)), np.vstack((y, y)),
                      lambda_w=(.001, .01), lambda_delta=(.001, .01), nfold=2,
                      max_iter=1000, tolerance=1e-8)
    report = dict(result.diagnostics, coefficient_error=float(np.linalg.norm(result.coefficient - coefficient)),
                  intercept_error=float(np.linalg.norm(result.intercept - np.array([.4, -.2]))))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_admm_stopping_criteria_met"] or report["coefficient_error"] > .1:
        raise SystemExit("Park sanity fixture failed its declared tolerances")


if __name__ == "__main__":
    _main()


__all__ = ["ParkFit", "ParkValidationResult", "ParkFitError", "fit_park",
           "fit_park_validation", "verify_upstream", "UPSTREAM_COMMIT"]
