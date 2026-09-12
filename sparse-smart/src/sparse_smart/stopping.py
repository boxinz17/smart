"""Validation patience in accepted-iteration units, separate from convergence."""
from __future__ import annotations

from dataclasses import dataclass
import numbers
import numpy as np


@dataclass(frozen=True)
class ValidationStopRequest:
    """Explicit callback return contract; ordinary callback values are ignored."""

    message: str = "Validation improvement did not meet the configured patience rule."


class ValidationStopping:
    """Track raw MSE without using it for gradient or line-search decisions.

    Small decreases accumulate against the last significant reference. The
    true observed minimum is tracked separately and never rounded to that
    reference. Patience counts accepted iterations, not validation checks.
    """

    def __init__(self, patience=None, min_iterations=500, min_relative_improvement=.001):
        for name, value, minimum in (("validation_patience", patience, 1),
                                     ("validation_min_iterations", min_iterations, 0)):
            if name == "validation_patience" and value is None:
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                    or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}" +
                                 (" or None" if name == "validation_patience" else ""))
        value = min_relative_improvement
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value < 0 or value >= 1):
            raise ValueError("validation_min_relative_improvement must be finite in [0, 1)")
        self.patience = None if patience is None else int(patience)
        self.min_iterations = int(min_iterations)
        self.min_relative_improvement = float(min_relative_improvement)
        self.checks = 0
        self.best_loss = self.best_iteration = None
        self.reference_loss = self.last_significant_iteration = None
        self.last_check_iteration = None
        self.stopped = False
        self.stop_iteration = None

    @property
    def enabled(self):
        return self.patience is not None

    def observe(self, iteration, loss, *, allow_stop=True):
        if not np.isfinite(loss) or loss < 0:
            raise ValueError("validation loss must be finite and nonnegative")
        if self.last_check_iteration is not None and iteration <= self.last_check_iteration:
            raise ValueError("validation stopping checks must increase strictly")
        self.checks += 1
        self.last_check_iteration = int(iteration)
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss, self.best_iteration = float(loss), int(iteration)
        significant = (self.reference_loss is None or
            (self.best_loss < self.reference_loss and
             self.reference_loss - self.best_loss >=
             self.min_relative_improvement * self.reference_loss))
        if significant:
            self.reference_loss = self.best_loss
            self.last_significant_iteration = int(iteration)
        since = iteration - self.last_significant_iteration
        if (allow_stop and self.enabled and iteration >= self.min_iterations
                and since >= self.patience):
            self.stopped, self.stop_iteration = True, int(iteration)
        return {"significant_improvement": bool(significant),
                "reference_loss": self.reference_loss,
                "last_significant_iteration": self.last_significant_iteration,
                "iterations_since_improvement": int(since),
                "stop_requested": self.stopped}

    def snapshot(self):
        return {"enabled": self.enabled, "patience": self.patience,
                "min_iterations": self.min_iterations,
                "min_relative_improvement": self.min_relative_improvement,
                "patience_units": "accepted_iterations", "metric": "raw_validation_mse",
                "checks": self.checks, "best_loss": self.best_loss,
                "best_iteration": self.best_iteration, "reference_loss": self.reference_loss,
                "last_significant_iteration": self.last_significant_iteration,
                "last_check_iteration": self.last_check_iteration,
                "stopped": self.stopped, "stop_iteration": self.stop_iteration}
