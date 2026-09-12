"""Explicit practical settings for the V2 procedure; no theorem calibration."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from sparse_smart.calibration import CalibrationError, Margins, penalty_pair


def _real(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise CalibrationError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise CalibrationError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise CalibrationError(f"{name} must be finite and {qualifier}")
    return result


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise CalibrationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _integer_pair(values, name):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise CalibrationError(f"{name} must contain two integers")
    return tuple(_integer(value, f"{name}[{side}]") for side, value in enumerate(values))


@dataclass(frozen=True)
class PracticalCalibration:
    """User-supplied numerical settings, without a statistical certificate.

    Support limits count entries outside the selected free source rows.
    Zero initialization penalty requests minimum-norm reduced least squares
    followed by coefficient-SVD truncation; it does not change the estimator.
    """

    init_penalty: float
    penalty: float | tuple[float, float]
    step_size_inverse: float
    support_limits: tuple[int, int]

    def __post_init__(self):
        object.__setattr__(self, "init_penalty", _real(self.init_penalty, "init_penalty"))
        penalties = penalty_pair(self.penalty)
        object.__setattr__(self, "penalty", penalties if isinstance(self.penalty, (tuple, list)) else penalties[0])
        object.__setattr__(self, "step_size_inverse", _real(self.step_size_inverse, "step_size_inverse", positive=True))
        object.__setattr__(self, "support_limits", _integer_pair(self.support_limits, "support_limits"))

    @property
    def penalties(self):
        return penalty_pair(self.penalty)


__all__ = ["CalibrationError", "Margins", "PracticalCalibration"]
