"""Stable pairwise validation ranking and fixed-reference score reporting."""

import numpy as np


SELECTION_RULE = "pairwise-validation-loss-v1"


def validation_loss_difference(prediction, incumbent, response):
    """Return mean loss(prediction)-loss(incumbent) without common losses.

    Center each residual at its own prediction. A fixed, distant reference
    can lose differences between two good fits even when its accumulated
    score uses extended precision. The pairwise product keeps those local
    differences visible and removes exactly unchanged prediction entries.
    """
    prediction, incumbent, response = (np.asarray(a, dtype=np.longdouble)
                                       for a in (prediction, incumbent, response))
    if (prediction.shape != response.shape or incumbent.shape != response.shape
            or not prediction.size or not all(np.isfinite(a).all()
                                              for a in (prediction, incumbent, response))):
        raise ValueError("validation predictions and responses must be matching finite arrays")
    delta = prediction - incumbent
    value = float(np.sum(delta * ((prediction-response) + (incumbent-response)),
                         dtype=np.longdouble) / prediction.size)
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite pairwise validation loss difference")
    return value


class ValidationReference:
    """One fit-scoped reference shared by every candidate and checkpoint.

    Absolute MSE and this fixed-reference score are reporting quantities.
    The latter uses
    ``mean(2 * (reference - response) * delta + delta**2)`` with
    ``delta = prediction - reference``, avoiding a large response-only
    constant. The reference does not enter training or line-search updates.
    Selection itself compares each prediction directly with its incumbent
    using ``validation_loss_difference``; rounded reporting-score ties do not
    prevent a later improvement from winning.
    """

    def __init__(self):
        self.prediction = None
        self.residual = None
        self.metadata = None

    def score(self, prediction, response, *, metadata=None):
        prediction = np.asarray(prediction, dtype=float)
        response = np.asarray(response, dtype=float)
        if (prediction.shape != response.shape or prediction.size == 0
                or not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(response))):
            raise ValueError("validation predictions and responses must be matching finite arrays")
        if self.prediction is None:
            self.prediction = prediction.copy()
            self.residual = prediction - response
            self.metadata = dict(kind="first_evaluated_initializer", iteration=0)
            self.metadata.update(metadata or {})
            self.prediction.setflags(write=False)
            self.residual.setflags(write=False)
            return 0.
        if prediction.shape != self.prediction.shape:
            raise ValueError("validation reference has a different prediction shape")
        # Cast before forming the inner products, as well as accumulating in
        # extended precision where the platform provides it.
        delta = prediction.astype(np.longdouble) - self.prediction
        residual = self.residual.astype(np.longdouble)
        value = float((2 * np.sum(residual * delta, dtype=np.longdouble)
                       + np.sum(delta * delta, dtype=np.longdouble)) / prediction.size)
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite validation selection score")
        return value
