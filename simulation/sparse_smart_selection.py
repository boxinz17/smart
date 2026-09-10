"""Stable selection metadata, with explicit support for historical MSE records."""

import math

import numpy as np

PAIRWISE_RULE = "pairwise-validation-loss-v1"


def pairwise_selection(rows):
    rules = {row.get("selection_rule") for row in rows}
    if rules - {None, PAIRWISE_RULE} or len(rules) > 1:
        raise ValueError("Mixed or unsupported validation selection rules")
    return rules == {PAIRWISE_RULE}


def replay_comparisons(rows, *, id_key, incumbent_key, comparison_key="selection_comparison",
                       prediction=None, response=None):
    """Replay ordered decisions, optionally checking their prediction evidence."""
    winner = None
    previous_id = -1
    for row in rows:
        identifier = row[id_key]
        if type(identifier) is not int or identifier <= previous_id:
            raise ValueError("Unordered or invalid validation comparison IDs")
        previous_id = identifier
        comparison = row.get(comparison_key)
        expected_id = winner[id_key] if winner is not None else None
        if (not isinstance(comparison, dict)
                or set(comparison) != {incumbent_key, "loss_difference"}
                or comparison[incumbent_key] != expected_id
                or (expected_id is not None and type(comparison[incumbent_key]) is not int)):
            raise ValueError("Validation comparison incumbent mismatch")
        difference = comparison["loss_difference"]
        if winner is None:
            if difference is not None:
                raise ValueError("First validation comparison must have null loss difference")
            winner = row
            continue
        if type(difference) not in (int, float) or not math.isfinite(difference):
            raise ValueError("Invalid pairwise validation loss difference")
        if prediction is not None:
            actual = pairwise_loss_difference(prediction(row), response, prediction(winner))
            if ((actual < 0) != (difference < 0)
                    or not math.isclose(actual, difference, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError("Pairwise validation difference disagrees with saved factors")
        if difference < 0:
            winner = row
    return winner


def history_winner(rows, *, prediction=None, response=None):
    rows = list(rows)
    if pairwise_selection(rows):
        selection_values(rows, "loss")
        return replay_comparisons(rows, id_key="iteration", incumbent_key="incumbent_iteration",
                                  prediction=prediction, response=response)
    scores = selection_keys(rows, "loss")
    return rows[min(range(len(rows)), key=scores.__getitem__)] if rows else None


def selection_value(row, loss_key="validation_mse"):
    """Read a signed ranking score; legacy records use their absolute MSE."""
    key = "selection_score" if "selection_score" in row else loss_key
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {key}")
    return float(value)


def selection_values(rows, loss_key="validation_mse"):
    flags = {"selection_score" in row for row in rows}
    if len(flags) > 1:
        raise ValueError("Mixed relative and absolute validation selection scores")
    return [selection_value(row, loss_key) for row in rows]


def selection_keys(rows, loss_key="validation_mse"):
    """Absolute MSE breaks rounded relative-score ties on small clean losses."""
    scores = selection_values(rows, loss_key)
    losses = []
    for row in rows:
        loss = row[loss_key]
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
            raise ValueError(f"Invalid {loss_key}")
        losses.append(float(loss))
    return list(zip(scores, losses))


def validation_winner(rows, *, comparison_key="selection_comparison", prediction=None, response=None):
    """Return the earliest minimum without mixing incompatible score schemes."""
    rows = list(rows)
    if pairwise_selection(rows):
        selection_values(rows)
        return replay_comparisons(rows, id_key="candidate_id", incumbent_key="incumbent_candidate_id",
                                  comparison_key=comparison_key, prediction=prediction, response=response)
    scores = selection_keys(rows)
    return rows[min(range(len(rows)), key=scores.__getitem__)] if rows else None


def selection_payload(tuner, *, include_factors=True):
    """Store one reference per fit, never one large array per checkpoint."""
    if not hasattr(tuner, "best_selection_score_"):
        return {}
    payload = dict(selection_score=tuner.best_selection_score_,
                   validation_reference_prediction=getattr(tuner, "validation_reference_prediction_", None),
                   selection_reference=getattr(tuner, "diagnostics_", {}).get("selection_reference"))
    if hasattr(tuner, "selection_rule_"):
        payload["selection_rule"] = tuner.selection_rule_
    model = getattr(tuner, "estimator_", None)
    if include_factors and getattr(tuner, "success_", False) and hasattr(model, "left_factors_"):
        payload["selected_factors"] = dict(left=model.left_factors_, singular_values=model.singular_values_,
                                          right=model.right_factors_)
    return payload


def validate_selected_score(record, winner):
    pairwise_selection([record, winner])
    if ("selection_score" in record) != ("selection_score" in winner):
        raise ValueError("Winner validation selection scheme mismatch")
    if "selection_score" in winner and not math.isclose(
            selection_value(record), selection_value(winner), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Winner validation selection score mismatch")


def prediction_selection_score(prediction, response, reference):
    """Independently rescore saved factors against the fit's fixed reference."""
    prediction, response, reference = (np.asarray(a, dtype=float) for a in (prediction, response, reference))
    if (prediction.shape != response.shape or reference.shape != response.shape
            or not prediction.size or not all(np.isfinite(a).all() for a in (prediction, response, reference))):
        raise ValueError("Invalid validation reference or prediction")
    delta = prediction.astype(np.longdouble) - reference
    residual = (reference - response).astype(np.longdouble)
    score = float((2*np.sum(residual*delta, dtype=np.longdouble)
                   + np.sum(delta*delta, dtype=np.longdouble)) / prediction.size)
    if not math.isfinite(score):
        raise ValueError("Nonfinite validation selection score")
    return score


def pairwise_loss_difference(prediction, response, incumbent):
    """Independent symmetric residual calculation for the pairwise rule."""
    prediction, response, incumbent = (np.asarray(a, dtype=np.longdouble)
                                      for a in (prediction, response, incumbent))
    if (prediction.shape != response.shape or incumbent.shape != response.shape
            or not prediction.size or not all(np.isfinite(a).all() for a in (prediction, response, incumbent))):
        raise ValueError("Invalid pairwise validation predictions")
    value = float(np.sum((prediction-incumbent)*((prediction-response)+(incumbent-response)),
                         dtype=np.longdouble) / prediction.size)
    if not math.isfinite(value):
        raise ValueError("Nonfinite pairwise validation difference")
    return value


def factor_prediction(factors, design):
    left, d, right = (np.asarray(factors[key], dtype=float)
                      for key in ("left", "singular_values", "right"))
    return ((design @ left)*d) @ right.T
