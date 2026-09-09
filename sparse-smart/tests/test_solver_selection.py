"""Practical solver dispatch, strict compatibility, and validation isolation."""
import numpy as np
import pytest

from sparse_smart import ExactSource, Margins, PracticalCalibration, PrescribedCalibration, SparseSMART


def fitted(*, limits=(2, 2), **options):
    rng = np.random.default_rng(21)
    X = rng.normal(size=(40, 3))
    C = np.diag([2., 0., 0.])
    defaults = dict(rank=1, source_rank=3, sparsity=(1, 1),
        margins=Margins(.05, 5., .01),
        calibration=PracticalCalibration(.01, .001, 10., limits), iterations=2)
    defaults.update(options)
    return SparseSMART(**defaults).fit(X, X @ C, source=ExactSource(np.eye(3), np.eye(3)))


def test_practical_auto_uses_full_constraint_solver_with_full_supports():
    model = fitted()
    assert model.success_, model.message_
    assert model.refinement_solver_ == "anchor_projected"
    assert model.diagnostics_["stationarity_scope"] == "full_chart_constraints"
    assert model.diagnostics_["diagnostic_coordinates"] == "omega,d,H"
    assert all(b.objective <= a.objective for a, b in zip(model.history_, model.history_[1:]))


@pytest.mark.parametrize("options", [dict(limits=(0, 0)), dict(spectral_step="reject"),
    dict(refinement_solver="chart"), dict(calibration=PrescribedCalibration(.00001), iterations=0)])
def test_auto_keeps_chart_solver_for_strict_or_hard_cap_modes(options):
    model = fitted(**options)
    assert model.success_, model.message_
    assert model.refinement_solver_ == "chart"
    assert model.diagnostics_["diagnostic_coordinates"] == "omega,d,Z"


@pytest.mark.parametrize("options,match", [(dict(limits=(1, 1)), "full complement"),
    (dict(spectral_step="reject"), "spectral_step")])
def test_explicit_anchor_solver_rejects_incompatible_settings(options, match):
    with pytest.raises(ValueError, match=match):
        fitted(refinement_solver="anchor_projected", **options)
