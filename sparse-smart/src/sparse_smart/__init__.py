"""Sparse two-stage SMART regression in fixed source coordinates."""
from .calibration import (CalibrationError, Margins, PracticalCalibration,
                          PrescribedCalibration, ResolvedCalibration,
                          SourceAccuracyError, resolve_calibration)
from .chart import AnchorChart
from .estimator import FitFailure, SparseSMART
from .initialization import LassoInitialization, reduced_lasso
from .solver import IterationRecord, RefinementResult, refine
from .anchor_solver import refine_anchor_projected
from .source import ExactSource, NoisySource, SourceBases, prepare_source
from .spectral import project_singular_values
from .tuning import SparseSMARTTuner

__version__ = "0.3.0"
__all__ = ["SparseSMART", "ExactSource", "NoisySource", "Margins", "PracticalCalibration",
           "PrescribedCalibration", "ResolvedCalibration", "resolve_calibration", "AnchorChart",
           "SourceBases", "prepare_source", "LassoInitialization", "reduced_lasso", "refine",
           "IterationRecord", "RefinementResult", "FitFailure", "CalibrationError", "SourceAccuracyError",
           "SparseSMARTTuner", "project_singular_values", "refine_anchor_projected"]
