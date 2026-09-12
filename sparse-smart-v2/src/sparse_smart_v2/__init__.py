"""SparseSMART v2: hard-sparse refinement with expanded free source rows."""

from sparse_smart.chart import AnchorChart

from .calibration import CalibrationError, Margins, PracticalCalibration
from .estimator import FitFailure, SparseSMARTv2, TrajectoryCheckpoint
from .initialization import InitializationFailure, LassoInitialization, reduced_lasso
from .solver import IterationRecord, RefinementResult, refine
from .source import ExactSource, NoisySource, ObservedSource, SourceBases, prepare_source
from .support import FreeRows, choose_free_rows, threshold_state
from .tuning import SparseSMARTv2Tuner

# Familiar names are scoped to the v2 import path; the original package is
# neither patched nor replaced.
SparseSMART = SparseSMARTv2
SparseSmart_v2 = SparseSMARTv2
SparseSMARTTuner = SparseSMARTv2Tuner

__version__ = "0.1.0"

__all__ = [
    "SparseSMARTv2", "SparseSMART", "SparseSmart_v2", "SparseSMARTv2Tuner", "SparseSMARTTuner",
    "CalibrationError", "Margins", "PracticalCalibration", "ExactSource", "NoisySource", "ObservedSource",
    "SourceBases", "prepare_source", "AnchorChart", "FreeRows", "choose_free_rows",
    "threshold_state", "FitFailure", "InitializationFailure", "LassoInitialization",
    "reduced_lasso", "TrajectoryCheckpoint", "IterationRecord", "RefinementResult", "refine",
]
