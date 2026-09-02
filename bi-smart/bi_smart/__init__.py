"""Public API for the block-invariant SMART research scaffold."""

from ._version import __version__
from .candidates import score_and_select_candidates, validation_loss
from .estimator import BISMART
from .initialization import (
    InitializationResult,
    JointParameterState,
    initialize_restricted_rrr,
    target_only_rrr,
)
from .refinement import (
    BISMARTDirection,
    BISMARTState,
    GaussNewtonDiagnostics,
    GaussNewtonResult,
    quotient_gauss_newton_direction,
    solve_quotient_gauss_newton,
)
from .screening import (
    block_hard_threshold,
    cell_hard_threshold,
    run_block_screen,
)
from .source_blocks import (
    build_source_block_library,
    evaluate_wedin_gate,
    gaussian_source_error_bound,
)
from .types import (
    BISMARTConfig,
    BISMARTResult,
    BlockPartition,
    Candidate,
    CandidateStatus,
    DEFAULT_PINV_RCOND,
    FailureReason,
    FoldData,
    RefinementControls,
    ScreenResult,
    SourceBlockLibrary,
    SourceDecomposition,
)

__all__ = [
    "BISMART",
    "BISMARTConfig",
    "BISMARTDirection",
    "BISMARTResult",
    "BISMARTState",
    "BlockPartition",
    "Candidate",
    "CandidateStatus",
    "DEFAULT_PINV_RCOND",
    "FailureReason",
    "FoldData",
    "GaussNewtonDiagnostics",
    "GaussNewtonResult",
    "InitializationResult",
    "JointParameterState",
    "RefinementControls",
    "ScreenResult",
    "SourceBlockLibrary",
    "SourceDecomposition",
    "__version__",
    "block_hard_threshold",
    "build_source_block_library",
    "cell_hard_threshold",
    "evaluate_wedin_gate",
    "gaussian_source_error_bound",
    "initialize_restricted_rrr",
    "quotient_gauss_newton_direction",
    "run_block_screen",
    "score_and_select_candidates",
    "solve_quotient_gauss_newton",
    "target_only_rrr",
    "validation_loss",
]
