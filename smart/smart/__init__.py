from .solver import SMARTSolver
from .rank_selector import RankSelectorRSC
from .smart import SMART
from .utils import generate_data, evaluate_model_avg_err, extract_svd_subspaces, fit_baseline
from .multitask_marginal_regression import MultitaskMarginalRegression

__all__ = [
    "SMARTSolver",
    "RankSelectorRSC",
    "SMART",
    "generate_data",
    "evaluate_model_avg_err",
    "extract_svd_subspaces",
    "fit_baseline",
    "MultitaskMarginalRegression",
]

__version__ = "0.1.0"
