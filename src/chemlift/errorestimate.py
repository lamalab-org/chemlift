import numpy as np
from gptchem.evaluator import get_regression_metrics


def estimate_rounding_error(y, num_digit):
    """
    Estimates the regression performance metrics (minimal error)
    due to rounding of the target values.
    """
    rounded = np.round(y, num_digit)
    return get_regression_metrics(y, rounded)
    