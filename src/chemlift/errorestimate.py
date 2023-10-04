import numpy as np
from gptchem.evaluator import get_regression_metrics


def estimate_rounding_error(y, num_digit) -> dict:
    """
    Estimates the regression performance metrics (minimal error)
    due to rounding of the target values.

    Args:
        y: The target values.
        num_digit: The number of digits to round to.

    Returns:
        A dictionary containing the regression performance metrics.
    """
    rounded = np.round(y, num_digit)
    return get_regression_metrics(y, rounded)
