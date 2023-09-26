from chemlift.errorestimate import estimate_rounding_error
import numpy as np


def test_estimate_rounding_error():
    # generate normal random numbers with mean 0 and std 1
    y = np.random.normal(0, 1, 1000)
    # test if the rounding error is 0
    assert estimate_rounding_error(y, 0)["mean_absolute_error"] <= 0.4
