"""Sigmoid curve fitting utilities for neurometric function analysis.

Provides a constrained sigmoid model and multi-start fitting routine used
to characterize whether neural population responses track stimulus continua
in a categorical (small k) or graded (large k) fashion.
"""

import numpy as np
from scipy.optimize import curve_fit


def sigmoid_model(x, a, x0, k):
    """Sigmoid constrained so f(x0) = 0.5 (x0 is the true PSE).

    Parameters
    ----------
    x : array-like
        Stimulus values (e.g., morph steps 1-6).
    a : float
        Amplitude — controls the range of the sigmoid.
    x0 : float
        Point of subjective equality — morph step where output crosses 0.5.
    k : float
        Steepness parameter. Small k (→0) = categorical/step-function,
        large k (→∞) = graded/linear. k > 10 is flagged as effectively linear.
    """
    return a / (1.0 + np.exp(-(x - x0) / k)) + (0.5 - a / 2.0)


def fit_model(func, x, y, p0_list, bounds, maxfev=5000):
    """Try multiple initial guesses, return (best_params, best_rss)."""
    best_rss = np.inf
    best_popt = None
    for p0 in p0_list:
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
            rss = float(np.sum((y - func(x, *popt)) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_popt = popt
        except Exception:
            pass
    return best_popt, best_rss


# Default initial guesses and bounds for sigmoid_model
SIGMOID_P0_LIST = [
    [1.0, 3.5, 0.5],
    [-1.0, 3.5, 0.5],
    [1.0, 3.5, 1.5],
    [-1.0, 3.5, 1.5],
    [1.0, 3.5, 5.0],
    [-1.0, 3.5, 5.0],
]
SIGMOID_BOUNDS = ([-3, 0.5, 0.05], [3, 6.5, 1000.0])

# Threshold above which k is considered "effectively linear"
EFFECTIVELY_LINEAR_K = 10.0


def fit_sigmoid(x, y):
    """Convenience wrapper: fit sigmoid_model and return a result dict.

    Parameters
    ----------
    x, y : array-like
        Stimulus values and corresponding responses.

    Returns
    -------
    dict or None
        Keys: ``params`` (a, x0, k), ``rss``, ``r2``,
        ``effectively_linear`` (bool). Returns None if fitting fails entirely.
    """
    popt, rss = fit_model(sigmoid_model, x, y, SIGMOID_P0_LIST, SIGMOID_BOUNDS)
    if popt is None:
        return None
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
    a, x0, k = popt
    return {
        "params": popt,
        "a": a,
        "x0": x0,
        "k": k,
        "rss": rss,
        "r2": r2,
        "effectively_linear": k > EFFECTIVELY_LINEAR_K,
    }
