"""Sigmoid curve fitting utilities for neurometric function analysis.

Provides sigmoid models and multi-start fitting routines used to characterize
whether neural population responses track stimulus continua in a categorical
(small k) or graded (large k) fashion.

Two models are available:

- ``sigmoid_model`` (3-param): free amplitude *a*, PSE *x0*, steepness *k*.
  Use when data is NOT normalized to endpoint means (e.g., behavioral
  psychometric functions where lapsing can compress the range).

- ``sigmoid_model_2p`` (2-param): PSE *x0* and steepness *k* only.
  Use when data IS normalized so that endpoint means map to 0 and 1.
  This is the default for neurometric fits via ``fit_sigmoid``.
"""

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# 3-parameter sigmoid (legacy, used for behavioral fits)
# ---------------------------------------------------------------------------

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
        large k (→∞) = graded/linear.
    """
    return a / (1.0 + np.exp(-(x - x0) / k)) + (0.5 - a / 2.0)


SIGMOID_P0_LIST = [
    [1.0, 3.5, 0.5],
    [-1.0, 3.5, 0.5],
    [1.0, 3.5, 1.5],
    [-1.0, 3.5, 1.5],
    [1.0, 3.5, 5.0],
    [-1.0, 3.5, 5.0],
]
SIGMOID_BOUNDS = ([-3, 0.5, 0.05], [3, 6.5, 10.0])


# ---------------------------------------------------------------------------
# 2-parameter sigmoid (for normalized neurometric data)
# ---------------------------------------------------------------------------

def sigmoid_model_2p(x, x0, k):
    """Standard logistic: f(x) = 1 / (1 + exp(-(x - x0) / k)).

    Use on data normalized so that endpoint means are 0 (step 1) and 1 (step 6).
    f(x0) = 0.5 by construction, so x0 is the PSE.

    Parameters
    ----------
    x : array-like
        Stimulus values (e.g., morph steps 1-6).
    x0 : float
        Point of subjective equality.
    k : float
        Steepness. Small k = categorical, large k = graded.
    """
    return 1.0 / (1.0 + np.exp(-(x - x0) / k))


SIGMOID_2P_P0_LIST = [
    [3.5, 0.5],
    [3.5, 1.5],
    [3.5, 5.0],
]
SIGMOID_2P_BOUNDS = ([0.5, 0.05], [6.5, 10.0])


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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


# Threshold above which k is considered "effectively linear".
# For a 6-step continuum (x ∈ [1,6]), k=1 already produces a nearly
# linear curve because the sigmoid argument (x-x0)/k spans only ~±2.5.
EFFECTIVELY_LINEAR_K = 2.0


def fit_sigmoid(x, y):
    """Fit 2-parameter sigmoid to normalized neurometric data.

    Input data must be normalized so that endpoint means are near 0 and 1.
    An assertion checks this precondition.

    Parameters
    ----------
    x, y : array-like
        Stimulus values and corresponding normalized responses.

    Returns
    -------
    dict or None
        Keys: ``params`` (x0, k), ``x0``, ``k``, ``rss``, ``r2``,
        ``effectively_linear`` (bool). Returns None if fitting fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Verify data is normalized to endpoint means
    y_at_1 = y[x == 1]
    y_at_6 = y[x == 6]
    if len(y_at_1) > 0 and len(y_at_6) > 0:
        mean_at_1 = float(np.mean(y_at_1))
        mean_at_6 = float(np.mean(y_at_6))
        assert abs(mean_at_1) < 0.3, (
            f"fit_sigmoid expects normalized data (step-1 mean near 0), got {mean_at_1:.3f}"
        )
        assert abs(mean_at_6 - 1.0) < 0.3, (
            f"fit_sigmoid expects normalized data (step-6 mean near 1), got {mean_at_6:.3f}"
        )

    popt, rss = fit_model(sigmoid_model_2p, x, y, SIGMOID_2P_P0_LIST, SIGMOID_2P_BOUNDS)
    if popt is None:
        return None
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
    x0, k = popt
    return {
        "params": popt,
        "x0": x0,
        "k": k,
        "rss": rss,
        "r2": r2,
        "effectively_linear": k > EFFECTIVELY_LINEAR_K,
    }
