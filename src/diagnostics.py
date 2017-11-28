import numpy as np

from src.linear_model_selection import rss


def error_variance(y, H):
    """
    Return the variance of the error.
    """
    return rss(y, H @ y) / (len(y) - np.trace(H))


def residual(y, H):
    """
    Return the residuals.
    """
    return (np.eye(*H.shape) - H) @ y


def residual_variance(y, H):
    """
    Return the residual variance.
    """
    return error_variance(y, H) * (1 - np.diag(H))


def deleted_residual(y, H):
    """
    Return the deleted residuals.

    The i-th deleted residual is the i-th residual where the i-th datapoint
    excluded from the fit.
    """
    return residual(y, H) / (1 - np.diag(H))


def studentised_residual(y, H):
    """
    Return the Studentised residuals.
    """
    return residual(y, H) / np.sqrt(residual_variance(y, H))


def leverage(H):
    """
    Return the leverages.
    """
    return np.diag(H)


def cook_distance(y, H):
    """
    Return the Cook distances.

    This can be thought of as the leverage of deleted residuals.

    https://en.wikipedia.org/wiki/Cook's_distance
    """
    e = studentised_residual(y, H) ** 2
    h = leverage(H)
    p = np.trace(H) # h.sum()
    return (e @ h) / (p * (1 - h))


def autocorrelation(y, H, lag=1):
    """
    Return the autocorrelation of the error with given lag.

    Σ_{i = k + 1}^n e_i e_{i - k} / Σ_1^n e_i ^2

    The variance of the autocorrelation is approximately 1 / sqrt(n), where n
    is the length of y.
    """
    idx = np.argsort(y)
    e = np.expand_dims(H @ y - y, 1)[idx]
    n = len(e)
    e0 = e[-(n - lag):]
    e1 = e[:n - lag]
    return ((e0.T @ e1) / (e.T @ e))[0, 0]
