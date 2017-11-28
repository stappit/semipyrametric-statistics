#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
logger = logging.getLogger(__name__)


import numpy as np
import scipy as sp
from scipy import stats
from scipy.interpolate import BSpline

from .linear_model_selection import rss


def orthogonalisation(y, C, D, c=10**-10):
    """
    Return (A, b, s) where

    * C.T C = R.T R, Cholesky decomposition
    * R^-T D R^-1 = U diag(s) U.T, SVD decomposition
    * A = C R^-1 U
    * b = A.T y

    The purpose is to find ŷ = C(C.T C + αD)^-1 C.T y = A diag(1 + αs)^-1 b.

    C :: (n, p) array, the design matrix
    D :: (p, p) array, symmetric penalisation matrix
    y :: (n, 1) array, response vector
    c :: float in [0, ∞)

    A :: (n, p) array
    b :: (p, 1) array
    s :: (p, 1) array
    """

    R = sp.linalg.cho_factor((C.T @ C) + (c * D), lower=False)[0]
    Rinv = sp.linalg.solve_triangular(R, np.eye(*R.shape), lower=False)
    U, s, _ = sp.linalg.svd(Rinv.T @ D @ Rinv)
    s = s.reshape(s.shape[0], 1)
    A = C @ Rinv @ U
    b = (A.T @ y).reshape(A.shape[1], 1)
    return A, b, s


def predict(a, A, b, s):
    return np.ravel(A @ (b / (np.ones_like(b) + a * s)))


def hat(α, A, b, s):
    #A diag(1 + αs)^-1 A.T
    return A @ sp.linalg.block_diag(*(1 / (1 + α * s))) @ A.T


def df_fit(a, s):
    ones = np.ones_like(s)
    return (ones.T @ (ones / (ones + a * s)))[0, 0]


def df_res(a, s, n):
    ones = np.ones_like(s)
    c = ones / (ones + a * s)
    return n - 2 * ones.T @ c + (c**2).sum()


def rss_approx(y, a, A, b, s):
    ones = np.ones_like(b)
    c = b / (ones + a * s)
    return y.T @ y - 2 * b.T @ c + c.T @ A.T @ A @ c


def error_variance(y, a, A, b, s, fast_rss=False):
    if fast_rss:
        return rss_approx(y, a, A, b, s) / df_res(a, s, len(y))
    else:
        return rss(y, predict(a, A, b, s)) / df_res(a, s, len(y))


def _residual_variance_component(a, A, b, s):
    return np.multiply(A @ np.diag((1 / (1 + a * s)).ravel()), A).sum(axis=1)


def residual_variance(y, a, A, b, s):
    v_err = error_variance(y, a, A, b, s)
    return v_err * _residual_variance_component(a, A, b, s)


def pointwise_confidence_interval(y, a, A, b, s, sig):
    n = len(y)
    df = df_res(a, s, n)
    v_err = error_variance(y, a, A, b, s)
    se_res = np.sqrt(v_err * _residual_variance_component(a, A, b, s))
    t = sp.stats.t(1 - sig / 2, df)
    return t * se_res


def pointwise_prediction_interval(y, a, A, b, s, sig):
    n = len(y)
    df = df_res(a, s, n)
    se_err = np.sqrt(error_variance(y, a, A, b, s))
    se_res = se_err * np.sqrt(1 + _residual_variance_component(a, A, b, s))
    t = sp.stats.t(1 - sig / 2, df)
    return t * se_res


def width(y, a, A, b, s, sig, iters):
    Σ = hat(a, A, b, s)
    n = Σ.shape[0]
    dist = stats.multivariate_normal(
        mean=np.zeros((n,)),
        cov=Σ,
        allow_singular=True # why is it sometimes singular?
    )

    sims = dist.rvs(iters) / residual_variance(y, a, A, b, s)
    return np.percentile(
        np.abs(sims).max(axis=0),
        100 * (1 - sig / 2)
    )


def simultaneous_confidence_interval(y, a, A, b, s, sig=0.95, iters=10000):
    m = width(y, a, A, b, s, sig, iters)
    v_err = error_variance(y, a, A, b, s)
    se_res = np.sqrt(v_err * _residual_variance_component(a, A, b, s))
    return m * se_res


def simultaneous_prediction_interval(y, a, A, b, s, sig=0.95, iters=10000):
    m = width(y, a, A, b, s, sig, iters)
    se_err = np.sqrt(error_variance(y, a, A, b, s))
    se_res = se_err * np.sqrt(1 + _residual_variance_component(a, A, b, s))
    return m * se_res


def minimise(f, y, A, b, s, grid=None):
    """
    Return the α that minimises f.

    The (A, b, s) are the output of Demmler-Reinsch orthogonalisation.
    """

    def _f(a):
        return f(y, predict(a, A, b, s), df_fit(a, s))

    if grid:
        arg_best = min(grid)
        best = _f(arg_best)
        for a in grid:
            score = _f(a)
            if score < best:
                best = score
                arg_best = a

    else:
        x, _ = minimise(f, y, A, b, s, grid=[2 ** n for n in range(-2, 20)])
        res = sp.optimize.minimize(_f, x, bounds=[(0.001, None)])
        arg_best = res.x[0]
        best = res.fun

    return (arg_best, best)
