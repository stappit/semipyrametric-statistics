import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import scipy as sp
import numpy as np

import src.demmler_reinsch as dr


class LogLikelihood(object):
    """
    A log-likelihood class for estimating the variance parameter of the mixed
    model y = Xβ + Zu + ε, where X  is the fixed component, Z the random
    component, and ε the error term.

    It is assumed that V = Z G Z.T + R, where R is a positive multiple of the
    identity representing the variance of fixed effects, and G is a positive
    diagonal matrix representing the variance of the random effects.

    y = Xβ + Zu + ε

    V := Cov(y)

    G := Cov(u) (assumed to be diagonal)
        (q, q) 2D array

    R := Cov(ε) (assumed to be σ^2 I)
        (n, n) 2D array

    V = Z G Z.T + R
    """

    restricted = True

    def __init__(self, y, X, Z, blocks=None):
        """
        Parameters
        ----------

        y : (n, ) array
        X : (n, p) 2D array
        Z : (n, q) 2D array
        blocks : list of ints
            each int represents the block size of G
            sum(blocks) == q
        """

        self.y = y
        self.X = X
        self.Z = Z

        self.n, self.p = X.shape
        logger.debug('X is {} x {}'.format(*X.shape))
        logger.debug('Z is {} x {}'.format(*Z.shape))
        logger.debug('y is {} x 1'.format(*y.shape))
        _, self.q = Z.shape

        self.blocks = blocks or [Z.shape[1]]
        logger.debug('{} blocks sum up to {}'.format(len(self.blocks),
                                                     sum(self.blocks)))

        self._In = np.eye(self.n)
        self._Iq = np.eye(self.q)

        self._A, self._b, self._s = dr.orthogonalisation(y, Z, np.eye(self.q))
        logger.debug('A is {} x {}'.format(*self._A.shape))
        logger.debug('b is {} x {}'.format(*self._b.shape))
        logger.debug('s is {} x {}'.format(*self._s.shape))

        self._λ = None

    def _make_params(self, λ):
        """
        Return the array of diagonal elements of G.

        Parameters
        ----------

        λ : iterable of unique parameters of G
        """

        return np.expand_dims(np.hstack(
            (np.full(b, l) for b, l in zip(self.blocks, λ))
        ), 1)

    def _psi(self, Λ):
        d = (1 / (1 + Λ * self._s))
        logger.debug(d.shape)
        return self._In - (self._A @ sp.linalg.block_diag(*d) @ self._A.T)

    def _beta(self, P):
        return np.linalg.inv(self.X.T @ P @ self.X) @ self.X.T @ P @ self.y

    def _var(self, r, P):
        return r.T @ P @ r / (self.n - self.p)

    def _det1(self, Λ):
        Λinv = sp.linalg.block_diag(*(1 / Λ))
        return np.log(np.linalg.det(self._Iq + self.Z.T @ self.Z @ Λinv))

    def _det2(self, P):
        return np.log(np.linalg.det(self.X.T @ P @ self.X))

    def log_likelihood(self, λ, restricted=restricted):
        """
        Return the log-likelihood evaluated at λ.
        """

        logger.debug(λ.shape)

        Λ = self._make_params(λ)
        P = self._psi(Λ)
        β = self._beta(P)
        r = self.y - self.X @ β
        v = self._var(r, P)
        d1 = self._det1(Λ)

        if restricted:
            d2 = self._det2(P)
        else:
            d2 = 0

        logger.debug(str(λ))

        n, p = self.n, self.p

        return (-1 / 2) * (
            (n - p) * np.log(v) + r.T @ P @ r / v + d1 + d2
        ) - n * np.log(2 * np.pi) / 2

    def __call__(self, *args, **kwargs):
        return self.log_likelihood(*args, **kwargs)

    def neg_log_likelihood(self, *args, **kwargs):
        """
        Return the negative log-likelihood evaluated at λ.
        """
        return -self.log_likelihood(*args, **kwargs)

    def _maximise(self, restricted=restricted, bounds=None):
        """
        Return the parameters optimising the variance V.
        """

        bounds = bounds or [(0.00001, None) for _ in self.blocks]
        x0 = self._λ or tuple(1 for _ in self.blocks)
        res = sp.optimize.minimize(
            self.neg_log_likelihood,
            x0,
            bounds=bounds,
            args=(restricted,)
        )
        self._λ = res.x
        return res.x

    def maximise(self, *args, **kwargs):
        """
        Return the parameters optimising the variance V.
        """

        for _ in range(3):
            x = self._maximise(*args, **kwargs)

        return x

    @property
    def g(self):
        pass

    @property
    def r(self):
        pass

    @property
    def v(self):
        return self.Z @ self.g @ self.Z.T + self.r

    def variance(self):
        """
        Return the variance of the fixed and random components.
        """

        Λ = self._make_params(self._λ)
        P = self._psi(Λ)
        β = self._beta(P)
        r = self.y - self.X @ β

        v_f = self._var(r, P)
        v_r = v_f / self._λ

        return (v_f, self._make_params(v_r))

    def covariance(self):
        pass
