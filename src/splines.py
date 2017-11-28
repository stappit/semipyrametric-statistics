#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


import numpy as np
from scipy.interpolate import BSpline


def powers(x, p):
    return np.column_stack((x ** i for i in range(p + 1)))


def truncated_powers(x, p, knots):
    knotted = np.column_stack((x for _ in range(len(knots)))) - knots
    return np.clip(knotted, 0, None) ** p


def design_power_spline(x, p, knots):
    P = powers(x, p)
    B = truncated_powers(x, p, knots)
    return np.column_stack((
        P,
        B
    ))


def bspline(x, p, knots):
    I = np.eye(len(knots))
    return BSpline(knots, I, p)(x)


def design_bspline(x, p, knots):
    P = powers(x, p)
    B = bspline(x, p, knots)
    return np.column_stack((
        P,
        B
    ))
