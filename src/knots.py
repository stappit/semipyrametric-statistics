#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


import numpy as np


def default_number_of_knots(x):
    return min(len(np.unique(x)) // 4, 35)


def add_boundary_knots(ks, p):
    b0 = np.asarray([ks[0] for _ in range(p)])
    b1 = np.asarray([ks[-1] for _ in range(p)])
    return np.hstack((b0, ks, b1))


def make_knots(x, k, boundary=0):
    pcts = [100 * (n + 2) / (k + 2) for n in range(k - 2 * boundary)]
    knots = np.percentile(x, pcts)
    if boundary > 0:
        knots = add_boundary_knots(knots, boundary)
    return knots
