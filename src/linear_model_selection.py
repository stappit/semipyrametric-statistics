import numpy as np


def rss(y_true, y_pred, df=None):
    return ((y_true - y_pred) ** 2).sum()


def cv(y_true, y_pred, df):
    return (((y_true - y_pred) / (1 - df)) ** 2).sum()


def gcv(y_true, y_pred, df):
    return rss(y_true, y_pred) / ((1 - (df / len(y_true))) ** 2)


def aic(y_true, y_pred, df):
    return np.log(rss(y_true, y_pred)) + 2 * df / len(y_true)


def aicc(y_true, y_pred, df):
    return np.log(rss(y_true, y_pred)) + (2 * (df + 1)) / (len(y_true) - df - 2)
