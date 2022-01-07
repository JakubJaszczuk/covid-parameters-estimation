import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray, x0=0.0, a=1.0):
    return 1 / (1 + np.exp(-a * (x - x0)))


def mse(x, y):
    return np.mean(np.square(x - y))


def mae(x, y):
    return np.mean(np.abs(x - y))


def medae(x, y):
    return np.median(np.abs(x - y))


def msle(x, y):
    return np.mean(np.square(np.log1p(x) - np.log1p(y)))


def normalize(x):
    ''' Divides by length '''
    #return x / np.linalg.norm(x)
    n = np.linalg.norm(x)
    if n == 0:
        return x
    else:
        return x / n


def smooth(x, t=7):
    return pd.Series(x).rolling(t, 1, center=True).mean().to_numpy()
