import bisect
import random


def Mean(t):
    return float(sum(t)) / len(t)


def MeanVar(t):
    mu = Mean(t)
    var = Var(t, mu)
    return mu, var


def Trim(t, p=0.01):
    n = int(p * len(t))
    t = sorted(t)[n:-n]
    return t


def Jitter(values, jitter=0.5):
    return [x + random.uniform(-jitter, jitter) for x in values]


def TrimmedMean(t, p=0.01):
    t = Trim(t, p)
    return Mean(t)


def TrimmedMeanVar(t, p=0.01):
    t = Trim(t, p)
    mu, var = MeanVar(t)
    return mu, var


def Var(t, mu=None):
    if mu is None:
        mu = Mean(t)
    # compute the squared deviations and return their mean.
    dev2 = [(x - mu) ** 2 for x in t]
    var = Mean(dev2)
    return var


def Binom(n, k, d={}):
    if k == 0:
        return 1
    if n == 0:
        return 0

    try:
        return d[n, k]
    except KeyError:
        res = Binom(n - 1, k, d) + Binom(n - 1, k - 1, d)
        d[n, k] = res
        return res


class Interpolator(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def Lookup(self, x):
        return self._Bisect(x, self.xs, self.ys)

    def Reverse(self, y):
        return self._Bisect(y, self.ys, self.xs)

    def _Bisect(self, x, xs, ys):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect(xs, x)
        frac = 1.0 * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        y = ys[i - 1] + frac * 1.0 * (ys[i] - ys[i - 1])
        return y