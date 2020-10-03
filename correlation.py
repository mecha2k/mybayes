from __future__ import print_function

import math
import random

import mystats


def Cov(xs, ys, mux=None, muy=None):
    if mux is None:
        mux = mystats.Mean(xs)
    if muy is None:
        muy = mystats.Mean(ys)

    total = 0.0
    for x, y in zip(xs, ys):
        total += (x - mux) * (y - muy)

    return total / len(xs)


def Corr(xs, ys):
    xbar, varx = mystats.MeanVar(xs)
    ybar, vary = mystats.MeanVar(ys)
    corr = Cov(xs, ys, xbar, ybar) / math.sqrt(varx * vary)

    return corr


def SerialCorr(xs):
    return Corr(xs[:-1], xs[1:])


def SpearmanCorr(xs, ys):
    xranks = MapToRanks(xs)
    yranks = MapToRanks(ys)
    return Corr(xranks, yranks)


def LeastSquares(xs, ys):
    xbar, varx = mystats.MeanVar(xs)
    ybar, vary = mystats.MeanVar(ys)

    slope = Cov(xs, ys, xbar, ybar) / varx
    inter = ybar - slope * xbar

    return inter, slope


def FitLine(xs, inter, slope):
    fxs = min(xs), max(xs)
    fys = [x * slope + inter for x in fxs]
    return fxs, fys


def Residuals(xs, ys, inter, slope):
    res = [y - inter - slope * x for x, y in zip(xs, ys)]
    return res


def CoefDetermination(ys, res):
    ybar, vary = mystats.MeanVar(ys)
    resbar, varres = mystats.MeanVar(res)
    return 1 - varres / vary


def MapToRanks(t):
    # pair up each value with its index
    pairs = enumerate(t)
    # sort by value
    sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
    # pair up each pair with its rank
    ranked = enumerate(sorted_pairs)
    # sort by index
    resorted = sorted(ranked, key=lambda trip: trip[1][0])
    # extract the ranks
    ranks = [trip[0] + 1 for trip in resorted]
    return ranks


def CorrelatedGenerator(rho):
    x = random.gauss(0, 1)
    yield x
    sigma = math.sqrt(1 - rho ** 2)
    while True:
        x = random.gauss(x * rho, sigma)
        yield x


def CorrelatedNormalGenerator(mu, sigma, rho):
    for x in CorrelatedGenerator(rho):
        yield x * sigma + mu


def main():
    pass


if __name__ == "__main__":
    main()