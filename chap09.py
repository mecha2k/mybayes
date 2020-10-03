from __future__ import print_function

import math
import sys
import matplotlib.pyplot as pyplot

import mybayes as mb
import myplots as mp


def StrafingSpeed(alpha, beta, x):
    theta = math.atan2(x - alpha, beta)
    speed = beta / math.cos(theta) ** 2
    return speed


def MakeLocationPmf(alpha, beta, locations):
    pmf = mb.Pmf()
    for x in locations:
        prob = 1.0 / StrafingSpeed(alpha, beta, x)
        pmf.Set(x, prob)
    pmf.Normalize()
    return pmf


class Paintball(mb.Suite, mb.Joint):
    def __init__(self, alphas, betas, locations):
        self.locations = locations
        pairs = [(alpha, beta) for alpha in alphas for beta in betas]
        mb.Suite.__init__(self, pairs)

    def Likelihood(self, data, hypo):
        alpha, beta = hypo
        x = data
        pmf = MakeLocationPmf(alpha, beta, self.locations)
        like = pmf.Prob(x)
        return like


def MakePmfPlot(alpha=10):
    locations = range(0, 31)
    betas = [10, 20, 40]
    mp.PrePlot(num=len(betas))
    for beta in betas:
        pmf = MakeLocationPmf(alpha, beta, locations)
        pmf.name = "beta = %d" % beta
        mp.Pmf(pmf)
    mp.Save("paintball1", xlabel="Distance", ylabel="Prob", formats=["png"])


def MakePosteriorPlot(suite):
    marginal_alpha = suite.Marginal(0)
    marginal_alpha.name = "alpha"
    marginal_beta = suite.Marginal(1)
    marginal_beta.name = "beta"
    print("alpha CI", marginal_alpha.CredibleInterval(50))
    print("beta CI", marginal_beta.CredibleInterval(50))

    mp.PrePlot(num=2)
    # mp.Pmf(marginal_alpha)
    # mp.Pmf(marginal_beta)
    mp.Cdf(mb.MakeCdfFromPmf(marginal_alpha))
    mp.Cdf(mb.MakeCdfFromPmf(marginal_beta))
    mp.Save("paintball2", xlabel="Distance", ylabel="Prob", loc=4, formats=["png"])


def MakeConditionalPlot(suite):
    betas = [10, 20, 40]
    mp.PrePlot(num=len(betas))

    for beta in betas:
        cond = suite.Conditional(0, 1, beta)
        cond.name = "beta = %d" % beta
        mp.Pmf(cond)

    mp.Save("paintball3", xlabel="Distance", ylabel="Prob", formats=["png"])


def MakeContourPlot(suite):
    mp.Contour(suite.GetDict(), contour=False, pcolor=True)
    mp.Save("paintball4", xlabel="alpha", ylabel="beta", axis=[0, 30, 0, 20], formats=["png"])


def MakeCrediblePlot(suite):
    d = dict((pair, 0) for pair in suite.Values())

    percentages = [75, 50, 25]
    for p in percentages:
        interval = suite.MaxLikeInterval(p)
        for pair in interval:
            d[pair] += 1

    mp.Contour(d, contour=False, pcolor=True)
    pyplot.text(17, 4, "25", color="white")
    pyplot.text(17, 15, "50", color="white")
    pyplot.text(17, 30, "75")

    mp.Save("paintball5", xlabel="alpha", ylabel="beta", formats=["png"])


def main(script):
    alphas = range(0, 31)
    betas = range(1, 51)
    locations = range(0, 31)

    suite = Paintball(alphas, betas, locations)
    suite.UpdateSet([15, 16, 18, 21])

    MakeCrediblePlot(suite)
    MakeContourPlot(suite)
    MakePosteriorPlot(suite)
    MakeConditionalPlot(suite)
    MakePmfPlot()


if __name__ == "__main__":
    main(*sys.argv)