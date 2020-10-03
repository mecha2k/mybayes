from __future__ import print_function

import mybayes as mb
import myplots as mp
import mystats as ms
import brfss

import math
import numpy
import pickle
import random
import scipy
import matplotlib.pyplot as pyplot

NUM_SIGMAS = 1


class Height(mb.Suite, mb.Joint):
    def __init__(self, mus, sigmas, name=""):
        pairs = [(mu, sigma) for mu in mus for sigma in sigmas]

        mb.Suite.__init__(self, pairs, name=name)

    def Likelihood(self, data, hypo):
        x = data
        mu, sigma = hypo
        like = scipy.stats.norm.pdf(x, mu, sigma)
        return like

    def LogLikelihood(self, data, hypo):
        x = data
        mu, sigma = hypo
        loglike = EvalGaussianLogPdf(x, mu, sigma)
        return loglike

    def LogUpdateSetFast(self, data):
        xs = tuple(data)
        n = len(xs)
        for hypo in self.Values():
            mu, sigma = hypo
            total = Summation(xs, mu)
            loglike = -n * math.log(sigma) - total / 2 / sigma ** 2
            self.Incr(hypo, loglike)

    def LogUpdateSetMeanVar(self, data):
        xs = data
        n = len(xs)
        m = numpy.mean(xs)
        s = numpy.std(xs)
        self.LogUpdateSetABC(n, m, s)

    def LogUpdateSetMedianIPR(self, data):
        xs = data
        n = len(xs)
        # compute summary stats
        median, s = MedianS(xs, num_sigmas=NUM_SIGMAS)
        print("median, s", median, s)
        self.LogUpdateSetABC(n, median, s)

    def LogUpdateSetABC(self, n, m, s):
        for hypo in sorted(self.Values()):
            mu, sigma = hypo
            # compute log likelihood of m, given hypo
            stderr_m = sigma / math.sqrt(n)
            loglike = EvalGaussianLogPdf(m, mu, stderr_m)
            # compute log likelihood of s, given hypo
            stderr_s = sigma / math.sqrt(2 * (n - 1))
            loglike += EvalGaussianLogPdf(s, sigma, stderr_s)
            self.Incr(hypo, loglike)


def EvalGaussianLogPdf(x, mu, sigma):
    return scipy.stats.norm.logpdf(x, mu, sigma)


def FindPriorRanges(xs, num_points, num_stderrs=3.0, median_flag=False):
    def MakeRange(estimate, stderr):
        spread = stderr * num_stderrs
        array = numpy.linspace(estimate - spread, estimate + spread, num_points)
        return array

    # estimate mean and stddev of xs
    n = len(xs)
    if median_flag:
        m, s = MedianS(xs, num_sigmas=NUM_SIGMAS)
    else:
        m = numpy.mean(xs)
        s = numpy.std(xs)
    print("classical estimators", m, s)

    # compute ranges for m and s
    stderr_m = s / math.sqrt(n)
    mus = MakeRange(m, stderr_m)
    stderr_s = s / math.sqrt(2 * (n - 1))
    sigmas = MakeRange(s, stderr_s)

    return mus, sigmas


def Summation(xs, mu, cache={}):
    try:
        return cache[xs, mu]
    except KeyError:
        ds = [(x - mu) ** 2 for x in xs]
        total = sum(ds)
        cache[xs, mu] = total
        return total


def CoefVariation(suite):
    pmf = mb.Pmf()
    for (m, s), p in suite.Items():
        pmf.Incr(s / m, p)
    return pmf


def PlotCdfs(d, labels):
    mp.Clf()
    for key, xs in d.items():
        mu = ms.Mean(xs)
        xs = ms.Jitter(xs, 1.3)
        xs = [x - mu for x in xs]
        cdf = mb.MakeCdfFromList(xs)
        mp.Cdf(cdf, label=labels[key])
    mp.Show()


def PlotPosterior(suite, pcolor=False, contour=True):
    mp.Clf()
    mp.Contour(suite.GetDict(), pcolor=pcolor, contour=contour)
    mp.Save(
        root="variability_posterior_%s" % suite.name,
        title="Posterior joint distribution",
        xlabel="Mean height (cm)",
        ylabel="Stddev (cm)",
    )


def PlotCoefVariation(suites):
    mp.Clf()
    mp.PrePlot(num=2)
    pmfs = {}
    for label, suite in suites.items():
        pmf = CoefVariation(suite)
        print("CV posterior mean", pmf.Mean())
        cdf = mb.MakeCdfFromPmf(pmf, label)
        mp.Cdf(cdf)
        pmfs[label] = pmf
    mp.Save(root="variability_cv", xlabel="Coefficient of variation", ylabel="Probability")
    print("female bigger", mb.PmfProbGreater(pmfs["female"], pmfs["male"]))
    print("male bigger", mb.PmfProbGreater(pmfs["male"], pmfs["female"]))


def PlotOutliers(samples):
    cdfs = []
    for label, sample in samples.items():
        outliers = [x for x in sample if x < 150]

        cdf = mb.MakeCdfFromList(outliers, label)
        cdfs.append(cdf)

    mp.Clf()
    mp.Cdfs(cdfs)
    mp.Save(
        root="variability_cdfs", title="CDF of height", xlabel="Reported height (cm)", ylabel="CDF"
    )


def PlotMarginals(suite):
    mp.Clf()
    pyplot.subplot(1, 2, 1)
    pmf_m = suite.Marginal(0)
    cdf_m = mb.MakeCdfFromPmf(pmf_m)
    mp.Cdf(cdf_m)
    pyplot.subplot(1, 2, 2)
    pmf_s = suite.Marginal(1)
    cdf_s = mb.MakeCdfFromPmf(pmf_s)
    mp.Cdf(cdf_s)
    mp.Show()


def DumpHeights(data_dir=".", n=10000):
    resp = brfss.Respondents()
    resp.ReadRecords(data_dir, n)

    d = {1: [], 2: []}
    [d[r.sex].append(r.htm3) for r in resp.records if r.htm3 != "NA"]

    fp = open("variability_data.pkl", "wb")
    pickle.dump(d, fp)
    fp.close()


def LoadHeights():
    fp = open("variability_data.pkl", "rb")
    d = pickle.load(fp)
    fp.close()
    return d


def UpdateSuite1(suite, xs):
    suite.UpdateSet(xs)


def UpdateSuite2(suite, xs):
    suite.Log()
    suite.LogUpdateSet(xs)
    suite.Exp()
    suite.Normalize()


def UpdateSuite3(suite, xs):
    suite.Log()
    suite.LogUpdateSetFast(xs)
    suite.Exp()
    suite.Normalize()


def UpdateSuite4(suite, xs):
    suite.Log()
    suite.LogUpdateSetMeanVar(xs)
    suite.Exp()
    suite.Normalize()


def UpdateSuite5(suite, xs):
    suite.Log()
    suite.LogUpdateSetMedianIPR(xs)
    suite.Exp()
    suite.Normalize()


def MedianIPR(xs, p):
    cdf = mb.MakeCdfFromList(xs)
    median = cdf.Percentile(50)

    alpha = (1 - p) / 2
    ipr = cdf.Value(1 - alpha) - cdf.Value(alpha)
    return median, ipr


def MedianS(xs, num_sigmas):
    half_p = mb.StandardGaussianCdf(num_sigmas) - 0.5
    median, ipr = MedianIPR(xs, half_p * 2)
    s = ipr / 2 / num_sigmas

    return median, s


def Summarize(xs):
    # print(smallest and largest)
    xs.sort()
    print("smallest", xs[:10])
    print("largest", xs[-10:])

    # print(median and interquartile range)
    cdf = mb.MakeCdfFromList(xs)
    print(cdf.Percentile(25), cdf.Percentile(50), cdf.Percentile(75))


def RunEstimate(update_func, num_points=31, median_flag=False):
    DumpHeights(n=10000000)
    d = LoadHeights()
    labels = {1: "male", 2: "female"}
    # PlotCdfs(d, labels)

    suites = {}
    for key, xs in d.items():
        name = labels[key]
        print(name, len(xs))
        Summarize(xs)

        xs = ms.Jitter(xs, 1.3)
        mus, sigmas = FindPriorRanges(xs, num_points, median_flag=median_flag)
        suite = Height(mus, sigmas, name)
        suites[name] = suite
        update_func(suite, xs)
        print("MLE", suite.MaximumLikelihood())
        PlotPosterior(suite)
        pmf_m = suite.Marginal(0)
        pmf_s = suite.Marginal(1)
        print("marginal mu", pmf_m.Mean(), pmf_m.Var())
        print("marginal sigma", pmf_s.Mean(), pmf_s.Var())
        # PlotMarginals(suite)

    PlotCoefVariation(suites)


def main():
    random.seed(17)
    func = UpdateSuite5
    median_flag = func == UpdateSuite5
    RunEstimate(func, median_flag=median_flag)


if __name__ == "__main__":
    main()


""" Results:
UpdateSuite1 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite2 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite3 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite4 (100):
marginal mu 162.816901408 0.547456009605
marginal sigma 6.30305516111 0.27544106054

UpdateSuite3 (1000):
marginal mu 163.722137405 0.0660294386397
marginal sigma 6.64453251495 0.0329935312671

UpdateSuite4 (1000):
marginal mu 163.722137405 0.0658920503302
marginal sigma 6.63692197049 0.0329689887609

UpdateSuite3 (all):
marginal mu 163.223475005 0.000203282582659
marginal sigma 7.26918836916 0.000101641131229

UpdateSuite4 (all):
marginal mu 163.223475004 0.000203281499857
marginal sigma 7.26916693422 0.000101640932082

UpdateSuite5 (all):
marginal mu 163.1805214 7.9399898468e-07
marginal sigma 7.29969524118 3.26257030869e-14

"""