from __future__ import print_function

import numpy
import math
import random
import sys

import mybayes as mb
import myplots as mp

# longest hypothetical time between trains, in seconds
UPPER_BOUND = 1200

# observed gaps between trains, in seconds
# collected using code in redline_data.py, run daily 4-6pm
# for 5 days, Monday 6 May 2013 to Friday 10 May 2013
OBSERVED_GAP_TIMES = [
    428.0,
    705.0,
    407.0,
    465.0,
    433.0,
    425.0,
    204.0,
    506.0,
    143.0,
    351.0,
    450.0,
    598.0,
    464.0,
    749.0,
    341.0,
    586.0,
    754.0,
    256.0,
    378.0,
    435.0,
    176.0,
    405.0,
    360.0,
    519.0,
    648.0,
    374.0,
    483.0,
    537.0,
    578.0,
    534.0,
    577.0,
    619.0,
    538.0,
    331.0,
    186.0,
    629.0,
    193.0,
    360.0,
    660.0,
    484.0,
    512.0,
    315.0,
    457.0,
    404.0,
    740.0,
    388.0,
    357.0,
    485.0,
    567.0,
    160.0,
    428.0,
    387.0,
    901.0,
    187.0,
    622.0,
    616.0,
    585.0,
    474.0,
    442.0,
    499.0,
    437.0,
    620.0,
    351.0,
    286.0,
    373.0,
    232.0,
    393.0,
    745.0,
    636.0,
    758.0,
]


def BiasPmf(pmf, name="", invert=False):
    new_pmf = pmf.Copy(name=name)

    for x in pmf.Values():
        if invert:
            new_pmf.Mult(x, 1.0 / x)
        else:
            new_pmf.Mult(x, x)

    new_pmf.Normalize()
    return new_pmf


def UnbiasPmf(pmf, name=""):
    return BiasPmf(pmf, name, invert=True)


def MakeUniformPmf(low, high):
    pmf = mb.Pmf()
    for x in MakeRange(low=low, high=high):
        pmf.Set(x, 1)
    pmf.Normalize()
    return pmf


def MakeRange(low=10, high=None, skip=10):
    if high is None:
        high = UPPER_BOUND

    return range(low, high + skip, skip)


class WaitTimeCalculator(object):
    def __init__(self, pmf, inverse=False):
        if inverse:
            self.pmf_zb = pmf
            self.pmf_z = UnbiasPmf(pmf, name="z")
        else:
            self.pmf_z = pmf
            self.pmf_zb = BiasPmf(pmf, name="zb")

        # distribution of wait time
        self.pmf_y = PmfOfWaitTime(self.pmf_zb)
        # the distribution of elapsed time is the same as the
        # distribution of wait time
        self.pmf_x = self.pmf_y

    def GenerateSampleWaitTimes(self, n):
        cdf_y = mb.MakeCdfFromPmf(self.pmf_y)
        sample = cdf_y.Sample(n)
        return sample

    def GenerateSampleGaps(self, n):
        cdf_zb = mb.MakeCdfFromPmf(self.pmf_zb)
        sample = cdf_zb.Sample(n)
        return sample

    def GenerateSamplePassengers(self, lam, n):
        zs = self.GenerateSampleGaps(n)
        xs, ys = SplitGaps(zs)
        res = []
        for x, y in zip(xs, ys):
            k1 = numpy.random.poisson(lam * x)
            k2 = numpy.random.poisson(lam * y)
            res.append((k1, y, k2))
        return res

    def PlotPmfs(self, root="redline0"):
        pmfs = ScaleDists([self.pmf_z, self.pmf_zb], 1.0 / 60)

        mp.Clf()
        mp.PrePlot(2)
        mp.Pmfs(pmfs)
        mp.Save(root=root, xlabel="Time (min)", ylabel="CDF", formats=["png"])

    def MakePlot(self, root="redline2"):
        print("Mean z", self.pmf_z.Mean() / 60)
        print("Mean zb", self.pmf_zb.Mean() / 60)
        print("Mean y", self.pmf_y.Mean() / 60)

        cdf_z = self.pmf_z.MakeCdf()
        cdf_zb = self.pmf_zb.MakeCdf()
        cdf_y = self.pmf_y.MakeCdf()
        cdfs = ScaleDists([cdf_z, cdf_zb, cdf_y], 1.0 / 60)

        mp.Clf()
        mp.PrePlot(3)
        mp.Cdfs(cdfs)
        mp.Save(root=root, xlabel="Time (min)", ylabel="CDF", formats=["png"])


def SplitGaps(zs):
    xs = [random.uniform(0, z) for z in zs]
    ys = [z - x for z, x in zip(zs, xs)]
    return xs, ys


def PmfOfWaitTime(pmf_zb):
    metapmf = mb.Pmf()
    for gap, prob in pmf_zb.Items():
        uniform = MakeUniformPmf(0, gap)
        metapmf.Set(uniform, prob)

    pmf_y = mb.MakeMixture(metapmf, name="y")
    return pmf_y


def ScaleDists(dists, factor):
    return [dist.Scale(factor) for dist in dists]


class ElapsedTimeEstimator(object):
    def __init__(self, wtc, lam, num_passengers):
        # prior for elapsed time
        self.prior_x = Elapsed(wtc.pmf_x, name="prior x")
        # posterior of elapsed time (based on number of passengers)
        self.post_x = self.prior_x.Copy(name="posterior x")
        self.post_x.Update((lam, num_passengers))
        # predictive distribution of wait time
        self.pmf_y = PredictWaitTime(wtc.pmf_zb, self.post_x)

    def MakePlot(self, root="redline3"):
        # observed gaps
        cdf_prior_x = self.prior_x.MakeCdf()
        cdf_post_x = self.post_x.MakeCdf()
        cdf_y = self.pmf_y.MakeCdf()

        cdfs = ScaleDists([cdf_prior_x, cdf_post_x, cdf_y], 1.0 / 60)

        mp.Clf()
        mp.PrePlot(3)
        mp.Cdfs(cdfs)
        mp.Save(root=root, xlabel="Time (min)", ylabel="CDF", formats=["png"])


class ArrivalRate(mb.Suite):
    def Likelihood(self, data, hypo):
        lam = hypo
        x, k = data
        like = mb.EvalPoissonPmf(k, lam * x)
        return like


class ArrivalRateEstimator(object):
    def __init__(self, passenger_data):
        # range for lambda
        low, high = 0, 5
        n = 51
        hypos = numpy.linspace(low, high, n) / 60
        self.prior_lam = ArrivalRate(hypos, name="prior")
        self.prior_lam.Remove(0)
        self.post_lam = self.prior_lam.Copy(name="posterior")

        for _k1, y, k2 in passenger_data:
            self.post_lam.Update((y, k2))
        print("Mean posterior lambda", self.post_lam.Mean())

    def MakePlot(self, root="redline1"):
        mp.Clf()
        mp.PrePlot(2)
        # convert units to passengers per minute
        prior = self.prior_lam.MakeCdf().Scale(60)
        post = self.post_lam.MakeCdf().Scale(60)
        mp.Cdfs([prior, post])
        mp.Save(root=root, xlabel="Arrival rate (passengers / min)", ylabel="CDF", formats=["png"])


class Elapsed(mb.Suite):
    def Likelihood(self, data, hypo):
        x = hypo
        lam, k = data
        like = mb.EvalPoissonPmf(k, lam * x)
        return like


def PredictWaitTime(pmf_zb, pmf_x):
    pmf_y = pmf_zb - pmf_x
    pmf_y.name = "pred y"
    RemoveNegatives(pmf_y)
    return pmf_y


def RemoveNegatives(pmf):
    for val in list(pmf.Values()):
        if val < 0:
            pmf.Remove(val)
    pmf.Normalize()


class Gaps(mb.Suite):
    def Likelihood(self, data, hypo):
        z = hypo
        y = data
        if y > z:
            return 0
        return 1.0 / z


class GapDirichlet(mb.Dirichlet):
    def __init__(self, xs):
        n = len(xs)
        mb.Dirichlet.__init__(self, n)
        self.xs = xs
        self.mean_zbs = []

    def PmfMeanZb(self):
        return mb.MakePmfFromList(self.mean_zbs)

    def Preload(self, data):
        mb.Dirichlet.Update(self, data)

    def Update(self, data):
        k, y = data
        print(k, y)
        prior = self.PredictivePmf(self.xs)
        gaps = Gaps(prior)
        gaps.Update(y)
        probs = gaps.Probs(self.xs)

        self.params += numpy.array(probs)


class GapDirichlet2(GapDirichlet):
    def Update(self, data):
        k, y = data
        # get the current best guess for pmf_z
        pmf_zb = self.PredictivePmf(self.xs)
        # use it to compute prior pmf_x, pmf_y, pmf_z
        wtc = WaitTimeCalculator(pmf_zb, inverse=True)
        # use the observed passengers to estimate posterior pmf_x
        elapsed = ElapsedTimeEstimator(wtc, lam=0.0333, num_passengers=k)
        # use posterior_x and observed y to estimate observed z
        obs_zb = elapsed.post_x + Floor(y)
        probs = obs_zb.Probs(self.xs)
        mean_zb = obs_zb.Mean()
        self.mean_zbs.append(mean_zb)
        print(k, y, mean_zb)
        # use observed z to update beliefs about pmf_z
        self.params += numpy.array(probs)


class GapTimeEstimator(object):
    def __init__(self, xs, pcounts, passenger_data):
        self.xs = xs
        self.pcounts = pcounts
        self.passenger_data = passenger_data
        self.wait_times = [y for _k1, y, _k2 in passenger_data]
        self.pmf_y = mb.MakePmfFromList(self.wait_times, name="y")

        dirichlet = GapDirichlet2(self.xs)
        dirichlet.params /= 1.0
        dirichlet.Preload(self.pcounts)
        dirichlet.params /= 20.0

        self.prior_zb = dirichlet.PredictivePmf(self.xs, name="prior zb")

        for k1, y, _k2 in passenger_data:
            dirichlet.Update((k1, y))

        self.pmf_mean_zb = dirichlet.PmfMeanZb()
        self.post_zb = dirichlet.PredictivePmf(self.xs, name="post zb")
        self.post_z = UnbiasPmf(self.post_zb, name="post z")

    def PlotPmfs(self):
        print("Mean y", self.pmf_y.Mean())
        print("Mean z", self.post_z.Mean())
        print("Mean zb", self.post_zb.Mean())
        mp.Pmf(self.pmf_y)
        mp.Pmf(self.post_z)
        mp.Pmf(self.post_zb)

    def MakePlot(self):
        mp.Cdf(self.pmf_y.MakeCdf())
        mp.Cdf(self.prior_zb.MakeCdf())
        mp.Cdf(self.post_zb.MakeCdf())
        mp.Cdf(self.pmf_mean_zb.MakeCdf())
        mp.Show()


def Floor(x, factor=10):
    return int(x / factor) * factor


def TestGte():
    random.seed(17)
    xs = [60, 120, 240]
    gap_times = [60, 60, 60, 60, 60, 120, 120, 120, 240, 240]
    # distribution of gap time (z)
    pdf_z = mb.EstimatedPdf(gap_times)
    pmf_z = pdf_z.MakePmf(xs, name="z")
    wtc = WaitTimeCalculator(pmf_z, inverse=False)

    lam = 0.0333
    n = 100
    passenger_data = wtc.GenerateSamplePassengers(lam, n)
    pcounts = [0, 0, 0]
    ite = GapTimeEstimator(xs, pcounts, passenger_data)
    mp.Clf()

    # mp.Cdf(wtc.pmf_z.MakeCdf(name="actual z"))
    mp.Cdf(wtc.pmf_zb.MakeCdf(name="actual zb"))
    ite.MakePlot()


class WaitMixtureEstimator(object):
    def __init__(self, wtc, are, num_passengers=15):
        self.metapmf = mb.Pmf()
        for lam, prob in sorted(are.post_lam.Items()):
            ete = ElapsedTimeEstimator(wtc, lam, num_passengers)
            self.metapmf.Set(ete.pmf_y, prob)
        self.mixture = mb.MakeMixture(self.metapmf)
        lam = are.post_lam.Mean()
        ete = ElapsedTimeEstimator(wtc, lam, num_passengers)
        self.point = ete.pmf_y

    def MakePlot(self, root="redline4"):
        mp.Clf()

        # plot the MetaPmf
        for pmf, prob in sorted(self.metapmf.Items()):
            cdf = pmf.MakeCdf().Scale(1.0 / 60)
            width = 2 / math.log(-math.log(prob))
            mp.Plot(cdf.xs, cdf.ps, alpha=0.2, linewidth=width, color="blue", label="")

        # plot the mixture and the distribution based on a point estimate
        mp.PrePlot(2)
        # mp.Cdf(self.point.MakeCdf(name='point').Scale(1.0/60))
        mp.Cdf(self.mixture.MakeCdf(name="mix").Scale(1.0 / 60))
        mp.Save(
            root=root, xlabel="Wait time (min)", ylabel="CDF", formats=["png"], axis=[0, 10, 0, 1]
        )


def GenerateSampleData(gap_times, lam=0.0333, n=10):
    xs = MakeRange(low=10)
    pdf_z = mb.EstimatedPdf(gap_times)
    pmf_z = pdf_z.MakePmf(xs, name="z")
    wtc = WaitTimeCalculator(pmf_z, inverse=False)
    passenger_data = wtc.GenerateSamplePassengers(lam, n)
    return wtc, passenger_data


def RandomSeed(x):
    random.seed(x)
    numpy.random.seed(x)


def RunSimpleProcess(gap_times, lam=0.0333, num_passengers=15, plot=True):
    global UPPER_BOUND
    UPPER_BOUND = 1200

    cdf_z = mb.MakeCdfFromList(gap_times).Scale(1.0 / 60)
    print("CI z", cdf_z.CredibleInterval(90))

    xs = MakeRange(low=10)
    pdf_z = mb.EstimatedPdf(gap_times)
    pmf_z = pdf_z.MakePmf(xs, name="z")
    wtc = WaitTimeCalculator(pmf_z, inverse=False)

    if plot:
        wtc.PlotPmfs()
        wtc.MakePlot()
    ete = ElapsedTimeEstimator(wtc, lam, num_passengers)
    if plot:
        ete.MakePlot()

    return wtc, ete


def RunMixProcess(gap_times, lam=0.0333, num_passengers=15, plot=True):
    global UPPER_BOUND
    UPPER_BOUND = 1200

    wtc, _ete = RunSimpleProcess(gap_times, lam, num_passengers)

    RandomSeed(20)
    passenger_data = wtc.GenerateSamplePassengers(lam, n=5)

    total_y = 0
    total_k2 = 0
    for k1, y, k2 in passenger_data:
        print(k1, y / 60, k2)
        total_y += y / 60
        total_k2 += k2
    print(total_k2, total_y)
    print("Average arrival rate", total_k2 / total_y)

    are = ArrivalRateEstimator(passenger_data)
    if plot:
        are.MakePlot()
    wme = WaitMixtureEstimator(wtc, are, num_passengers)
    if plot:
        wme.MakePlot()

    return wme


def RunLoop(gap_times, nums, lam=0.0333):
    global UPPER_BOUND
    UPPER_BOUND = 4000

    mp.Clf()
    RandomSeed(18)
    # resample gap_times
    n = 220
    cdf_z = mb.MakeCdfFromList(gap_times)
    sample_z = cdf_z.Sample(n)
    pmf_z = mb.MakePmfFromList(sample_z)

    # compute the biased pmf and add some long delays
    cdf_zp = BiasPmf(pmf_z).MakeCdf()
    sample_zb = cdf_zp.Sample(n) + [1800, 2400, 3000]

    # smooth the distribution of zb
    pdf_zb = mb.EstimatedPdf(sample_zb)
    xs = MakeRange(low=60)
    pmf_zb = pdf_zb.MakePmf(xs)

    # unbias the distribution of zb and make wtc
    pmf_z = UnbiasPmf(pmf_zb)
    wtc = WaitTimeCalculator(pmf_z)

    probs = []
    for num_passengers in nums:
        ete = ElapsedTimeEstimator(wtc, lam, num_passengers)

        # compute the posterior prob of waiting more than 15 minutes
        cdf_y = ete.pmf_y.MakeCdf()
        prob = 1 - cdf_y.Prob(900)
        probs.append(prob)

        # mp.Cdf(ete.pmf_y.MakeCdf(name=str(num_passengers)))

    mp.Plot(nums, probs)
    mp.Save(
        root="redline5",
        xlabel="Num passengers",
        ylabel="P(y > 15 min)",
        formats=["png"],
    )


def main(script):
    RunLoop(OBSERVED_GAP_TIMES, nums=[0, 5, 10, 15, 20, 25, 30, 35])
    RunMixProcess(OBSERVED_GAP_TIMES)


if __name__ == "__main__":
    main(*sys.argv)