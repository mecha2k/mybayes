from __future__ import print_function

import math
import numpy
import random
import sys
import matplotlib.pyplot as pyplot

import correlation
import myplots
import mybayes


INTERVAL = 245 / 365.0
MINSIZE = 0.2
MAXSIZE = 20
BUCKET_FACTOR = 10


def log2(x, denom=math.log(2)):
    return math.log(x) / denom


def SimpleModel():
    # time between discharge and diagnosis, in days
    interval = 3291.0

    # doubling time in linear measure is doubling time in volume * 3
    dt = 811.0 * 3

    # number of doublings since discharge
    doublings = interval / dt

    # how big was the tumor at time of discharge (diameter in cm)
    d1 = 15.5
    d0 = d1 / 2.0 ** doublings

    print("interval (days)", interval)
    print("interval (years)", interval / 365)
    print("dt", dt)
    print("doublings", doublings)
    print("d1", d1)
    print("d0", d0)

    # assume an initial linear measure of 0.1 cm
    d0 = 0.1
    d1 = 15.5

    # how many doublings would it take to get from d0 to d1
    doublings = log2(d1 / d0)

    # what linear doubling time does that imply?
    dt = interval / doublings

    print("doublings", doublings)
    print("dt", dt)

    # compute the volumetric doubling time and RDT
    vdt = dt / 3
    rdt = 365 / vdt

    print("vdt", vdt)
    print("rdt", rdt)

    cdf = MakeCdf()
    p = cdf.Prob(rdt)
    print("Prob{RDT > 2.4}", 1 - p)


def MakeCdf():
    n = 53.0
    freqs = [0, 2, 31, 42, 48, 51, 52, 53]
    ps = [freq / n for freq in freqs]
    xs = numpy.arange(-1.5, 6.5, 1.0)

    cdf = mybayes.Cdf(xs, ps)
    return cdf


def PlotCdf(cdf):
    xs, ps = cdf.xs, cdf.ps
    cps = [1 - p for p in ps]

    # CCDF on logy scale: shows exponential behavior
    myplots.Clf()
    myplots.Plot(xs, cps, "bo-")
    myplots.Save(
        root="kidney1", formats=["png"], xlabel="RDT", ylabel="CCDF (log scale)", yscale="log"
    )

    # CDF, model and data
    myplots.Clf()
    myplots.PrePlot(num=2)
    mxs, mys = ModelCdf()
    myplots.Plot(mxs, mys, label="model", linestyle="dashed")

    myplots.Plot(xs, ps, "gs", label="data")
    myplots.Save(
        root="kidney2",
        formats=["png"],
        xlabel="RDT (volume doublings per year)",
        ylabel="CDF",
        title="Distribution of RDT",
        axis=[-2, 7, 0, 1],
        loc=4,
    )


def QQPlot(cdf, fit):
    xs = [-1.5, 5.5]
    myplots.Clf()
    myplots.Plot(xs, xs, "b-")

    xs, ps = cdf.xs, cdf.ps
    fs = [fit.Value(p) for p in ps]

    myplots.Plot(xs, fs, "gs")
    myplots.Save(root="kidney3", formats=["png"], xlabel="Actual", ylabel="Model")


def FitCdf(cdf):
    xs, ps = cdf.xs, cdf.ps
    cps = [1 - p for p in ps]

    xs = xs[1:-1]
    lcps = [math.log(p) for p in cps[1:-1]]

    _inter, slope = correlation.LeastSquares(xs, lcps)
    return -slope


def CorrelatedGenerator(cdf, rho):
    def Transform(x):
        p = mybayes.GaussianCdf(x)
        y = cdf.Value(p)
        return y

    # for the first value, choose from a Gaussian and transform it
    x = random.gauss(0, 1)
    yield Transform(x)

    # for subsequent values, choose from the conditional distribution
    # based on the previous value
    sigma = math.sqrt(1 - rho ** 2)
    while True:
        x = random.gauss(x * rho, sigma)
        yield Transform(x)


def UncorrelatedGenerator(cdf, _rho=None):
    while True:
        x = cdf.Random()
        yield x


def RdtGenerator(cdf, rho):
    if rho == 0.0:
        return UncorrelatedGenerator(cdf)
    else:
        return CorrelatedGenerator(cdf, rho)


def GenerateRdt(pc, lam1, lam2):
    if random.random() < pc:
        return -random.expovariate(lam2)
    else:
        return random.expovariate(lam1)


def GenerateSample(n, pc, lam1, lam2):
    xs = [GenerateRdt(pc, lam1, lam2) for _ in range(n)]
    return xs


def GenerateCdf(n=1000, pc=0.35, lam1=0.79, lam2=5.0):
    xs = GenerateSample(n, pc, lam1, lam2)
    cdf = mybayes.MakeCdfFromList(xs)
    return cdf


def ModelCdf(pc=0.35, lam1=0.79, lam2=5.0):
    cdf = mybayes.EvalExponentialCdf
    x1 = numpy.arange(-2, 0, 0.1)
    y1 = [pc * (1 - cdf(-x, lam2)) for x in x1]
    x2 = numpy.arange(0, 7, 0.1)
    y2 = [pc + (1 - pc) * cdf(x, lam1) for x in x2]
    return list(x1) + list(x2), y1 + y2


def BucketToCm(y, factor=BUCKET_FACTOR):
    return math.exp(y / factor)


def CmToBucket(x, factor=BUCKET_FACTOR):
    return round(factor * math.log(x))


def Diameter(volume, factor=3 / math.pi / 4, exp=1 / 3.0):
    return 2 * (factor * volume) ** exp


def Volume(diameter, factor=4 * math.pi / 3):
    return factor * (diameter / 2.0) ** 3


class Cache(object):
    def __init__(self):
        self.joint = mybayes.Joint()
        self.sequences = {}
        self.initial_rdt = []

    def GetBuckets(self):
        return self.sequences.keys()

    def GetSequence(self, bucket):
        return self.sequences[bucket]

    def ConditionalCdf(self, bucket, name=""):
        pmf = self.joint.Conditional(0, 1, bucket, name=name)
        cdf = pmf.MakeCdf()
        return cdf

    def ProbOlder(self, cm, age):
        bucket = CmToBucket(cm)
        cdf = self.ConditionalCdf(bucket)
        p = cdf.Prob(age)
        return 1 - p

    def GetDistAgeSize(self, size_thresh=MAXSIZE):
        joint = mybayes.Joint()

        for val, freq in self.joint.Items():
            age, bucket = val
            cm = BucketToCm(bucket)
            if cm > size_thresh:
                continue
            log_cm = math.log10(cm)
            joint.Set((age, log_cm), math.log(freq) * 10)

        return joint

    def Add(self, age, seq, rdt):
        final = seq[-1]
        cm = Diameter(final)
        bucket = CmToBucket(cm)
        self.joint.Incr((age, bucket))

        self.sequences.setdefault(bucket, []).append(seq)

        initial = seq[-2]
        self.initial_rdt.append((initial, rdt))

    def Print(self):
        for bucket in sorted(self.GetBuckets()):
            ss = self.GetSequence(bucket)
            diameter = BucketToCm(bucket)
            print(diameter, len(ss))

    def Correlation(self):
        vs, rdts = zip(*self.initial_rdt)
        lvs = [math.log(v) for v in vs]
        return correlation.Corr(lvs, rdts)


class Calculator(object):
    def __init__(self):
        self.cache = Cache()

    def MakeSequences(self, n, rho, cdf):
        sequences = []
        for i in range(n):
            rdt_seq = RdtGenerator(cdf, rho)
            seq = self.MakeSequence(rdt_seq)
            sequences.append(seq)

            if i % 100 == 0:
                print(i)

        return sequences

    def MakeSequence(self, rdt_seq, v0=0.01, interval=INTERVAL, vmax=Volume(MAXSIZE)):
        seq = (v0,)
        age = 0

        for rdt in rdt_seq:
            age += interval
            final, seq = self.ExtendSequence(age, seq, rdt, interval)
            if final > vmax:
                break

        return seq

    def ExtendSequence(self, age, seq, rdt, interval):
        initial = seq[-1]
        doublings = rdt * interval
        final = initial * 2 ** doublings
        new_seq = seq + (final,)
        self.cache.Add(age, new_seq, rdt)

        return final, new_seq

    def PlotBucket(self, bucket, color="blue"):
        sequences = self.cache.GetSequence(bucket)
        for seq in sequences:
            n = len(seq)
            age = n * INTERVAL
            ts = numpy.linspace(-age, 0, n)
            PlotSequence(ts, seq, color)

    def PlotBuckets(self):
        # 2.01, 4.95 cm, 9.97 cm
        buckets = [7.0, 16.0, 23.0]
        buckets = [23.0]
        colors = ["blue", "green", "red", "cyan"]

        myplots.Clf()
        for bucket, color in zip(buckets, colors):
            self.PlotBucket(bucket, color)

        myplots.Save(
            root="kidney5",
            formats=["png"],
            title="History of simulated tumors",
            axis=[-40, 1, MINSIZE, 12],
            xlabel="years",
            ylabel="diameter (cm, log scale)",
            yscale="log",
        )

    def PlotJointDist(self):
        myplots.Clf()

        joint = self.cache.GetDistAgeSize()
        myplots.Contour(joint, contour=False, pcolor=True)
        myplots.Save(
            root="kidney8",
            formats=["png"],
            axis=[0, 41, -0.7, 1.31],
            yticks=MakeLogTicks([0.2, 0.5, 1, 2, 5, 10, 20]),
            xlabel="ages",
            ylabel="diameter (cm, log scale)",
        )

    def PlotConditionalCdfs(self):
        buckets = [7.0, 16.0, 23.0, 27.0]
        # 2.01, 4.95 cm, 9.97 cm, 14.879 cm
        names = ["2 cm", "5 cm", "10 cm", "15 cm"]
        cdfs = []

        for bucket, name in zip(buckets, names):
            cdf = self.cache.ConditionalCdf(bucket, name)
            cdfs.append(cdf)

        myplots.Clf()
        myplots.PrePlot(num=len(cdfs))
        myplots.Cdfs(cdfs)
        myplots.Save(
            root="kidney6",
            title="Distribution of age for several diameters",
            formats=["png"],
            xlabel="tumor age (years)",
            ylabel="CDF",
            loc=4,
        )

    def PlotCredibleIntervals(self, xscale="linear"):
        xs = []
        ts = []
        percentiles = [95, 75, 50, 25, 5]
        min_size = 0.3

        # loop through the buckets, accumulate
        # xs: sequence of sizes in cm
        # ts: sequence of percentile tuples
        for _, bucket in enumerate(sorted(self.cache.GetBuckets())):
            cm = BucketToCm(bucket)
            if cm < min_size or cm > 20.0:
                continue
            xs.append(cm)
            cdf = self.cache.ConditionalCdf(bucket)
            ps = [cdf.Percentile(p) for p in percentiles]
            ts.append(ps)

        # dump the results into a table
        fp = open("kidney_table.tex", "w")
        PrintTable(fp, xs, ts)
        fp.close()

        # make the figure
        linewidths = [1, 2, 3, 2, 1]
        alphas = [0.3, 0.5, 1, 0.5, 0.3]
        labels = ["95th", "75th", "50th", "25th", "5th"]

        # transpose the ts so we have sequences for each percentile rank
        myplots.Clf()
        yys = zip(*ts)

        for ys, linewidth, alpha, label in zip(yys, linewidths, alphas, labels):
            options = dict(
                color="blue", linewidth=linewidth, alpha=alpha, label=label, markersize=2
            )

            # plot the data points
            myplots.Plot(xs, ys, "bo", **options)

            # plot the fit lines
            fxs = [min_size, 20.0]
            fys = FitLine(xs, ys, fxs)

            myplots.Plot(fxs, fys, **options)

            # put a label at the end of each line
            x, y = fxs[-1], fys[-1]
            pyplot.text(
                x * 1.05,
                y,
                label,
                color="blue",
                horizontalalignment="left",
                verticalalignment="center",
            )

        # make the figure
        myplots.Save(
            root="kidney7",
            formats=["png"],
            title="Credible interval for age vs diameter",
            xlabel="diameter (cm, log scale)",
            ylabel="tumor age (years)",
            xscale=xscale,
            xticks=MakeTicks([0.5, 1, 2, 5, 10, 20]),
            axis=[0.25, 35, 0, 45],
            legend=False,
        )


def PlotSequences(sequences):
    myplots.Clf()

    options = dict(color="gray", linewidth=1, linestyle="dashed")
    myplots.Plot([0, 40], [10, 10], **options)

    for seq in sequences:
        n = len(seq)
        age = n * INTERVAL
        ts = numpy.linspace(0, age, n)
        PlotSequence(ts, seq)

    myplots.Save(
        root="kidney4",
        formats=["png"],
        axis=[0, 40, MINSIZE, 20],
        title="Simulations of tumor growth",
        xlabel="tumor age (years)",
        yticks=MakeTicks([0.2, 0.5, 1, 2, 5, 10, 20]),
        ylabel="diameter (cm, log scale)",
        yscale="log",
    )


def PlotSequence(ts, seq, color="blue"):
    options = dict(color=color, linewidth=1, alpha=0.2)
    xs = [Diameter(v) for v in seq]

    myplots.Plot(ts, xs, **options)


def PrintCI(fp, cm, ps):
    fp.write("%0.1f" % round(cm, 1))
    for p in reversed(ps):
        fp.write(" & %0.1f " % round(p, 1))
    fp.write(r"\\" "\n")


def PrintTable(fp, xs, ts):
    fp.write(r"\begin{tabular}{|r||r|r|r|r|r|}" "\n")
    fp.write(r"\hline" "\n")
    fp.write(r"Diameter   & \multicolumn{5}{c|}{Percentiles of age} \\" "\n")
    fp.write(r"(cm)   & 5th & 25th & 50th & 75th & 95th \\" "\n")
    fp.write(r"\hline" "\n")

    for i, (cm, ps) in enumerate(zip(xs, ts)):
        # print(cm, ps)
        if i % 3 == 0:
            PrintCI(fp, cm, ps)

    fp.write(r"\hline" "\n")
    fp.write(r"\end{tabular}" "\n")


def FitLine(xs, ys, fxs):
    lxs = [math.log(x) for x in xs]
    inter, slope = correlation.LeastSquares(lxs, ys)
    # res = correlation.Residuals(lxs, ys, inter, slope)
    # r2 = correlation.CoefDetermination(ys, res)

    lfxs = [math.log(x) for x in fxs]
    fys = [inter + slope * x for x in lfxs]
    return fys


def MakeTicks(xs):
    labels = [str(x) for x in xs]
    return xs, labels


def MakeLogTicks(xs):
    lxs = [math.log10(x) for x in xs]
    labels = [str(x) for x in xs]
    return lxs, labels


def TestCorrelation(cdf):
    n = 10000
    rho = 0.4

    rdt_seq = CorrelatedGenerator(cdf, rho)
    xs = [next(rdt_seq) for _ in range(n)]

    rho2 = correlation.SerialCorr(xs)
    print(rho, rho2)
    cdf2 = mybayes.MakeCdfFromList(xs)

    myplots.Cdfs([cdf, cdf2])
    myplots.Show()


def main(script):
    for size in [1, 5, 10]:
        bucket = CmToBucket(size)
        print("Size, bucket", size, bucket)

    SimpleModel()
    random.seed(17)
    cdf = MakeCdf()
    lam1 = FitCdf(cdf)
    fit = GenerateCdf(lam1=lam1)
    # TestCorrelation(fit)
    PlotCdf(cdf)
    # QQPlot(cdf, fit)

    calc = Calculator()
    rho = 0.0
    sequences = calc.MakeSequences(100, rho, fit)
    PlotSequences(sequences)

    calc.PlotBuckets()
    _ = calc.MakeSequences(1900, rho, fit)
    print("V0-RDT correlation", calc.cache.Correlation())
    print("15.5 Probability age > 8 year", calc.cache.ProbOlder(15.5, 8))
    print("6.0 Probability age > 8 year", calc.cache.ProbOlder(6.0, 8))

    calc.PlotConditionalCdfs()
    calc.PlotCredibleIntervals(xscale="log")
    calc.PlotJointDist()


if __name__ == "__main__":
    main(*sys.argv)