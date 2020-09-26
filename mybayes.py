from __future__ import print_function

import bisect
import copy
import logging
import math
import numpy
import random

import scipy.stats
from scipy.special import erf, erfinv, gammaln

ROOT2 = math.sqrt(2)


def RandomSeed(x):
    random.seed(x)
    numpy.random.seed(x)


def Odds(p):
    if p == 1:
        return float("inf")
    return p / (1 - p)


def Probability(o):
    return o / (o + 1)


def Probability2(yes, no):
    return float(yes) / (yes + no)


class UnimplementedMethodException(Exception):
    # Exception if someone calls a method that should be overridden.
    pass


class Cdf(object):
    def __init__(self, xs=None, ps=None, name=""):
        self.xs = [] if xs is None else xs
        self.ps = [] if ps is None else ps
        self.name = name

    def Copy(self, name=None):
        if name is None:
            name = self.name
        return Cdf(list(self.xs), list(self.ps), name)

    def MakePmf(self, name=None):
        return MakePmfFromCdf(self, name=name)

    def Values(self):
        return self.xs

    def Items(self):
        return zip(self.xs, self.ps)

    def Append(self, x, p):
        self.xs.append(x)
        self.ps.append(p)

    def Shift(self, term):
        new = self.Copy()
        new.xs = [x + term for x in self.xs]
        return new

    def Scale(self, factor):
        new = self.Copy()
        new.xs = [x * factor for x in self.xs]
        return new

    def Prob(self, x):
        if x < self.xs[0]:
            return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def Value(self, p):
        if p < 0 or p > 1:
            raise ValueError("Probability p must be in range [0, 1]")

        if p == 0:
            return self.xs[0]
        if p == 1:
            return self.xs[-1]
        index = bisect.bisect(self.ps, p)
        if p == self.ps[index - 1]:
            return self.xs[index - 1]
        else:
            return self.xs[index]

    def Percentile(self, p):
        return self.Value(p / 100.0)

    def Random(self):
        return self.Value(random.random())

    def Sample(self, n):
        return [self.Random() for i in range(n)]

    def Mean(self):
        old_p = 0
        total = 0.0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def CredibleInterval(self, percentage=90):
        prob = (1 - percentage / 100.0) / 2
        interval = self.Value(prob), self.Value(1 - prob)
        return interval

    def _Round(self, multiplier=1000.0):
        # TODO(write this method)
        raise UnimplementedMethodException()

    def Render(self):
        xs = [self.xs[0]]
        ps = [0.0]
        for i, p in enumerate(self.ps):
            xs.append(self.xs[i])
            ps.append(p)

            try:
                xs.append(self.xs[i + 1])
                ps.append(p)
            except IndexError:
                pass
        return xs, ps

    def Max(self, k):
        cdf = self.Copy()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf


class _DictWrapper(object):
    def __init__(self, values=None, name=""):
        self.name = name
        self.d = {}

        # flag whether the distribution is under a log transform
        self.log = False

        if values is None:
            return

        init_methods = [
            self.InitPmf,
            self.InitMapping,
            self.InitSequence,
            self.InitFailure,
        ]

        for method in init_methods:
            try:
                method(values)
                break
            except AttributeError:
                continue

        if len(self) > 0:
            self.Normalize()

    def InitSequence(self, values):
        for value in values:
            self.Set(value, 1)

    def InitMapping(self, values):
        for value, prob in values.items():
            self.Set(value, prob)

    def InitPmf(self, values):
        for value, prob in values.Items():
            self.Set(value, prob)

    def InitFailure(self, values):
        raise ValueError("None of the initialization methods worked.")

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def keys(self):
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def Copy(self, name=None):
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.name = name if name is not None else self.name
        return new

    def Scale(self, factor):
        new = self.Copy()
        new.d.clear()

        for val, prob in self.Items():
            new.Set(val * factor, prob)
        return new

    def Log(self, m=None):
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            if p:
                self.Set(x, math.log(p / m))
            else:
                self.Remove(x)

    def Exp(self, m=None):
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            self.Set(x, math.exp(p - m))

    def GetDict(self):
        return self.d

    def SetDict(self, d):
        self.d = d

    def Values(self):
        return self.d.keys()

    def Items(self):
        return self.d.items()

    def Render(self):
        return zip(*sorted(self.Items()))

    def Print(self):
        for val, prob in sorted(self.d.items()):
            print(val, prob)

    def Set(self, x, y=0):
        self.d[x] = y

    def Incr(self, x, term=1):
        self.d[x] = self.d.get(x, 0) + term

    def Mult(self, x, factor):
        self.d[x] = self.d.get(x, 0) * factor

    def Remove(self, x):
        del self.d[x]

    def Total(self):
        total = sum(self.d.values())
        return total

    def MaxLike(self):
        return max(self.d.values())


class Pmf(_DictWrapper):
    def Prob(self, x, default=0):
        return self.d.get(x, default)

    def Probs(self, xs):
        return [self.Prob(x) for x in xs]

    def MakeCdf(self, name=None):
        return MakeCdfFromPmf(self, name=name)

    def ProbGreater(self, x):
        t = [prob for (val, prob) in self.d.items() if val > x]
        return sum(t)

    def ProbLess(self, x):
        t = [prob for (val, prob) in self.d.items() if val < x]
        return sum(t)

    def __lt__(self, obj):
        if isinstance(obj, _DictWrapper):
            return PmfProbLess(self, obj)
        else:
            return self.ProbLess(obj)

    def __gt__(self, obj):
        if isinstance(obj, _DictWrapper):
            return PmfProbGreater(self, obj)
        else:
            return self.ProbGreater(obj)

    def __ge__(self, obj):
        return 1 - (self < obj)

    def __le__(self, obj):
        return 1 - (self > obj)

    def __eq__(self, obj):
        if isinstance(obj, _DictWrapper):
            return PmfProbEqual(self, obj)
        else:
            return self.Prob(obj)

    def __ne__(self, obj):
        return 1 - (self == obj)

    def Normalize(self, fraction=1.0):
        if self.log:
            raise ValueError("Pmf is under a log transform")

        total = self.Total()
        if total == 0.0:
            raise ValueError("total probability is zero.")
            logging.warning("Normalize: total probability is zero.")
            return total

        factor = float(fraction) / total
        for x in self.d:
            self.d[x] *= factor

        return total

    def Random(self):
        if len(self.d) == 0:
            raise ValueError("Pmf contains no values.")

        target = random.random()
        total = 0.0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        assert False

    def Mean(self):
        mu = 0.0
        for x, p in self.d.items():
            mu += p * x
        return mu

    def Var(self, mu=None):
        if mu is None:
            mu = self.Mean()

        var = 0.0
        for x, p in self.d.items():
            var += p * (x - mu) ** 2
        return var

    def MaximumLikelihood(self):
        prob, val = max((prob, val) for val, prob in self.Items())
        return val

    def CredibleInterval(self, percentage=90):
        cdf = self.MakeCdf()
        return cdf.CredibleInterval(percentage)

    def __add__(self, other):
        try:
            return self.AddPmf(other)
        except AttributeError:
            return self.AddConstant(other)

    def AddPmf(self, other):
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 + v2, p1 * p2)
        return pmf

    def AddConstant(self, other):
        pmf = Pmf()
        for v1, p1 in self.Items():
            pmf.Set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 - v2, p1 * p2)
        return pmf

    def Max(self, k):
        cdf = self.MakeCdf()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf

    def __hash__(self):
        # FIXME
        # This imitates python2 implicit behaviour, which was removed in python3

        # Some problems with an id based hash:
        # looking up different pmfs with the same contents will give different values
        # looking up a new Pmf will always produce a keyerror

        # A solution might be to make a "FrozenPmf" immutable class (like frozenset)
        # and base a hash on a tuple of the items of self.d
        return id(self)


class Hist(_DictWrapper):
    def Freq(self, x):
        return self.d.get(x, 0)

    def Freqs(self, xs):
        return [self.Freq(x) for x in xs]

    def IsSubset(self, other):
        for val, freq in self.Items():
            if freq > other.Freq(val):
                return False
        return True

    def Subtract(self, other):
        for val, freq in other.Items():
            self.Incr(val, -freq)


class Suite(Pmf):
    def Update(self, data):
        for hypo in list(self.Values()):
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        return self.Normalize()

    def LogUpdate(self, data):
        for hypo in self.Values():
            like = self.LogLikelihood(data, hypo)
            self.Incr(hypo, like)

    def UpdateSet(self, dataset):
        for data in dataset:
            for hypo in self.Values():
                like = self.Likelihood(data, hypo)
                self.Mult(hypo, like)
        return self.Normalize()

    def LogUpdateSet(self, dataset):
        for data in dataset:
            self.LogUpdate(data)

    def Likelihood(self, data, hypo):
        raise UnimplementedMethodException()

    def LogLikelihood(self, data, hypo):
        raise UnimplementedMethodException()

    def Print(self):
        for hypo, prob in sorted(self.Items()):
            print(hypo, prob)

    def MakeOdds(self):
        for hypo, prob in self.Items():
            if prob:
                self.Set(hypo, Odds(prob))
            else:
                self.Remove(hypo)

    def MakeProbs(self):
        for hypo, odds in self.Items():
            self.Set(hypo, Probability(odds))


class Pdf(object):
    def Density(self, x):
        raise UnimplementedMethodException()

    def MakePmf(self, xs, name=""):
        pmf = Pmf(name=name)
        for x in xs:
            pmf.Set(x, self.Density(x))
        pmf.Normalize()
        return pmf


class GaussianPdf(Pdf):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def Density(self, x):
        return EvalGaussianPdf(x, self.mu, self.sigma)


class EstimatedPdf(Pdf):
    def __init__(self, sample):
        self.kde = scipy.stats.gaussian_kde(sample)

    def Density(self, x):
        return self.kde.evaluate(x)

    def MakePmf(self, xs, name=""):
        ps = self.kde.evaluate(xs)
        pmf = MakePmfFromItems(zip(xs, ps), name=name)
        return pmf


class Joint(Pmf):
    def Marginal(self, i, name=""):
        pmf = Pmf(name=name)
        for vs, prob in self.Items():
            pmf.Incr(vs[i], prob)
        return pmf

    def Conditional(self, i, j, val, name=""):
        pmf = Pmf(name=name)
        for vs, prob in self.Items():
            if vs[j] != val:
                continue
            pmf.Incr(vs[i], prob)

        pmf.Normalize()
        return pmf

    def MaxLikeInterval(self, percentage=90):
        interval = []
        total = 0

        t = [(prob, val) for val, prob in self.Items()]
        t.sort(reverse=True)

        for prob, val in t:
            interval.append(val)
            total += prob
            if total >= percentage / 100.0:
                break

        return interval


def MakeJoint(pmf1, pmf2):
    joint = Joint()
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            joint.Set((v1, v2), p1 * p2)
    return joint


def MakeHistFromList(t, name=""):
    hist = Hist(name=name)
    [hist.Incr(x) for x in t]
    return hist


def MakeHistFromDict(d, name=""):
    return Hist(d, name)


def MakePmfFromList(t, name=""):
    hist = MakeHistFromList(t)
    d = hist.GetDict()
    pmf = Pmf(d, name)
    pmf.Normalize()
    return pmf


def MakePmfFromDict(d, name=""):
    pmf = Pmf(d, name)
    pmf.Normalize()
    return pmf


def MakePmfFromItems(t, name=""):
    pmf = Pmf(dict(t), name)
    pmf.Normalize()
    return pmf


def MakePmfFromHist(hist, name=None):
    if name is None:
        name = hist.name

    # make a copy of the dictionary
    d = dict(hist.GetDict())
    pmf = Pmf(d, name)
    pmf.Normalize()
    return pmf


def MakePmfFromCdf(cdf, name=None):
    if name is None:
        name = cdf.name

    pmf = Pmf(name=name)

    prev = 0.0
    for val, prob in cdf.Items():
        pmf.Incr(val, prob - prev)
        prev = prob

    return pmf


def MakeMixture(metapmf, name="mix"):
    mix = Pmf(name=name)
    for pmf, p1 in metapmf.Items():
        for x, p2 in pmf.Items():
            mix.Incr(x, p1 * p2)
    return mix


def MakeUniformPmf(low, high, n):
    pmf = Pmf()
    for x in numpy.linspace(low, high, n):
        pmf.Set(x, 1)
    pmf.Normalize()
    return pmf


def MakeCdfFromItems(items, name=""):
    runsum = 0
    xs = []
    cs = []

    for value, count in sorted(items):
        runsum += count
        xs.append(value)
        cs.append(runsum)

    total = float(runsum)
    ps = [c / total for c in cs]

    cdf = Cdf(xs, ps, name)
    return cdf


def MakeCdfFromDict(d, name=""):
    return MakeCdfFromItems(d.items(), name)


def MakeCdfFromHist(hist, name=""):
    return MakeCdfFromItems(hist.Items(), name)


def MakeCdfFromPmf(pmf, name=None):
    if name == None:
        name = pmf.name
    return MakeCdfFromItems(pmf.Items(), name)


def MakeCdfFromList(seq, name=""):
    hist = MakeHistFromList(seq)
    return MakeCdfFromHist(hist, name)


def Percentile(pmf, percentage):
    p = percentage / 100.0
    total = 0
    for val, prob in pmf.Items():
        total += prob
        if total >= p:
            return val


def CredibleInterval(pmf, percentage=90):
    cdf = pmf.MakeCdf()
    prob = (1 - percentage / 100.0) / 2
    interval = cdf.Value(prob), cdf.Value(1 - prob)
    return interval


def PmfProbLess(pmf1, pmf2):
    total = 0.0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 < v2:
                total += p1 * p2
    return total


def PmfProbGreater(pmf1, pmf2):
    total = 0.0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 > v2:
                total += p1 * p2
    return total


def PmfProbEqual(pmf1, pmf2):
    total = 0.0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 == v2:
                total += p1 * p2
    return total


def RandomSum(dists):
    total = sum(dist.Random() for dist in dists)
    return total


def SampleSum(dists, n):
    pmf = MakePmfFromList(RandomSum(dists) for i in range(n))
    return pmf


def EvalGaussianPdf(x, mu, sigma):
    return scipy.stats.norm.pdf(x, mu, sigma)


def MakeGaussianPmf(mu, sigma, num_sigmas, n=201):
    pmf = Pmf()
    low = mu - num_sigmas * sigma
    high = mu + num_sigmas * sigma

    for x in numpy.linspace(low, high, n):
        p = EvalGaussianPdf(x, mu, sigma)
        pmf.Set(x, p)
    pmf.Normalize()
    return pmf


def EvalBinomialPmf(k, n, p):
    return scipy.stats.binom.pmf(k, n, p)


def EvalPoissonPmf(k, lam):
    return scipy.stats.poisson.pmf(k, lam)


def MakePoissonPmf(lam, high, step=1):
    pmf = Pmf()
    for k in range(0, high + 1, step):
        p = EvalPoissonPmf(k, lam)
        pmf.Set(k, p)
    pmf.Normalize()
    return pmf


def EvalExponentialPdf(x, lam):
    return lam * math.exp(-lam * x)


def EvalExponentialCdf(x, lam):
    return 1 - math.exp(-lam * x)


def MakeExponentialPmf(lam, high, n=200):
    pmf = Pmf()
    for x in numpy.linspace(0, high, n):
        p = EvalExponentialPdf(x, lam)
        pmf.Set(x, p)
    pmf.Normalize()
    return pmf


def StandardGaussianCdf(x):
    return (erf(x / ROOT2) + 1) / 2


def GaussianCdf(x, mu=0, sigma=1):
    return StandardGaussianCdf(float(x - mu) / sigma)


def GaussianCdfInverse(p, mu=0, sigma=1):
    x = ROOT2 * erfinv(2 * p - 1)
    return mu + x * sigma


class Beta(object):
    def __init__(self, alpha=1, beta=1, name=""):
        self.alpha = alpha
        self.beta = beta
        self.name = name

    def Update(self, data):
        heads, tails = data
        self.alpha += heads
        self.beta += tails

    def Mean(self):
        return float(self.alpha) / (self.alpha + self.beta)

    def Random(self):
        return random.betavariate(self.alpha, self.beta)

    def Sample(self, n):
        size = (n,)
        return numpy.random.beta(self.alpha, self.beta, size)

    def EvalPdf(self, x):
        return x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)

    def MakePmf(self, steps=101, name=""):
        if self.alpha < 1 or self.beta < 1:
            cdf = self.MakeCdf()
            pmf = cdf.MakePmf()
            return pmf

        xs = [i / (steps - 1.0) for i in range(steps)]
        probs = [self.EvalPdf(x) for x in xs]
        pmf = MakePmfFromDict(dict(zip(xs, probs)), name)
        return pmf

    def MakeCdf(self, steps=101):
        xs = [i / (steps - 1.0) for i in range(steps)]
        ps = [scipy.special.betainc(self.alpha, self.beta, x) for x in xs]
        cdf = Cdf(xs, ps)
        return cdf


class Dirichlet(object):
    def __init__(self, n, conc=1, name=""):
        if n < 2:
            raise ValueError("A Dirichlet distribution with " "n<2 makes no sense")

        self.n = n
        self.params = numpy.ones(n, dtype=numpy.float) * conc
        self.name = name

    def Update(self, data):
        m = len(data)
        self.params[:m] += data

    def Random(self):
        p = numpy.random.gamma(self.params)
        return p / p.sum()

    def Likelihood(self, data):
        m = len(data)
        if self.n < m:
            return 0

        x = data
        p = self.Random()
        q = p[:m] ** x
        return q.prod()

    def LogLikelihood(self, data):
        m = len(data)
        if self.n < m:
            return float("-inf")

        x = self.Random()
        y = numpy.log(x[:m]) * data
        return y.sum()

    def MarginalBeta(self, i):
        alpha0 = self.params.sum()
        alpha = self.params[i]
        return Beta(alpha, alpha0 - alpha)

    def PredictivePmf(self, xs, name=""):
        alpha0 = self.params.sum()
        ps = self.params / alpha0
        return MakePmfFromItems(zip(xs, ps), name=name)


def BinomialCoef(n, k):
    return scipy.misc.comb(n, k)


def LogBinomialCoef(n, k):
    return n * numpy.log(n) - k * numpy.log(k) - (n - k) * numpy.log(n - k)