from __future__ import print_function

import csv
import math
import numpy
import sys
import matplotlib.pyplot as pyplot

import mybayes
import myplots


def ReadScale(filename="sat_scale.csv", col=2):
    def ParseRange(s):
        t = [int(x) for x in s.split("-")]
        return 1.0 * sum(t) / len(t)

    fp = open(filename)
    reader = csv.reader(fp)
    raws = []
    scores = []

    for t in reader:
        try:
            raw = int(t[col])
            raws.append(raw)
            score = ParseRange(t[col + 1])
            scores.append(score)
        except ValueError:
            pass

    raws.sort()
    scores.sort()
    return mybayes.Interpolator(raws, scores)


def ReadRanks(filename="sat_ranks.csv"):
    fp = open(filename)
    reader = csv.reader(fp)
    res = []

    for t in reader:
        try:
            score = int(t[0])
            freq = int(t[1])
            res.append((score, freq))
        except ValueError:
            pass

    return res


def DivideValues(pmf, denom):
    new = mybayes.Pmf()
    denom = float(denom)
    for val, prob in pmf.Items():
        x = val / denom
        new.Set(x, prob)
    return new


class Exam(object):
    def __init__(self):
        self.scale = ReadScale()

        scores = ReadRanks()
        score_pmf = mybayes.MakePmfFromDict(dict(scores))

        self.raw = self.ReverseScale(score_pmf)
        self.max_score = max(self.raw.Values())
        self.prior = DivideValues(self.raw, denom=self.max_score)

        center = -0.05
        width = 1.8
        self.difficulties = MakeDifficulties(center, width, self.max_score)

    def CompareScores(self, a_score, b_score, constructor):
        a_sat = constructor(self, a_score)
        b_sat = constructor(self, b_score)
        a_sat.PlotPosteriors(b_sat)

        if constructor is Sat:
            PlotJointDist(a_sat, b_sat)

        top = TopLevel("AB")
        top.Update((a_sat, b_sat))
        top.Print()

        ratio = top.Prob("A") / top.Prob("B")
        print("Likelihood ratio", ratio)

        posterior = ratio / (ratio + 1)
        print("Posterior", posterior)

        if constructor is Sat2:
            ComparePosteriorPredictive(a_sat, b_sat)

    def MakeRawScoreDist(self, efficacies):
        pmfs = mybayes.Pmf()
        for efficacy, prob in efficacies.Items():
            scores = self.PmfCorrect(efficacy)
            pmfs.Set(scores, prob)

        mix = mybayes.MakeMixture(pmfs)
        return mix

    def CalibrateDifficulty(self):
        myplots.Clf()
        myplots.PrePlot(num=2)

        cdf = mybayes.MakeCdfFromPmf(self.raw, name="data")
        myplots.Cdf(cdf)

        efficacies = mybayes.MakeGaussianPmf(0, 1.5, 3)
        pmf = self.MakeRawScoreDist(efficacies)
        cdf = mybayes.MakeCdfFromPmf(pmf, name="model")
        myplots.Cdf(cdf)

        myplots.Save(root="sat_calibrate", xlabel="raw score", ylabel="CDF", formats=["png"])

    def PmfCorrect(self, efficacy):
        pmf = PmfCorrect(efficacy, self.difficulties)
        return pmf

    def Lookup(self, raw):
        return self.scale.Lookup(raw)

    def Reverse(self, score):
        raw = self.scale.Reverse(score)
        return raw if raw > 0 else 0

    def ReverseScale(self, pmf):
        new = mybayes.Pmf()
        for val, prob in pmf.Items():
            raw = self.Reverse(val)
            new.Incr(raw, prob)
        return new


class Sat(mybayes.Suite):
    def __init__(self, exam, score):
        self.exam = exam
        self.score = score

        # start with the prior distribution
        mybayes.Suite.__init__(self, exam.prior)

        # update based on an exam score
        self.Update(score)

    def Likelihood(self, data, hypo):
        p_correct = hypo
        score = data

        k = self.exam.Reverse(score)
        n = self.exam.max_score
        like = mybayes.EvalBinomialPmf(k, n, p_correct)
        return like

    def PlotPosteriors(self, other):
        myplots.Clf()
        myplots.PrePlot(num=2)

        cdf1 = mybayes.MakeCdfFromPmf(self, "posterior %d" % self.score)
        cdf2 = mybayes.MakeCdfFromPmf(other, "posterior %d" % other.score)

        myplots.Cdfs([cdf1, cdf2])
        myplots.Save(
            xlabel="p_correct",
            ylabel="CDF",
            axis=[0.7, 1.0, 0.0, 1.0],
            root="sat_posteriors_p_corr",
            formats=["png"],
        )


class Sat2(mybayes.Suite):
    def __init__(self, exam, score):
        self.exam = exam
        self.score = score

        # start with the Gaussian prior
        efficacies = mybayes.MakeGaussianPmf(0, 1.5, 3)
        mybayes.Suite.__init__(self, efficacies)

        # update based on an exam score
        self.Update(score)

    def Likelihood(self, data, hypo):
        efficacy = hypo
        score = data
        raw = self.exam.Reverse(score)

        pmf = self.exam.PmfCorrect(efficacy)
        like = pmf.Prob(raw)
        return like

    def MakePredictiveDist(self):
        raw_pmf = self.exam.MakeRawScoreDist(self)
        return raw_pmf

    def PlotPosteriors(self, other):
        myplots.Clf()
        myplots.PrePlot(num=2)

        cdf1 = mybayes.MakeCdfFromPmf(self, "posterior %d" % self.score)
        cdf2 = mybayes.MakeCdfFromPmf(other, "posterior %d" % other.score)

        myplots.Cdfs([cdf1, cdf2])
        myplots.Save(
            xlabel="efficacy",
            ylabel="CDF",
            axis=[0, 4.6, 0.0, 1.0],
            root="sat_posteriors_eff",
            formats=["png"],
        )


def PlotJointDist(pmf1, pmf2, thresh=0.8):
    def Clean(pmf):
        vals = [val for val in pmf.Values() if val < thresh]
        [pmf.Remove(val) for val in vals]

    Clean(pmf1)
    Clean(pmf2)
    pmf = mybayes.MakeJoint(pmf1, pmf2)

    myplots.Figure(figsize=(6, 6))
    myplots.Contour(pmf, contour=False, pcolor=True)

    myplots.Plot([thresh, 1.0], [thresh, 1.0], color="gray", alpha=0.2, linewidth=4)

    myplots.Save(
        root="sat_joint",
        xlabel="p_correct Alice",
        ylabel="p_correct Bob",
        axis=[thresh, 1.0, thresh, 1.0],
        formats=["png"],
    )


def ComparePosteriorPredictive(a_sat, b_sat):
    a_pred = a_sat.MakePredictiveDist()
    b_pred = b_sat.MakePredictiveDist()

    # myplots.Clf()
    # myplots.Pmfs([a_pred, b_pred])
    # myplots.Show()

    a_like = mybayes.PmfProbGreater(a_pred, b_pred)
    b_like = mybayes.PmfProbLess(a_pred, b_pred)
    c_like = mybayes.PmfProbEqual(a_pred, b_pred)

    print("Posterior predictive")
    print("A", a_like)
    print("B", b_like)
    print("C", c_like)


def PlotPriorDist(pmf):
    myplots.Clf()
    myplots.PrePlot(num=1)

    cdf1 = mybayes.MakeCdfFromPmf(pmf, "prior")
    myplots.Cdf(cdf1)
    myplots.Save(root="sat_prior", xlabel="p_correct", ylabel="CDF", formats=["png"])


class TopLevel(mybayes.Suite):
    def Update(self, data):
        a_sat, b_sat = data

        a_like = mybayes.PmfProbGreater(a_sat, b_sat)
        b_like = mybayes.PmfProbLess(a_sat, b_sat)
        c_like = mybayes.PmfProbEqual(a_sat, b_sat)

        a_like += c_like / 2
        b_like += c_like / 2

        self.Mult("A", a_like)
        self.Mult("B", b_like)

        self.Normalize()


def ProbCorrect(efficacy, difficulty, a=1):
    return 1 / (1 + math.exp(-a * (efficacy - difficulty)))


def BinaryPmf(p):
    pmf = mybayes.Pmf()
    pmf.Set(1, p)
    pmf.Set(0, 1 - p)
    return pmf


def PmfCorrect(efficacy, difficulties):
    pmf0 = mybayes.Pmf([0])
    ps = [ProbCorrect(efficacy, difficulty) for difficulty in difficulties]
    pmfs = [BinaryPmf(p) for p in ps]
    dist = sum(pmfs, pmf0)
    return dist


def MakeDifficulties(center, width, n):
    low, high = center - width, center + width
    return numpy.linspace(low, high, n)


def ProbCorrectTable():
    efficacies = [3, 1.5, 0, -1.5, -3]
    difficulties = [-1.85, -0.05, 1.75]

    for eff in efficacies:
        print("%0.2f & " % eff, end="")
        for diff in difficulties:
            p = ProbCorrect(eff, diff)
            print("%0.2f & " % p, end="")
        print(r"\\")


def main(script):
    ProbCorrectTable()
    exam = Exam()

    PlotPriorDist(exam.prior)
    exam.CalibrateDifficulty()
    exam.CompareScores(780, 740, constructor=Sat)
    exam.CompareScores(780, 740, constructor=Sat2)


if __name__ == "__main__":
    main(*sys.argv)