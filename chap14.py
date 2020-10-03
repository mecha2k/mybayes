from __future__ import print_function

import mybayes
import myplots

from math import exp


class Emitter(mybayes.Suite):
    def __init__(self, rs, f=0.1):
        detectors = [Detector(r, f) for r in rs]
        mybayes.Suite.__init__(self, detectors)

    def Update(self, data):
        mybayes.Suite.Update(self, data)

        for detector in self.Values():
            detector.Update()

    def Likelihood(self, data, hypo):
        detector = hypo
        like = detector.SuiteLikelihood(data)
        return like

    def DistOfR(self, name=""):
        items = [(detector.r, prob) for detector, prob in self.Items()]
        return mybayes.MakePmfFromItems(items, name=name)

    def DistOfN(self, name=""):
        return mybayes.MakeMixture(self, name=name)


class Emitter2(mybayes.Suite):
    def __init__(self, rs, f=0.1):
        detectors = [Detector(r, f) for r in rs]
        mybayes.Suite.__init__(self, detectors)

    def Likelihood(self, data, hypo):
        return hypo.Update(data)

    def DistOfR(self, name=""):
        items = [(detector.r, prob) for detector, prob in self.Items()]
        return mybayes.MakePmfFromItems(items, name=name)

    def DistOfN(self, name=""):
        return mybayes.MakeMixture(self, name=name)


class Detector(mybayes.Suite):
    def __init__(self, r, f, high=500, step=5):
        pmf = mybayes.MakePoissonPmf(r, high, step=step)
        mybayes.Suite.__init__(self, pmf, name=r)
        self.r = r
        self.f = f

    def Likelihood(self, data, hypo):
        k = data
        n = hypo
        p = self.f

        return mybayes.EvalBinomialPmf(k, n, p)

    def SuiteLikelihood(self, data):
        total = 0
        for hypo, prob in self.Items():
            like = self.Likelihood(data, hypo)
            total += prob * like
        return total


def main():
    k = 15
    f = 0.1

    # plot Detector suites for a range of hypothetical r
    myplots.PrePlot(num=3)
    for r in [100, 250, 400]:
        suite = Detector(r, f, step=1)
        suite.Update(k)
        myplots.Pmf(suite)
        print(suite.MaximumLikelihood())
    myplots.Save(root="jaynes1", xlabel="Number of particles (n)", ylabel="PMF", formats=["png"])

    # plot the posterior distributions of r and n
    hypos = range(1, 501, 5)
    suite = Emitter2(hypos, f=f)
    suite.Update(k)
    myplots.PrePlot(num=2)
    post_r = suite.DistOfR(name="posterior r")
    post_n = suite.DistOfN(name="posterior n")
    myplots.Pmf(post_r)
    myplots.Pmf(post_n)
    myplots.Save(root="jaynes2", xlabel="Emission rate", ylabel="PMF", formats=["png"])


if __name__ == "__main__":
    main()
