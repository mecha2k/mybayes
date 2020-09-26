from __future__ import print_function

import mybayes
import myplots


class Euro(mybayes.Suite):
    def Likelihood(self, data, hypo):
        x = hypo / 100.0
        if data == "H":
            return x
        else:
            return 1 - x


class Euro2(mybayes.Suite):
    def Likelihood(self, data, hypo):
        x = hypo / 100.0
        heads, tails = data
        like = x ** heads * (1 - x) ** tails
        return like


def UniformPrior():
    suite = Euro(range(0, 101))
    return suite


def TrianglePrior():
    suite = Euro()
    for x in range(0, 51):
        suite.Set(x, x)
    for x in range(51, 101):
        suite.Set(x, 100 - x)
    suite.Normalize()
    return suite


def RunUpdate(suite, heads=140, tails=110):
    dataset = "H" * heads + "T" * tails

    for data in dataset:
        suite.Update(data)


def Summarize(suite):
    print(suite.Prob(50))

    print("MLE", suite.MaximumLikelihood())

    print("Mean", suite.Mean())
    print("Median", mybayes.Percentile(suite, 50))

    print("5th %ile", mybayes.Percentile(suite, 5))
    print("95th %ile", mybayes.Percentile(suite, 95))

    print("CI", mybayes.CredibleInterval(suite, 90))


def PlotSuites(suites, root):
    myplots.Clf()
    myplots.PrePlot(len(suites))
    myplots.Pmfs(suites)

    myplots.Save(root=root, xlabel="x", ylabel="Probability", formats=["pdf", "eps"])


def main():
    suite1 = UniformPrior()
    suite1.name = "uniform"

    suite2 = TrianglePrior()
    suite2.name = "triangle"

    PlotSuites([suite1, suite2], "euro2")

    # update
    RunUpdate(suite1)
    Summarize(suite1)

    RunUpdate(suite2)
    Summarize(suite2)

    # plot the posteriors
    PlotSuites([suite1], "euro1")
    PlotSuites([suite1, suite2], "euro3")


if __name__ == "__main__":
    main()
