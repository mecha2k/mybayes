from mybayes import Suite, Percentile, CredibleInterval
import myplots


class Euro(Suite):
    def Likelihood(self, data, hypo):
        x = hypo / 100.0
        if data == "H":
            return x
        else:
            return 1 - x


class Euro2(Suite):
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
    print("Median", Percentile(suite, 50))
    print("5th %ile", Percentile(suite, 5))
    print("95th %ile", Percentile(suite, 95))
    print("CI", CredibleInterval(suite, 90))


def PlotSuites(suites, root):
    myplots.Clf()
    myplots.PrePlot(len(suites))
    myplots.Pmfs(suites)
    myplots.Save(root=root, xlabel="x", ylabel="Probability", formats=["png"])


def Version1():
    suite = Euro(range(0, 101))
    heads, tails = 140, 110
    dataset = "H" * heads + "T" * tails

    for data in dataset:
        suite.Update(data)

    return suite


def Version2():
    suite = Euro(range(0, 101))
    heads, tails = 140, 110
    dataset = "H" * heads + "T" * tails

    suite.UpdateSet(dataset)
    return suite


def Version3():
    suite = Euro2(range(0, 101))
    heads, tails = 140, 110

    suite.Update((heads, tails))
    return suite


def SuiteLikelihood(suite, data):
    total = 0
    for hypo, prob in suite.Items():
        like = suite.Likelihood(data, hypo)
        total += prob * like
    return total


def main():
    suite1 = UniformPrior()
    suite1.name = "uniform"

    suite2 = TrianglePrior()
    suite2.name = "triangle"

    PlotSuites([suite1, suite2], "euro2")

    RunUpdate(suite1)
    Summarize(suite1)

    RunUpdate(suite2)
    Summarize(suite2)

    PlotSuites([suite1], "euro1")
    PlotSuites([suite1, suite2], "euro3")

    suite = Version3()
    print(suite.Mean())

    myplots.Pmf(suite)
    myplots.Show(xlabel="x", ylabel="Probability")

    data = 140, 110
    data = 8, 12

    suite = Euro()
    like_f = suite.Likelihood(data, 50)
    print("p(D|F)", like_f)

    actual_percent = 100.0 * 140 / 250
    likelihood = suite.Likelihood(data, actual_percent)
    print("p(D|B_cheat)", likelihood)
    print("p(D|B_cheat) / p(D|F)", likelihood / like_f)

    like40 = suite.Likelihood(data, 40)
    like60 = suite.Likelihood(data, 60)
    likelihood = 0.5 * like40 + 0.5 * like60
    print("p(D|B_two)", likelihood)
    print("p(D|B_two) / p(D|F)", likelihood / like_f)

    b_uniform = Euro(range(0, 101))
    b_uniform.Remove(50)
    b_uniform.Normalize()
    likelihood = SuiteLikelihood(b_uniform, data)
    print("p(D|B_uniform)", likelihood)
    print("p(D|B_uniform) / p(D|F)", likelihood / like_f)

    b_tri = TrianglePrior()
    b_tri.Remove(50)
    b_tri.Normalize()
    likelihood = b_tri.Update(data)
    print("p(D|B_tri)", likelihood)
    print("p(D|B_tri) / p(D|F)", likelihood / like_f)


if __name__ == "__main__":
    main()