from mybayes import Suite, Pmf, MakeCdfFromPmf, Percentile
import myplots


class Dice(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1.0 / hypo


class Train(Dice):
    pass


class Train2(Dice):
    def __init__(self, hypos, alpha=1.0):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, pow(hypo, -alpha))
        self.Normalize()


def MakePosterior2(high, dataset, construct):
    hypos = range(1, high + 1)
    suite = construct(hypos)
    suite.name = str(high)

    for data in dataset:
        suite.Update(data)

    return suite


def ComparePriors():
    dataset = [60]
    high = 1000

    myplots.Clf()
    myplots.PrePlot(num=2)

    constructors = [Train, Train2]
    lables = ["uniform", "power law"]

    for constructor, label in zip(constructors, lables):
        suite = MakePosterior2(high, dataset, constructor)
        suite.name = label
        myplots.Pmf(suite)

    myplots.Save(root="train4", xlabel="Number of trains", ylabel="Probability")


def Mean(suite):
    total = 0
    for hypo, prob in suite.Items():
        total += hypo * prob
    return total


def MakePosterior(high, dataset):
    hypos = range(1, high + 1)
    suite = Train(hypos)
    suite.name = str(high)

    for data in dataset:
        suite.Update(data)

    myplots.Pmf(suite)
    return suite


def main():
    suite = Dice([4, 6, 8, 12, 20])
    suite.Update(6)
    print("After one 6")
    suite.Print()

    for roll in [6, 8, 7, 7, 5, 10]:
        suite.Update(roll)
    print("After more rolls...")
    suite.Print()

    dataset = [30, 60, 90]
    for high in [500, 1000, 2000]:
        suite = MakePosterior(high, dataset)
        print(high, suite.Mean())
    myplots.Save(root="train2", xlabel="Number of trains", ylabel="Probability")

    ComparePriors()
    myplots.Clf()
    myplots.PrePlot(num=3)

    for high in [500, 1000, 2000]:
        suite = MakePosterior2(high, dataset, Train2)
        print(high, suite.Mean())
    myplots.Save(root="train3", xlabel="Number of trains", ylabel="Probability")

    interval = (Percentile(suite, 5), Percentile(suite, 95))
    print(interval)
    print(type(interval))

    cdf = MakeCdfFromPmf(suite)
    interval = cdf.Percentile(5), cdf.Percentile(95)
    print(interval)


if __name__ == "__main__":
    main()