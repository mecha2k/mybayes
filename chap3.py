from mybayes import Suite
import myplots


class Dice(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1.0 / hypo


class Train(Dice):
    pass


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


if __name__ == "__main__":
    main()