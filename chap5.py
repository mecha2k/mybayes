from __future__ import print_function

from mybayes import Pmf, MakeMixture, SampleSum, MakePmfFromCdf
import myplots
import random


class Die(Pmf):
    def __init__(self, sides, name=""):
        Pmf.__init__(self, name=name)
        for x in range(1, sides + 1):
            self.Set(x, 1)
        self.Normalize()


def PmfMax(pmf1, pmf2):
    res = Pmf()
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            res.Incr(max(v1, v2), p1 * p2)
    return res


def main():
    pmf_dice = Pmf()
    pmf_dice.Set(Die(4), 5)
    pmf_dice.Set(Die(6), 4)
    pmf_dice.Set(Die(8), 3)
    pmf_dice.Set(Die(12), 2)
    pmf_dice.Set(Die(20), 1)
    pmf_dice.Normalize()

    mix = Pmf()
    for die, weight in pmf_dice.Items():
        for outcome, prob in die.Items():
            mix.Incr(outcome, weight * prob)

    mix = MakeMixture(pmf_dice)
    myplots.Hist(mix, width=0.9)
    myplots.Save(root="dungeons3", xlabel="Outcome", ylabel="Probability", formats=["png"])

    random.seed(17)
    d6 = Die(6, "d6")

    dice = [d6] * 3
    three = SampleSum(dice, 1000)
    three.name = "sample"
    three.Print()

    three_exact = d6 + d6 + d6
    three_exact.name = "exact"
    three_exact.Print()

    myplots.PrePlot(num=2)
    myplots.Pmf(three)
    myplots.Pmf(three_exact, linestyle="dashed")
    myplots.Save(
        root="dungeons1",
        xlabel="Sum of three d6",
        ylabel="Probability",
        axis=[2, 19, 0, 0.15],
        formats=["png"],
    )

    myplots.Clf()
    myplots.PrePlot(num=1)

    # compute the distribution of the best attribute the hard way
    best_attr2 = PmfMax(three_exact, three_exact)
    best_attr4 = PmfMax(best_attr2, best_attr2)
    best_attr6 = PmfMax(best_attr4, best_attr2)
    # myplots.Pmf(best_attr6)

    # and the easy way
    best_attr_cdf = three_exact.Max(6)
    best_attr_cdf.name = ""
    best_attr_pmf = MakePmfFromCdf(best_attr_cdf)
    best_attr_pmf.Print()

    myplots.Pmf(best_attr_pmf)
    myplots.Save(
        root="dungeons2",
        xlabel="Best of three d6",
        ylabel="Probability",
        axis=[2, 19, 0, 0.23],
        formats=["png"],
    )


if __name__ == "__main__":
    main()