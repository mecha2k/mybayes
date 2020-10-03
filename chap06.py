from mybayes import Suite, EstimatedPdf, MakeCdfFromList, GaussianPdf
import myplots


import csv
import numpy
import matplotlib.pyplot as pyplot


def ReadData(filename=None):
    fp = open(filename)
    reader = csv.reader(fp)
    res = []

    for t in reader:
        _heading = t[0]
        data = t[1:]
        try:
            data = [int(x) for x in data]
            res.append(data)
        except ValueError:
            pass

    fp.close()
    return list(zip(*res))


class Price(Suite):
    def __init__(self, pmf, player, name=""):
        Suite.__init__(self, pmf, name=name)
        self.player = player

    def Likelihood(self, data, hypo):
        price = hypo
        guess = data
        error = price - guess
        like = self.player.ErrorDensity(error)

        return like


class GainCalculator(object):
    def __init__(self, player, opponent):
        self.player = player
        self.opponent = opponent

    def ExpectedGains(self, low=0, high=75000, n=101):
        bids = numpy.linspace(low, high, n)

        gains = [self.ExpectedGain(bid) for bid in bids]

        return bids, gains

    def ExpectedGain(self, bid):
        suite = self.player.posterior
        total = 0
        for price, prob in sorted(suite.Items()):
            gain = self.Gain(bid, price)
            total += prob * gain
        return total

    def Gain(self, bid, price):
        # if you overbid, you get nothing
        if bid > price:
            return 0
        # otherwise compute the probability of winning
        diff = price - bid
        prob = self.ProbWin(diff)
        # if you are within 250 dollars, you win both showcases
        if diff <= 250:
            return 2 * price * prob
        else:
            return price * prob

    def ProbWin(self, diff):
        prob = self.opponent.ProbOverbid() + self.opponent.ProbWorseThan(diff)
        return prob


class Player(object):
    n = 101
    price_xs = numpy.linspace(0, 75000, n)

    def __init__(self, prices, bids, diffs):
        self.pdf_price = EstimatedPdf(prices)
        self.cdf_diff = MakeCdfFromList(diffs)

        mu = 0
        sigma = numpy.std(diffs)
        self.pdf_error = GaussianPdf(mu, sigma)

    def ErrorDensity(self, error):
        return self.pdf_error.Density(error)

    def PmfPrice(self):
        return self.pdf_price.MakePmf(self.price_xs)

    def CdfDiff(self):
        return self.cdf_diff

    def ProbOverbid(self):
        return self.cdf_diff.Prob(-1)

    def ProbWorseThan(self, diff):
        return 1 - self.cdf_diff.Prob(diff)

    def MakeBeliefs(self, guess):
        pmf = self.PmfPrice()
        self.prior = Price(pmf, self, name="prior")
        self.posterior = self.prior.Copy(name="posterior")
        self.posterior.Update(guess)

    def OptimalBid(self, guess, opponent):
        self.MakeBeliefs(guess)
        calc = GainCalculator(self, opponent)
        bids, gains = calc.ExpectedGains()
        gain, bid = max(zip(gains, bids))
        return bid, gain

    def PlotBeliefs(self, root):
        myplots.Clf()
        myplots.PrePlot(num=2)
        myplots.Pmfs([self.prior, self.posterior])
        myplots.Save(root=root, xlabel="price ($)", ylabel="PMF", formats=["png"])


def MakePlots(player1, player2):
    # plot the prior distribution of price for both players
    myplots.Clf()
    myplots.PrePlot(num=2)
    pmf1 = player1.PmfPrice()
    pmf1.name = "showcase 1"
    pmf2 = player2.PmfPrice()
    pmf2.name = "showcase 2"
    myplots.Pmfs([pmf1, pmf2])
    myplots.Save(root="price1", xlabel="price ($)", ylabel="PDF", formats=["png"])

    # plot the historical distribution of underness for both players
    myplots.Clf()
    myplots.PrePlot(num=2)
    cdf1 = player1.CdfDiff()
    cdf1.name = "player 1"
    cdf2 = player2.CdfDiff()
    cdf2.name = "player 2"

    print("Player median", cdf1.Percentile(50))
    print("Player median", cdf2.Percentile(50))
    print("Player 1 overbids", player1.ProbOverbid())
    print("Player 2 overbids", player2.ProbOverbid())

    myplots.Cdfs([cdf1, cdf2])
    myplots.Save(root="price2", xlabel="diff ($)", ylabel="CDF", formats=["png"])


def MakePlayers():
    data = ReadData(filename="data/showcases.2011.csv")
    data += ReadData(filename="data/showcases.2012.csv")
    cols = zip(*data)
    price1, price2, bid1, bid2, diff1, diff2 = cols

    # print(list(sorted(price1)))
    # print(len(price1))
    player1 = Player(price1, bid1, diff1)
    player2 = Player(price2, bid2, diff2)

    return player1, player2


def PlotExpectedGains(guess1=20000, guess2=40000):
    player1, player2 = MakePlayers()
    MakePlots(player1, player2)

    player1.MakeBeliefs(guess1)
    player2.MakeBeliefs(guess2)
    print("Player 1 prior mle", player1.prior.MaximumLikelihood())
    print("Player 2 prior mle", player2.prior.MaximumLikelihood())
    print("Player 1 mean", player1.posterior.Mean())
    print("Player 2 mean", player2.posterior.Mean())
    print("Player 1 mle", player1.posterior.MaximumLikelihood())
    print("Player 2 mle", player2.posterior.MaximumLikelihood())

    player1.PlotBeliefs("price3")
    player2.PlotBeliefs("price4")
    calc1 = GainCalculator(player1, player2)
    calc2 = GainCalculator(player2, player1)

    myplots.Clf()
    myplots.PrePlot(num=2)
    bids, gains = calc1.ExpectedGains()
    myplots.Plot(bids, gains, label="Player 1")
    print("Player 1 optimal bid", max(zip(gains, bids)))

    bids, gains = calc2.ExpectedGains()
    myplots.Plot(bids, gains, label="Player 2")
    print("Player 2 optimal bid", max(zip(gains, bids)))

    myplots.Save(root="price5", xlabel="bid ($)", ylabel="expected gain ($)", formats=["png"])


def PlotOptimalBid():
    player1, player2 = MakePlayers()
    guesses = numpy.linspace(15000, 60000, 21)

    res = []
    for guess in guesses:
        player1.MakeBeliefs(guess)
        mean = player1.posterior.Mean()
        mle = player1.posterior.MaximumLikelihood()
        calc = GainCalculator(player1, player2)
        bids, gains = calc.ExpectedGains()
        gain, bid = max(zip(gains, bids))
        res.append((guess, mean, mle, gain, bid))

    guesses, means, _mles, gains, bids = zip(*res)

    myplots.PrePlot(num=3)
    pyplot.plot([15000, 60000], [15000, 60000], color="gray")
    myplots.Plot(guesses, means, label="mean")
    # myplots.Plot(guesses, mles, label='MLE')
    myplots.Plot(guesses, bids, label="bid")
    myplots.Plot(guesses, gains, label="gain")
    myplots.Save(root="price6", xlabel="guessed price ($)", formats=["png"])


def TestCode(calc):
    # test ProbWin
    for diff in [0, 100, 1000, 10000, 20000]:
        print(diff, calc.ProbWin(diff))
    print
    # test Return
    price = 20000
    for bid in [17000, 18000, 19000, 19500, 19800, 20001]:
        print(bid, calc.Gain(bid, price))
    print


def main():
    PlotExpectedGains()
    PlotOptimalBid()


if __name__ == "__main__":
    main()