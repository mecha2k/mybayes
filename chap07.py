from __future__ import print_function

import mybayes as mb
import myplots as mp
import mystats as ms
import columns

import math

USE_SUMMARY_DATA = True


class Hockey(mb.Suite):
    def __init__(self, name=""):
        if USE_SUMMARY_DATA:
            # prior based on each team's average goals scored
            mu = 2.8
            sigma = 0.3
        else:
            # prior based on each pair-wise match-up
            mu = 2.8
            sigma = 0.85
        pmf = mb.MakeGaussianPmf(mu, sigma, 4)
        mb.Suite.__init__(self, pmf, name=name)

    def Likelihood(self, data, hypo):
        lam = hypo
        k = data
        like = mb.EvalPoissonPmf(k, lam)
        return like


def MakeGoalPmf(suite, high=10):
    metapmf = mb.Pmf()
    for lam, prob in suite.Items():
        pmf = mb.MakePoissonPmf(lam, high)
        metapmf.Set(pmf, prob)
    mix = mb.MakeMixture(metapmf, name=suite.name)
    return mix


def MakeGoalTimePmf(suite):
    metapmf = mb.Pmf()
    for lam, prob in suite.Items():
        pmf = mb.MakeExponentialPmf(lam, high=2, n=2001)
        metapmf.Set(pmf, prob)
    mix = mb.MakeMixture(metapmf, name=suite.name)
    return mix


class Game(object):
    convert = dict()

    def clean(self):
        self.goals = self.pd1 + self.pd2 + self.pd3


def ReadHockeyData(filename=None):
    game_list = columns.read_csv(filename, Game)

    # map from gameID to list of two games
    games = {}
    for game in game_list:
        if game.season != 2011:
            continue
        key = game.game
        games.setdefault(key, []).append(game)

    # map from (team1, team2) to (score1, score2)
    pairs = {}
    for key, pair in games.items():
        t1, t2 = pair
        key = t1.team, t2.team
        entry = t1.total, t2.total
        pairs.setdefault(key, []).append(entry)

    ProcessScoresTeamwise(pairs)
    ProcessScoresPairwise(pairs)


def ProcessScoresPairwise(pairs):
    # map from (team1, team2) to list of goals scored
    goals_scored = {}
    for key, entries in pairs.items():
        t1, t2 = key
        for entry in entries:
            g1, g2 = entry
            goals_scored.setdefault((t1, t2), []).append(g1)
            goals_scored.setdefault((t2, t1), []).append(g2)

    # make a list of average goals scored
    lams = []
    for key, goals in goals_scored.items():
        if len(goals) < 3:
            continue
        lam = ms.Mean(goals)
        lams.append(lam)

    # make the distribution of average goals scored
    cdf = mb.MakeCdfFromList(lams)
    mp.Cdf(cdf)
    mp.Show()
    mu, var = ms.MeanVar(lams)
    print("mu, sig", mu, math.sqrt(var))
    print("BOS v VAN", pairs["BOS", "VAN"])


def ProcessScoresTeamwise(pairs):
    # map from team to list of goals scored
    goals_scored = {}
    for key, entries in pairs.items():
        t1, t2 = key
        for entry in entries:
            g1, g2 = entry
            goals_scored.setdefault(t1, []).append(g1)
            goals_scored.setdefault(t2, []).append(g2)

    # make a list of average goals scored
    lams = []
    for key, goals in goals_scored.items():
        lam = ms.Mean(goals)
        lams.append(lam)
    # make the distribution of average goals scored
    cdf = mb.MakeCdfFromList(lams)
    mp.Cdf(cdf)
    mp.Show()
    mu, var = ms.MeanVar(lams)
    print("mu, sig", mu, math.sqrt(var))


def main():
    suite1 = Hockey("bruins")
    suite2 = Hockey("canucks")

    mp.Clf()
    mp.PrePlot(num=2)
    mp.Pmf(suite1)
    mp.Pmf(suite2)
    mp.Save(root="hockey0", xlabel="Goals per game", ylabel="Probability", formats=["png"])

    suite1.UpdateSet([0, 2, 8, 4])
    suite2.UpdateSet([1, 3, 1, 0])

    mp.Clf()
    mp.PrePlot(num=2)
    mp.Pmf(suite1)
    mp.Pmf(suite2)
    mp.Save(root="hockey1", xlabel="Goals per game", ylabel="Probability", formats=["png"])

    goal_dist1 = MakeGoalPmf(suite1)
    goal_dist2 = MakeGoalPmf(suite2)

    mp.Clf()
    mp.PrePlot(num=2)
    mp.Pmf(goal_dist1)
    mp.Pmf(goal_dist2)
    mp.Save(root="hockey2", xlabel="Goals", ylabel="Probability", formats=["png"])

    time_dist1 = MakeGoalTimePmf(suite1)
    time_dist2 = MakeGoalTimePmf(suite2)

    print("MLE bruins", suite1.MaximumLikelihood())
    print("MLE canucks", suite2.MaximumLikelihood())

    mp.Clf()
    mp.PrePlot(num=2)
    mp.Pmf(time_dist1)
    mp.Pmf(time_dist2)
    mp.Save(root="hockey3", xlabel="Games until goal", ylabel="Probability", formats=["png"])

    diff = goal_dist1 - goal_dist2
    p_win = diff.ProbGreater(0)
    p_loss = diff.ProbLess(0)
    p_tie = diff.Prob(0)

    print(p_win, p_loss, p_tie)
    p_overtime = mb.PmfProbLess(time_dist1, time_dist2)
    p_adjust = mb.PmfProbEqual(time_dist1, time_dist2)
    p_overtime += p_adjust / 2
    print("p_overtime", p_overtime)
    print(p_overtime * p_tie)
    p_win += p_overtime * p_tie
    print("p_win", p_win)

    # win the next two
    p_series = p_win ** 2
    # split the next two, win the third
    p_series += 2 * p_win * (1 - p_win) * p_win
    print("p_series", p_series)


if __name__ == "__main__":
    main()