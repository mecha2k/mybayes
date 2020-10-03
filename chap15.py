from __future__ import print_function

import numpy
import csv
import random
import shelve
import sys
import time
import matplotlib.pyplot as pyplot
import warnings

import mybayes
import myplots

warnings.simplefilter("error", RuntimeWarning)


class Locker(object):
    def __init__(self, shelf_file):
        self.shelf = shelve.open(shelf_file)

    def Close(self):
        self.shelf.close()

    def Add(self, key, value):
        self.shelf[str(key)] = value

    def Lookup(self, key):
        return self.shelf.get(str(key))

    def Keys(self):
        return self.shelf.keys()

    def Read(self):
        return dict(self.shelf)


class Subject(object):
    def __init__(self, code):
        self.code = code
        self.species = []
        self.suite = None
        self.num_reads = None
        self.num_species = None
        self.total_reads = None
        self.total_species = None
        self.prev_unseen = None
        self.pmf_n = None
        self.pmf_q = None
        self.pmf_l = None

    def Add(self, species, count):
        self.species.append((count, species))

    def Done(self, reverse=False, clean_param=0):
        if clean_param:
            self.Clean(clean_param)

        self.species.sort(reverse=reverse)
        counts = self.GetCounts()
        self.num_species = len(counts)
        self.num_reads = sum(counts)

    def Clean(self, clean_param=50):
        def prob_bogus(k, r):
            q = clean_param / r
            p = (1 - q) ** k
            return p

        print(self.code, clean_param)

        counts = self.GetCounts()
        r = 1.0 * sum(counts)

        species_seq = []
        for k, species in sorted(self.species):

            if random.random() < prob_bogus(k, r):
                continue
            species_seq.append((k, species))
        self.species = species_seq

    def GetM(self):
        return len(self.species)

    def GetCounts(self):
        return [count for count, _ in self.species]

    def MakeCdf(self):
        counts = self.GetCounts()
        counts.sort(reverse=True)
        cdf = mybayes.MakeCdfFromItems(enumerate(counts))
        return cdf

    def GetNames(self):
        return [name for _, name in self.species]

    def PrintCounts(self):
        for count, name in reversed(self.species):
            print(count, name)

    def GetSpecies(self, index):
        return self.species[index]

    def GetCdf(self):
        counts = self.GetCounts()
        items = enumerate(counts)
        cdf = mybayes.MakeCdfFromItems(items)
        return cdf

    def GetPrevalences(self):
        counts = self.GetCounts()
        total = sum(counts)
        prevalences = numpy.array(counts, dtype=numpy.float) / total
        return prevalences

    def Process(self, low=None, high=500, conc=1, iters=100):
        counts = self.GetCounts()
        m = len(counts)
        if low is None:
            low = max(m, 2)
        ns = range(low, high + 1)

        # start = time.time()
        self.suite = Species5(ns, conc=conc, iters=iters)
        self.suite.Update(counts)
        # end = time.time()

        # print('Processing time' end-start)

    def MakePrediction(self, num_sims=100):
        add_reads = self.total_reads - self.num_reads
        curves = self.RunSimulations(num_sims, add_reads)
        self.pmf_l = self.MakePredictive(curves)

    def MakeQuickPrediction(self, num_sims=100):
        add_reads = self.total_reads - self.num_reads
        pmf = mybayes.Pmf()
        _, seen = self.GetSeenSpecies()

        for _ in range(num_sims):
            _, observations = self.GenerateObservations(add_reads)
            all_seen = seen.union(observations)
            l = len(all_seen) - len(seen)
            pmf.Incr(l)

        pmf.Normalize()
        self.pmf_l = pmf

    def DistL(self):
        return self.pmf_l

    def MakeFigures(self):
        self.PlotDistN()
        self.PlotPrevalences()

    def PlotDistN(self):
        pmf = self.suite.DistN()
        print("90% CI for N:", pmf.CredibleInterval(90))
        pmf.name = self.code

        myplots.Clf()
        myplots.PrePlot(num=1)
        myplots.Pmf(pmf)
        root = "species-ndist-%s" % self.code
        myplots.Save(
            root=root,
            xlabel="Number of species",
            ylabel="Prob",
            formats=["png"],
        )

    def PlotPrevalences(self, num=5):
        myplots.Clf()
        myplots.PrePlot(num=5)
        for rank in range(1, num + 1):
            self.PlotPrevalence(rank)
        root = "species-prev-%s" % self.code
        myplots.Save(
            root=root,
            xlabel="Prevalence",
            ylabel="Prob",
            formats=["png"],
            axis=[0, 0.3, 0, 1],
        )

    def PlotPrevalence(self, rank=1, cdf_flag=True):
        # convert rank to index
        index = self.GetM() - rank

        _, mix = self.suite.DistOfPrevalence(index)
        count, _ = self.GetSpecies(index)
        mix.name = "%d (%d)" % (rank, count)
        print("90%% CI for prevalence of species %d:" % rank, end="")
        print(mix.CredibleInterval(90))

        if cdf_flag:
            cdf = mix.MakeCdf()
            myplots.Cdf(cdf)
        else:
            myplots.Pmf(mix)

    def PlotMixture(self, rank=1):
        # convert rank to index
        index = self.GetM() - rank
        print(self.GetSpecies(index))
        print(self.GetCounts()[index])
        metapmf, mix = self.suite.DistOfPrevalence(index)

        myplots.Clf()
        for pmf in metapmf.Values():
            myplots.Pmf(pmf, color="blue", alpha=0.2, linewidth=0.5)
        myplots.Pmf(mix, color="blue", alpha=0.9, linewidth=2)

        root = "species-mix-%s" % self.code
        myplots.Save(
            root=root,
            xlabel="Prevalence",
            ylabel="Prob",
            formats=["png"],
            axis=[0, 0.3, 0, 0.3],
            legend=False,
        )

    def GetSeenSpecies(self):
        names = self.GetNames()
        m = len(names)
        seen = set(SpeciesGenerator(names, m))
        return m, seen

    def GenerateObservations(self, num_reads):
        n, prevalences = self.suite.SamplePosterior()

        names = self.GetNames()
        name_iter = SpeciesGenerator(names, n)

        items = zip(name_iter, prevalences)

        cdf = mybayes.MakeCdfFromItems(items)
        observations = cdf.Sample(num_reads)

        # for ob in observations:
        #    print(ob)

        return n, observations

    def Resample(self, num_reads):
        t = []
        for count, species in self.species:
            t.extend([species] * count)

        random.shuffle(t)
        reads = t[:num_reads]

        subject = Subject(self.code)
        hist = mybayes.MakeHistFromList(reads)
        for species, count in hist.Items():
            subject.Add(species, count)

        subject.Done()
        return subject

    def Match(self, match):
        self.total_reads = match.num_reads
        self.total_species = match.num_species

        # compute the prevalence of unseen species (at least approximately,
        # based on all species counts in match
        _, seen = self.GetSeenSpecies()

        seen_total = 0.0
        unseen_total = 0.0
        for count, species in match.species:
            if species in seen:
                seen_total += count
            else:
                unseen_total += count

        self.prev_unseen = unseen_total / (seen_total + unseen_total)

    def RunSimulation(self, num_reads, frac_flag=False, jitter=0.01):
        m, seen = self.GetSeenSpecies()
        n, observations = self.GenerateObservations(num_reads)

        curve = []
        for i, obs in enumerate(observations):
            seen.add(obs)

            if frac_flag:
                frac_seen = len(seen) / float(n)
                frac_seen += random.uniform(-jitter, jitter)
                curve.append((i + 1, frac_seen))
            else:
                num_new = len(seen) - m
                curve.append((i + 1, num_new))

        return curve

    def RunSimulations(self, num_sims, num_reads, frac_flag=False):
        curves = [self.RunSimulation(num_reads, frac_flag) for _ in range(num_sims)]
        return curves

    def MakePredictive(self, curves):
        pred = mybayes.Pmf(name=self.code)
        for curve in curves:
            _, last_num_new = curve[-1]
            pred.Incr(last_num_new)
        pred.Normalize()
        return pred


def MakeConditionals(curves, ks):
    joint = MakeJointPredictive(curves)

    cdfs = []
    for k in ks:
        pmf = joint.Conditional(1, 0, k)
        pmf.name = "k=%d" % k
        cdf = pmf.MakeCdf()
        cdfs.append(cdf)
        print("90%% credible interval for %d" % k, end="")
        print(cdf.CredibleInterval(90))
    return cdfs


def MakeJointPredictive(curves):
    joint = mybayes.Joint()
    for curve in curves:
        for k, num_new in curve:
            joint.Incr((k, num_new))
    joint.Normalize()
    return joint


def MakeFracCdfs(curves, ks):
    d = {}
    for curve in curves:
        for k, frac in curve:
            if k in ks:
                d.setdefault(k, []).append(frac)

    cdfs = {}
    for k, fracs in d.items():
        cdf = mybayes.MakeCdfFromList(fracs)
        cdfs[k] = cdf

    return cdfs


def SpeciesGenerator(names, num):
    i = 0
    for name in names:
        yield name
        i += 1

    while i < num:
        yield "unseen-%d" % i
        i += 1


def ReadRarefactedData(filename="journal.pone.0047712.s001.csv", clean_param=0):
    fp = open(filename)
    reader = csv.reader(fp)
    _ = next(reader)
    subject = Subject("")
    subject_map = {}

    i = 0
    for t in reader:
        code = t[0]
        if code != subject.code:
            # start a new subject
            subject = Subject(code)
            subject_map[code] = subject

        # append a number to the species names so they're unique
        species = t[1]
        species = "%s-%d" % (species, i)
        i += 1

        count = int(t[2])
        subject.Add(species, count)

    for code, subject in subject_map.items():
        subject.Done(clean_param=clean_param)

    return subject_map


def ReadCompleteDataset(filename="BBB_data_from_Rob.csv", clean_param=0):
    fp = open(filename)
    reader = csv.reader(fp)

    header = next(reader)
    header = next(reader)

    subject_codes = header[1:-1]
    subject_codes = ["B" + code for code in subject_codes]

    # create the subject map
    uber_subject = Subject("uber")
    subject_map = {}
    for code in subject_codes:
        subject_map[code] = Subject(code)

    # read lines
    i = 0
    for t in reader:
        otu_code = t[0]
        if otu_code == "":
            continue

        # pull out a species name and give it a number
        otu_names = t[-1]
        taxons = otu_names.split(";")
        species = taxons[-1]
        species = "%s-%d" % (species, i)
        i += 1

        counts = [int(x) for x in t[1:-1]]
        # print(otu_code, species)
        for code, count in zip(subject_codes, counts):
            if count > 0:
                subject_map[code].Add(species, count)
                uber_subject.Add(species, count)

    uber_subject.Done(clean_param=clean_param)
    for code, subject in subject_map.items():
        subject.Done(clean_param=clean_param)

    return subject_map, uber_subject


def JoinSubjects():
    # read the rarefacted dataset
    sampled_subjects = ReadRarefactedData()

    # read the complete dataset
    all_subjects, _ = ReadCompleteDataset()

    for code, subject in sampled_subjects.items():
        if code in all_subjects:
            match = all_subjects[code]
            subject.Match(match)

    return sampled_subjects


def JitterCurve(curve, dx=0.2, dy=0.3):
    curve = [(x + random.uniform(-dx, dx), y + random.uniform(-dy, dy)) for x, y in curve]
    return curve


def OffsetCurve(curve, i, n, dx=0.3, dy=0.3):
    xoff = -dx + 2 * dx * i / (n - 1)
    yoff = -dy + 2 * dy * i / (n - 1)
    curve = [(x + xoff, y + yoff) for x, y in curve]
    return curve


def PlotCurves(curves, root="species-rare"):
    myplots.Clf()
    color = "#225EA8"

    n = len(curves)
    for i, curve in enumerate(curves):
        curve = OffsetCurve(curve, i, n)
        xs, ys = zip(*curve)
        myplots.Plot(xs, ys, color=color, alpha=0.3, linewidth=0.5)

    myplots.Save(root=root, xlabel="# samples", ylabel="# species", formats=["png"], legend=False)


def PlotConditionals(cdfs, root="species-cond"):
    myplots.Clf()
    myplots.PrePlot(num=len(cdfs))

    myplots.Cdfs(cdfs)

    myplots.Save(root=root, xlabel="# new species", ylabel="Prob", formats=["png"])


def PlotFracCdfs(cdfs, root="species-frac"):
    myplots.Clf()
    color = "#225EA8"

    for k, cdf in cdfs.items():
        xs, ys = cdf.Render()
        ys = [1 - y for y in ys]
        myplots.Plot(xs, ys, color=color, linewidth=1)

        x = 0.9
        y = 1 - cdf.Prob(x)
        pyplot.text(
            x,
            y,
            str(k),
            fontsize=9,
            color=color,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", edgecolor="none"),
        )

    myplots.Save(
        root=root,
        xlabel="Fraction of species seen",
        ylabel="Probability",
        formats=["png"],
        legend=False,
    )


class Species(mybayes.Suite):
    def __init__(self, ns, conc=1, iters=1000):
        hypos = [mybayes.Dirichlet(n, conc) for n in ns]
        mybayes.Suite.__init__(self, hypos)
        self.iters = iters

    def Update(self, data):
        # call Update in the parent class, which calls Likelihood
        mybayes.Suite.Update(self, data)

        # update the next level of the hierarchy
        for hypo in self.Values():
            hypo.Update(data)

    def Likelihood(self, data, hypo):
        dirichlet = hypo

        # draw sample Likelihoods from the hypothetical Dirichlet dist
        # and add them up
        like = 0
        for _ in range(self.iters):
            like += dirichlet.Likelihood(data)

        # correct for the number of ways the observed species
        # might have been chosen from all species
        m = len(data)
        like *= mybayes.BinomialCoef(dirichlet.n, m)

        return like

    def DistN(self):
        pmf = mybayes.Pmf()
        for hypo, prob in self.Items():
            pmf.Set(hypo.n, prob)
        return pmf


class Species2(object):
    def __init__(self, ns, conc=1, iters=1000):
        self.ns = ns
        self.conc = conc
        self.probs = numpy.ones(len(ns), dtype=numpy.float)
        self.params = numpy.ones(self.ns[-1], dtype=numpy.float) * conc
        self.iters = iters
        self.num_reads = 0
        self.m = 0

    def Preload(self, data):
        m = len(data)
        singletons = data.count(1)
        num = m - singletons
        print(m, singletons, num)
        addend = numpy.ones(num, dtype=numpy.float) * 1
        print(len(addend))
        print(len(self.params[singletons:m]))
        self.params[singletons:m] += addend
        print("Preload", num)

    def Update(self, data):
        self.num_reads += sum(data)

        like = numpy.zeros(len(self.ns), dtype=numpy.float)
        for _ in range(self.iters):
            like += self.SampleLikelihood(data)

        self.probs *= like
        self.probs /= self.probs.sum()

        self.m = len(data)
        # self.params[:self.m] += data * self.conc
        self.params[: self.m] += data

    def SampleLikelihood(self, data):
        gammas = numpy.random.gamma(self.params)

        m = len(data)
        row = gammas[:m]
        col = numpy.cumsum(gammas)

        log_likes = []
        for n in self.ns:
            ps = row / col[n - 1]
            terms = numpy.log(ps) * data
            log_like = terms.sum()
            log_likes.append(log_like)

        log_likes -= numpy.max(log_likes)
        likes = numpy.exp(log_likes)

        coefs = [mybayes.BinomialCoef(n, m) for n in self.ns]
        likes *= coefs

        return likes

    def DistN(self):
        pmf = mybayes.MakePmfFromItems(zip(self.ns, self.probs))
        return pmf

    def RandomN(self):
        return self.DistN().Random()

    def DistQ(self, iters=100):
        cdf_n = self.DistN().MakeCdf()
        sample_n = cdf_n.Sample(iters)

        pmf = mybayes.Pmf()
        for n in sample_n:
            q = self.RandomQ(n)
            pmf.Incr(q)

        pmf.Normalize()
        return pmf

    def RandomQ(self, n):
        # generate random prevalences
        dirichlet = mybayes.Dirichlet(n, conc=self.conc)
        prevalences = dirichlet.Random()

        # generate a simulated sample
        pmf = mybayes.MakePmfFromItems(enumerate(prevalences))
        cdf = pmf.MakeCdf()
        sample = cdf.Sample(self.num_reads)
        seen = set(sample)

        # add up the prevalence of unseen species
        q = 0
        for species, prev in enumerate(prevalences):
            if species not in seen:
                q += prev

        return q

    def MarginalBeta(self, n, index):
        alpha0 = self.params[:n].sum()
        alpha = self.params[index]
        return mybayes.Beta(alpha, alpha0 - alpha)

    def DistOfPrevalence(self, index):
        metapmf = mybayes.Pmf()

        for n, prob in zip(self.ns, self.probs):
            beta = self.MarginalBeta(n, index)
            pmf = beta.MakePmf()
            metapmf.Set(pmf, prob)

        mix = mybayes.MakeMixture(metapmf)
        return metapmf, mix

    def SamplePosterior(self):
        n = self.RandomN()
        prevalences = self.SamplePrevalences(n)
        # print('Peeking at n_cheat')
        # n = n_cheat

        return n, prevalences

    def SamplePrevalences(self, n):
        if n == 1:
            return [1.0]

        q_desired = self.RandomQ(n)
        q_desired = max(q_desired, 1e-6)

        params = self.Unbias(n, self.m, q_desired)
        gammas = numpy.random.gamma(params)
        gammas /= gammas.sum()
        return gammas

    def Unbias(self, n, m, q_desired):
        params = self.params[:n].copy()

        if n == m:
            return params

        x = sum(params[:m])
        y = sum(params[m:])
        a = x + y
        # print(x, y, a, x/a, y/a)

        g = q_desired * a / y
        f = (a - g * y) / x
        params[:m] *= f
        params[m:] *= g

        return params


class Species3(Species2):
    def Update(self, data):

        # sample the likelihoods and add them up
        like = numpy.zeros(len(self.ns), dtype=numpy.float)
        for _ in range(self.iters):
            like += self.SampleLikelihood(data)

        self.probs *= like
        self.probs /= self.probs.sum()

        m = len(data)
        self.params[:m] += data

    def SampleLikelihood(self, data):
        # get a random sample
        gammas = numpy.random.gamma(self.params)

        # row is just the first m elements of gammas
        m = len(data)
        row = gammas[:m]

        # col is the cumulative sum of gammas
        col = numpy.cumsum(gammas)[self.ns[0] - 1 :]

        # each row of the array is a set of ps, normalized
        # for each hypothetical value of n
        array = row / col[:, numpy.newaxis]

        # computing the multinomial PDF under a log transform
        # take the log of the ps and multiply by the data
        terms = numpy.log(array) * data

        # add up the rows
        log_likes = terms.sum(axis=1)

        # before exponentiating, scale into a reasonable range
        log_likes -= numpy.max(log_likes)
        likes = numpy.exp(log_likes)

        # correct for the number of ways we could see m species
        # out of a possible n
        coefs = [mybayes.BinomialCoef(n, m) for n in self.ns]
        likes *= coefs

        return likes


class Species4(Species):
    def Update(self, data):
        m = len(data)

        # loop through the species and update one at a time
        for i in range(m):
            one = numpy.zeros(i + 1)
            one[i] = data[i]

            # call the parent class
            Species.Update(self, one)

    def Likelihood(self, data, hypo):
        dirichlet = hypo
        like = 0
        for _ in range(self.iters):
            like += dirichlet.Likelihood(data)

        # correct for the number of unseen species the new one
        # could have been
        m = len(data)
        num_unseen = dirichlet.n - m + 1
        like *= num_unseen

        return like


class Species5(Species2):
    def Update(self, data):
        # loop through the species and update one at a time
        m = len(data)
        for i in range(m):
            self.UpdateOne(i + 1, data[i])
            self.params[i] += data[i]

    def UpdateOne(self, i, count):
        # how many species have we seen so far
        self.m = i

        # how many reads have we seen
        self.num_reads += count

        if self.iters == 0:
            return

        # sample the likelihoods and add them up
        likes = numpy.zeros(len(self.ns), dtype=numpy.float)
        for _ in range(self.iters):
            likes += self.SampleLikelihood(i, count)

        # correct for the number of unseen species the new one
        # could have been
        unseen_species = [n - i + 1 for n in self.ns]
        likes *= unseen_species

        # multiply the priors by the likelihoods and renormalize
        self.probs *= likes
        self.probs /= self.probs.sum()

    def SampleLikelihood(self, i, count):
        # get a random sample of p
        gammas = numpy.random.gamma(self.params)

        # sums is the cumulative sum of p, for each value of n
        sums = numpy.cumsum(gammas)[self.ns[0] - 1 :]

        # get p for the mth species, for each value of n
        ps = gammas[i - 1] / sums
        log_likes = numpy.log(ps) * count

        # before exponentiating, scale into a reasonable range
        log_likes -= numpy.max(log_likes)
        likes = numpy.exp(log_likes)

        return likes


def MakePosterior(constructor, data, ns, conc=1, iters=1000):
    suite = constructor(ns, conc=conc, iters=iters)

    # print(constructor.__name__)
    start = time.time()
    suite.Update(data)
    end = time.time()
    print("Processing time", end - start)

    return suite


def PlotAllVersions():
    data = [1, 2, 3]
    m = len(data)
    n = 20
    ns = range(m, n)

    for constructor in [Species, Species2, Species3, Species4, Species5]:
        suite = MakePosterior(constructor, data, ns)
        pmf = suite.DistN()
        pmf.name = "%s" % (constructor.__name__)
        myplots.Pmf(pmf)

    myplots.Save(root="species3", xlabel="Number of species", ylabel="Prob")


def PlotMedium():
    data = [1, 1, 1, 1, 2, 3, 5, 9]
    m = len(data)
    n = 20
    ns = range(m, n)

    for constructor in [Species, Species2, Species3, Species4, Species5]:
        suite = MakePosterior(constructor, data, ns)
        pmf = suite.DistN()
        pmf.name = "%s" % (constructor.__name__)
        myplots.Pmf(pmf)

    myplots.Show()


def SimpleDirichletExample():
    myplots.Clf()
    myplots.PrePlot(3)

    names = ["lions", "tigers", "bears"]
    data = [3, 2, 1]

    dirichlet = mybayes.Dirichlet(3)
    for i in range(3):
        beta = dirichlet.MarginalBeta(i)
        print("mean", names[i], beta.Mean())

    dirichlet.Update(data)
    for i in range(3):
        beta = dirichlet.MarginalBeta(i)
        print("mean", names[i], beta.Mean())

        pmf = beta.MakePmf(name=names[i])
        myplots.Pmf(pmf)

    myplots.Save(
        root="species1",
        xlabel="Prevalence",
        ylabel="Prob",
        formats=["png"],
    )


def HierarchicalExample():
    ns = range(3, 30)
    suite = Species(ns, iters=8000)

    data = [3, 2, 1]
    suite.Update(data)

    myplots.Clf()
    myplots.PrePlot(num=1)

    pmf = suite.DistN()
    myplots.Pmf(pmf)
    myplots.Save(
        root="species2",
        xlabel="Number of species",
        ylabel="Prob",
        formats=["png"],
    )


def CompareHierarchicalExample():
    data = [3, 2, 1]
    m = len(data)
    n = 30
    ns = range(m, n)

    constructors = [Species, Species5]
    iters = [1000, 100]

    for constructor, iters in zip(constructors, iters):
        suite = MakePosterior(constructor, data, ns, iters)
        pmf = suite.DistN()
        pmf.name = "%s" % (constructor.__name__)
        myplots.Pmf(pmf)

    myplots.Show()


def ProcessSubjects(codes):
    myplots.Clf()
    myplots.PrePlot(len(codes))

    subjects = ReadRarefactedData()
    pmfs = []
    for code in codes:
        subject = subjects[code]

        subject.Process()
        pmf = subject.suite.DistN()
        pmf.name = subject.code
        myplots.Pmf(pmf)

        pmfs.append(pmf)

    print("ProbGreater", mybayes.PmfProbGreater(pmfs[0], pmfs[1]))
    print("ProbLess", mybayes.PmfProbLess(pmfs[0], pmfs[1]))

    myplots.Save(
        root="species4",
        xlabel="Number of species",
        ylabel="Prob",
        formats=["png"],
    )


def RunSubject(code, conc=1, high=500):
    subjects = JoinSubjects()
    subject = subjects[code]

    subject.Process(conc=conc, high=high, iters=300)
    subject.MakeQuickPrediction()

    PrintSummary(subject)
    actual_l = subject.total_species - subject.num_species
    cdf_l = subject.DistL().MakeCdf()
    PrintPrediction(cdf_l, actual_l)

    subject.MakeFigures()

    num_reads = 400
    curves = subject.RunSimulations(100, num_reads)
    root = "species-rare-%s" % subject.code
    PlotCurves(curves, root=root)

    num_reads = 800
    curves = subject.RunSimulations(500, num_reads)
    ks = [100, 200, 400, 800]
    cdfs = MakeConditionals(curves, ks)
    root = "species-cond-%s" % subject.code
    PlotConditionals(cdfs, root=root)

    num_reads = 1000
    curves = subject.RunSimulations(500, num_reads, frac_flag=True)
    ks = [10, 100, 200, 400, 600, 800, 1000]
    cdfs = MakeFracCdfs(curves, ks)
    root = "species-frac-%s" % subject.code
    PlotFracCdfs(cdfs, root=root)


def PrintSummary(subject):
    print(subject.code)
    print("found %d species in %d reads" % (subject.num_species, subject.num_reads))

    print("total %d species in %d reads" % (subject.total_species, subject.total_reads))

    cdf = subject.suite.DistN().MakeCdf()
    print("n")
    PrintPrediction(cdf, "unknown")


def PrintPrediction(cdf, actual):
    median = cdf.Percentile(50)
    low, high = cdf.CredibleInterval(75)

    print("predicted %0.2f (%0.2f %0.2f)" % (median, low, high))
    print("actual", actual)


def RandomSeed(x):
    random.seed(x)
    numpy.random.seed(x)


def GenerateFakeSample(n, r, tr, conc=1):
    # generate random prevalences
    dirichlet = mybayes.Dirichlet(n, conc=conc)
    prevalences = dirichlet.Random()
    prevalences.sort()

    # generate a simulated sample
    pmf = mybayes.MakePmfFromItems(enumerate(prevalences))
    cdf = pmf.MakeCdf()
    sample = cdf.Sample(tr)

    # collect the species counts
    hist = mybayes.MakeHistFromList(sample)

    # extract a subset of the data
    if tr > r:
        random.shuffle(sample)
        subsample = sample[:r]
        subhist = mybayes.MakeHistFromList(subsample)
    else:
        subhist = hist

    # add up the prevalence of unseen species
    prev_unseen = 0
    for species, prev in enumerate(prevalences):
        if species not in subhist:
            prev_unseen += prev

    return hist, subhist, prev_unseen


def PlotActualPrevalences():
    # read data
    subject_map, _ = ReadCompleteDataset()

    # for subjects with more than 50 species,
    # PMF of max prevalence, and PMF of max prevalence
    # generated by a simulation
    pmf_actual = mybayes.Pmf()
    pmf_sim = mybayes.Pmf()

    # concentration parameter used in the simulation
    conc = 0.06

    for code, subject in subject_map.items():
        prevalences = subject.GetPrevalences()
        m = len(prevalences)
        if m < 2:
            continue

        actual_max = max(prevalences)
        print(code, m, actual_max)

        # incr the PMFs
        if m > 50:
            pmf_actual.Incr(actual_max)
            pmf_sim.Incr(SimulateMaxPrev(m, conc))

    # plot CDFs for the actual and simulated max prevalence
    cdf_actual = pmf_actual.MakeCdf(name="actual")
    cdf_sim = pmf_sim.MakeCdf(name="sim")

    myplots.Cdfs([cdf_actual, cdf_sim])
    myplots.Show()


def ScatterPrevalences(ms, actual):
    for conc in [1, 0.5, 0.2, 0.1]:
        expected = [ExpectedMaxPrev(m, conc) for m in ms]
        myplots.Plot(ms, expected)

    myplots.Scatter(ms, actual)
    myplots.Show(xscale="log")


def SimulateMaxPrev(m, conc=1):
    dirichlet = mybayes.Dirichlet(m, conc)
    prevalences = dirichlet.Random()
    return max(prevalences)


def ExpectedMaxPrev(m, conc=1, iters=100):
    dirichlet = mybayes.Dirichlet(m, conc)

    t = []
    for _ in range(iters):
        prevalences = dirichlet.Random()
        t.append(max(prevalences))

    return numpy.mean(t)


class Calibrator(object):
    def __init__(self, conc=0.1):
        self.conc = conc
        self.ps = range(10, 100, 10)
        self.total_n = numpy.zeros(len(self.ps))
        self.total_q = numpy.zeros(len(self.ps))
        self.total_l = numpy.zeros(len(self.ps))
        self.n_seq = []
        self.q_seq = []
        self.l_seq = []

    def Calibrate(self, num_runs=100, n_low=30, n_high=400, r=400, tr=1200):
        for seed in range(num_runs):
            self.RunCalibration(seed, n_low, n_high, r, tr)

        self.total_n *= 100.0 / num_runs
        self.total_q *= 100.0 / num_runs
        self.total_l *= 100.0 / num_runs

    def Validate(self, num_runs=100, clean_param=0):
        subject_map, _ = ReadCompleteDataset(clean_param=clean_param)

        i = 0
        for match in subject_map.values():
            if match.num_reads < 400:
                continue
            num_reads = 100

            print("Validate", match.code)
            subject = match.Resample(num_reads)
            subject.Match(match)

            n_actual = None
            q_actual = subject.prev_unseen
            l_actual = subject.total_species - subject.num_species
            self.RunSubject(subject, n_actual, q_actual, l_actual)

            i += 1
            if i == num_runs:
                break

        self.total_n *= 100.0 / num_runs
        self.total_q *= 100.0 / num_runs
        self.total_l *= 100.0 / num_runs

    def PlotN(self, root="species-n"):
        xs, ys = zip(*self.n_seq)
        if None in xs:
            return

        high = max(xs + ys)

        myplots.Plot([0, high], [0, high], color="gray")
        myplots.Scatter(xs, ys)
        myplots.Save(root=root, xlabel="Actual n", ylabel="Predicted")

    def PlotQ(self, root="species-q"):
        myplots.Plot([0, 0.2], [0, 0.2], color="gray")
        xs, ys = zip(*self.q_seq)
        myplots.Scatter(xs, ys)
        myplots.Save(root=root, xlabel="Actual q", ylabel="Predicted")

    def PlotL(self, root="species-n"):
        myplots.Plot([0, 20], [0, 20], color="gray")
        xs, ys = zip(*self.l_seq)
        myplots.Scatter(xs, ys)
        myplots.Save(root=root, xlabel="Actual l", ylabel="Predicted")

    def PlotCalibrationCurves(self, root="species5"):
        print(self.total_n)
        print(self.total_q)
        print(self.total_l)

        myplots.Plot([0, 100], [0, 100], color="gray", alpha=0.2)

        if self.total_n[0] >= 0:
            myplots.Plot(self.ps, self.total_n, label="n")

        myplots.Plot(self.ps, self.total_q, label="q")
        myplots.Plot(self.ps, self.total_l, label="l")

        myplots.Save(
            root=root,
            axis=[0, 100, 0, 100],
            xlabel="Ideal percentages",
            ylabel="Predictive distributions",
            formats=["png"],
        )

    def RunCalibration(self, seed, n_low, n_high, r, tr):
        # generate a random number of species and their prevalences
        # (from a Dirichlet distribution with alpha_i = conc for all i)
        RandomSeed(seed)
        n_actual = random.randrange(n_low, n_high + 1)

        hist, subhist, q_actual = GenerateFakeSample(n_actual, r, tr, self.conc)

        l_actual = len(hist) - len(subhist)
        print("Run low, high, conc", n_low, n_high, self.conc)
        print("Run r, tr", r, tr)
        print("Run n, q, l", n_actual, q_actual, l_actual)

        # extract the data
        data = [count for species, count in subhist.Items()]
        data.sort()
        print("data", data)

        # make a Subject and process
        subject = Subject("simulated")
        subject.num_reads = r
        subject.total_reads = tr

        for species, count in subhist.Items():
            subject.Add(species, count)
        subject.Done()

        self.RunSubject(subject, n_actual, q_actual, l_actual)

    def RunSubject(self, subject, n_actual, q_actual, l_actual):
        # process and make prediction
        subject.Process(conc=self.conc, iters=100)
        subject.MakeQuickPrediction()

        # extract the posterior suite
        suite = subject.suite

        # check the distribution of n
        pmf_n = suite.DistN()
        print("n")
        self.total_n += self.CheckDistribution(pmf_n, n_actual, self.n_seq)

        # check the distribution of q
        pmf_q = suite.DistQ()
        print("q")
        self.total_q += self.CheckDistribution(pmf_q, q_actual, self.q_seq)

        # check the distribution of additional species
        pmf_l = subject.DistL()
        print("l")
        self.total_l += self.CheckDistribution(pmf_l, l_actual, self.l_seq)

    def CheckDistribution(self, pmf, actual, seq):
        mean = pmf.Mean()
        seq.append((actual, mean))

        cdf = pmf.MakeCdf()
        PrintPrediction(cdf, actual)

        sv = ScoreVector(cdf, self.ps, actual)
        return sv


def ScoreVector(cdf, ps, actual):
    scores = []
    for p in ps:
        low, high = cdf.CredibleInterval(p)
        score = Score(low, high, actual)
        scores.append(score)

    return numpy.array(scores)


def Score(low, high, n):
    if n is None:
        return -1
    if low < n < high:
        return 1
    if n == low or n == high:
        return 0.5
    else:
        return 0


def FakeSubject(n=300, conc=0.1, num_reads=400, prevalences=None):
    # generate random prevalences
    if prevalences is None:
        dirichlet = mybayes.Dirichlet(n, conc=conc)
        prevalences = dirichlet.Random()
        prevalences.sort()

    # generate a simulated sample
    pmf = mybayes.MakePmfFromItems(enumerate(prevalences))
    cdf = pmf.MakeCdf()
    sample = cdf.Sample(num_reads)

    # collect the species counts
    hist = mybayes.MakeHistFromList(sample)

    # extract the data
    data = [count for species, count in hist.Items()]
    data.sort()

    # make a Subject and process
    subject = Subject("simulated")

    for species, count in hist.Items():
        subject.Add(species, count)
    subject.Done()

    return subject


def PlotSubjectCdf(code=None, clean_param=0):
    subject_map, uber_subject = ReadCompleteDataset(clean_param=clean_param)

    if code is None:
        subjects = subject_map.values()
        subject = random.choice(subjects)
        code = subject.code
    elif code == "uber":
        subject = uber_subject
    else:
        subject = subject_map[code]

    print(subject.code)

    m = subject.GetM()

    subject.Process(high=m, conc=0.1, iters=0)
    print(subject.suite.params[:m])

    # plot the cdf
    options = dict(linewidth=3, color="blue", alpha=0.5)
    cdf = subject.MakeCdf()
    myplots.Cdf(cdf, **options)

    options = dict(linewidth=1, color="green", alpha=0.5)

    # generate fake subjects and plot their CDFs
    for _ in range(10):
        prevalences = subject.suite.SamplePrevalences(m)
        fake = FakeSubject(prevalences=prevalences)
        cdf = fake.MakeCdf()
        myplots.Cdf(cdf, **options)

    root = "species-cdf-%s" % code
    myplots.Save(
        root=root,
        xlabel="rank",
        ylabel="CDF",
        xscale="log",
        formats=["png"],
    )


def RunCalibration(flag="cal", num_runs=100, clean_param=50):
    cal = Calibrator(conc=0.1)

    if flag == "val":
        cal.Validate(num_runs=num_runs, clean_param=clean_param)
    else:
        cal.Calibrate(num_runs=num_runs)

    cal.PlotN(root="species-n-%s" % flag)
    cal.PlotQ(root="species-q-%s" % flag)
    cal.PlotL(root="species-l-%s" % flag)
    cal.PlotCalibrationCurves(root="species5-%s" % flag)


def RunTests():
    RunCalibration(flag="val")
    RunCalibration(flag="cal")
    PlotSubjectCdf("B1558.G", clean_param=50)
    PlotSubjectCdf(None)


def main(script):
    RandomSeed(17)
    RunSubject("B1242", conc=1, high=100)
    RandomSeed(17)
    SimpleDirichletExample()
    RandomSeed(17)
    HierarchicalExample()


if __name__ == "__main__":
    main(*sys.argv)