from mybayes import Pmf, Suite


def cookie():
    pmf = Pmf()
    pmf.Set("Bowl 1", 0.5)
    pmf.Set("Bowl 2", 0.5)
    pmf.Mult("Bowl 1", 0.75)
    pmf.Mult("Bowl 2", 0.5)
    pmf.Normalize()
    print(pmf.Prob("Bowl 1"))


class Cookie(Pmf):
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Keys():
            like = self.Likelihood(hypo, data)
            self.Mult(hypo, like)
        self.Normalize()

    mixes = {
        "bowl1": dict(vanilla=0.75, chocolate=0.25),
        "bowl2": {"vanilla": 0.5, "chocolate": 0.5},
    }

    def Likelihood(self, hypo, data):
        mix = self.mixes[hypo]
        like = mix[data]
        return like


class Monty(Pmf):
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Likelihood(self, hypo, data):
        if hypo == data:
            return 0
        elif hypo == "A":
            return 0.5
        else:
            return 1

    def Update(self, data):
        for hypo in self.Keys():
            like = self.Likelihood(hypo, data)
            self.Mult(hypo, like)
        self.Normalize()


class Monty1(Suite):
    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        elif hypo == "A":
            return 0.5
        else:
            return 1


class MandM(Suite):
    mix94 = dict(brown=30, yellow=20, red=20, green=10, orange=10, tan=10, blue=0)
    mix96 = dict(blue=24, green=20, orange=16, yellow=14, red=13, brown=13, tan=0)

    hypoA = dict(bag1=mix94, bag2=mix96)
    hypoB = dict(bag1=mix96, bag2=mix94)

    hypotheses = dict(A=hypoA, B=hypoB)

    def Likelihood(self, data, hypo):
        bag, color = data
        mix = self.hypotheses[hypo][bag]
        like = mix[color]
        return like


def main():
    cookie()
    hypos = ["bowl1", "bowl2"]
    pmf = Cookie(hypos)
    pmf.Update("vanilla")

    for hypo, prob in pmf.Items():
        print(hypo, prob)

    hypos = "ABC"
    pmf = Monty(hypos)

    data = "B"
    pmf.Update(data)

    for hypo, prob in pmf.Items():
        print(hypo, prob)

    suite = Monty1("ABC")
    suite.Update("B")
    suite.Print()
    print(suite.Items())
    print(sorted(suite.Items()))

    suite = MandM("AB")
    suite.Update(("bag2", "yellow"))
    suite.Update(("bag1", "green"))
    suite.Update(("bag1", "orange"))
    suite.Print()


if __name__ == "__main__":
    main()
