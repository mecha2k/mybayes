from mybayes import Suite, Pmf, MakeCdfFromPmf, Percentile
import myplots


def main():
    for i in range(0, 5):
        print(i)

    def save_ranking(*args, **kwargs):
        print(args)
        print(kwargs)

    save_ranking("ming", "alice", "tom", fourth="wilson", fifth="roy")


if __name__ == "__main__":
    main()