from __future__ import print_function

import logging
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas
import chart_studio.plotly as plotly

# customize some matplotlib attributes
# matplotlib.rc('figure', figsize=(4, 3))
# matplotlib.rc('font', size=14.0)
# matplotlib.rc('axes', labelsize=22.0, titlesize=22.0)
# matplotlib.rc('legend', fontsize=20.0)
# matplotlib.rc('xtick.major', size=6.0)
# matplotlib.rc('xtick.minor', size=3.0)
# matplotlib.rc('ytick.major', size=6.0)
# matplotlib.rc('ytick.minor', size=3.0)


class _Brewer(object):
    color_iter = None

    colors = [
        "#081D58",
        "#253494",
        "#225EA8",
        "#1D91C0",
        "#41B6C4",
        "#7FCDBB",
        "#C7E9B4",
        "#EDF8B1",
        "#FFFFD9",
    ]

    which_colors = [
        [],
        [1],
        [1, 3],
        [0, 2, 4],
        [0, 2, 4, 6],
        [0, 2, 3, 5, 6],
        [0, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6],
    ]

    @classmethod
    def Colors(cls):
        return cls.colors

    @classmethod
    def ColorGenerator(cls, n):
        for i in cls.which_colors[n]:
            yield cls.colors[i]

    @classmethod
    def InitializeIter(cls, num):
        cls.color_iter = cls.ColorGenerator(num)

    @classmethod
    def ClearIter(cls):
        cls.color_iter = None

    @classmethod
    def GetIter(cls):
        if cls.color_iter is None:
            cls.InitializeIter(7)

        return cls.color_iter


def PrePlot(num=None, rows=None, cols=None):
    if num:
        _Brewer.InitializeIter(num)

    if rows is None and cols is None:
        return

    if rows is not None and cols is None:
        cols = 1

    if cols is not None and rows is None:
        rows = 1

    # resize the image, depending on the number of rows and cols
    size_map = {
        (1, 1): (8, 6),
        (1, 2): (14, 6),
        (1, 3): (14, 6),
        (2, 2): (10, 10),
        (2, 3): (16, 10),
        (3, 1): (8, 10),
    }

    if (rows, cols) in size_map:
        fig = pyplot.gcf()
        fig.set_size_inches(*size_map[rows, cols])

    # create the first subplot
    if rows > 1 or cols > 1:
        pyplot.subplot(rows, cols, 1)
        global SUBPLOT_ROWS, SUBPLOT_COLS
        SUBPLOT_ROWS = rows
        SUBPLOT_COLS = cols


def SubPlot(plot_number, rows=None, cols=None):
    rows = rows or SUBPLOT_ROWS
    cols = cols or SUBPLOT_COLS
    pyplot.subplot(rows, cols, plot_number)


def _Underride(d, **options):
    if d is None:
        d = {}
    for key, val in options.items():
        d.setdefault(key, val)
    return d


def Clf():
    _Brewer.ClearIter()
    pyplot.clf()
    fig = pyplot.gcf()
    fig.set_size_inches(8, 6)


def Figure(**options):
    _Underride(options, figsize=(6, 8))
    pyplot.figure(**options)


def _UnderrideColor(options):
    if "color" in options:
        return options

    color_iter = _Brewer.GetIter()

    if color_iter:
        try:
            options["color"] = next(color_iter)
        except StopIteration:
            print("Warning: Brewer ran out of colors.")
            _Brewer.ClearIter()
    return options


def Plot(obj, ys=None, style="", **options):
    options = _UnderrideColor(options)
    label = getattr(obj, "name", "_nolegend_")
    options = _Underride(options, linewidth=3, alpha=0.8, label=label)

    xs = obj
    if ys is None:
        if hasattr(obj, "Render"):
            xs, ys = obj.Render()
        if isinstance(obj, pandas.Series):
            ys = obj.values
            xs = obj.index

    if ys is None:
        pyplot.plot(xs, style, **options)
    else:
        pyplot.plot(xs, ys, style, **options)


def FillBetween(xs, y1, y2=None, where=None, **options):
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.5)
    pyplot.fill_between(xs, y1, y2, where, **options)


def Bar(xs, ys, **options):
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.6)
    pyplot.bar(xs, ys, **options)


def Scatter(xs, ys=None, **options):
    options = _Underride(options, color="blue", alpha=0.2, s=30, edgecolors="none")
    if ys is None and isinstance(xs, pandas.Series):
        ys = xs.values
        xs = xs.index
    pyplot.scatter(xs, ys, **options)


def HexBin(xs, ys, **options):
    options = _Underride(options, cmap=matplotlib.cm.Blues)
    pyplot.hexbin(xs, ys, **options)


def Pdf(pdf, **options):
    low, high = options.pop("low", None), options.pop("high", None)
    n = options.pop("n", 101)
    xs, ps = pdf.Render(low=low, high=high, n=n)
    options = _Underride(options, label=pdf.name)
    Plot(xs, ps, **options)


def Pdfs(pdfs, **options):
    for pdf in pdfs:
        Pdf(pdf, **options)


def Hist(hist, **options):
    # find the minimum distance between adjacent values
    xs, ys = hist.Render()

    if "width" not in options:
        try:
            options["width"] = 0.9 * np.diff(xs).min()
        except TypeError:
            logging.warning(
                "Hist: Can't compute bar width automatically."
                "Check for non-numeric types in Hist."
                "Or try providing width option."
            )

    options = _Underride(options, label=hist.name)
    options = _Underride(options, align="center")
    if options["align"] == "left":
        options["align"] = "edge"
    elif options["align"] == "right":
        options["align"] = "edge"
        options["width"] *= -1

    Bar(xs, ys, **options)


def Hists(hists, **options):
    for hist in hists:
        Hist(hist, **options)


def Pmf(pmf, **options):
    xs, ys = pmf.Render()
    low, high = min(xs), max(xs)

    width = options.pop("width", None)
    if width is None:
        try:
            width = np.diff(xs).min()
        except TypeError:
            logging.warning(
                "Pmf: Can't compute bar width automatically."
                "Check for non-numeric types in Pmf."
                "Or try providing width option."
            )
    points = []

    lastx = np.nan
    lasty = 0
    for x, y in zip(xs, ys):
        if (x - lastx) > 1e-5:
            points.append((lastx, 0))
            points.append((x, 0))

        points.append((x, lasty))
        points.append((x, y))
        points.append((x + width, y))

        lastx = x + width
        lasty = y
    points.append((lastx, 0))
    pxs, pys = zip(*points)

    align = options.pop("align", "center")
    if align == "center":
        pxs = np.array(pxs) - width / 2.0
    if align == "right":
        pxs = np.array(pxs) - width

    options = _Underride(options, label=pmf.name)
    Plot(pxs, pys, **options)


def Pmfs(pmfs, **options):
    for pmf in pmfs:
        Pmf(pmf, **options)


def Diff(t):
    diffs = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    return diffs


def Cdf(cdf, complement=False, transform=None, **options):
    xs, ps = cdf.Render()
    xs = np.asarray(xs)
    ps = np.asarray(ps)

    scale = dict(xscale="linear", yscale="linear")

    for s in ["xscale", "yscale"]:
        if s in options:
            scale[s] = options.pop(s)

    if transform == "exponential":
        complement = True
        scale["yscale"] = "log"

    if transform == "pareto":
        complement = True
        scale["yscale"] = "log"
        scale["xscale"] = "log"

    if complement:
        ps = [1.0 - p for p in ps]

    if transform == "weibull":
        xs = np.delete(xs, -1)
        ps = np.delete(ps, -1)
        ps = [-math.log(1.0 - p) for p in ps]
        scale["xscale"] = "log"
        scale["yscale"] = "log"

    if transform == "gumbel":
        xs = np.delete(xs, 0)
        ps = np.delete(ps, 0)
        ps = [-math.log(p) for p in ps]
        scale["yscale"] = "log"

    options = _Underride(options, label=cdf.name)
    Plot(xs, ps, **options)
    return scale


def Cdfs(cdfs, complement=False, transform=None, **options):
    for cdf in cdfs:
        Cdf(cdf, complement, transform, **options)


def Contour(obj, pcolor=False, contour=True, imshow=False, **options):
    try:
        d = obj.GetDict()
    except AttributeError:
        d = obj

    _Underride(options, linewidth=3, cmap=matplotlib.cm.Blues)

    xs, ys = zip(*d.keys())
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    X, Y = np.meshgrid(xs, ys)
    func = lambda x, y: d.get((x, y), 0)
    func = np.vectorize(func)
    Z = func(X, Y)

    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = pyplot.gca()
    axes.xaxis.set_major_formatter(x_formatter)

    if pcolor:
        pyplot.pcolormesh(X, Y, Z, **options)
    if contour:
        cs = pyplot.contour(X, Y, Z, **options)
        pyplot.clabel(cs, inline=1, fontsize=10)
    if imshow:
        extent = xs[0], xs[-1], ys[0], ys[-1]
        pyplot.imshow(Z, extent=extent, **options)


def Pcolor(xs, ys, zs, pcolor=True, contour=False, **options):
    _Underride(options, linewidth=3, cmap=matplotlib.cm.Blues)

    X, Y = np.meshgrid(xs, ys)
    Z = zs

    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = pyplot.gca()
    axes.xaxis.set_major_formatter(x_formatter)

    if pcolor:
        pyplot.pcolormesh(X, Y, Z, **options)

    if contour:
        cs = pyplot.contour(X, Y, Z, **options)
        pyplot.clabel(cs, inline=1, fontsize=10)


def Text(x, y, s, **options):
    options = _Underride(options, verticalalignment="top", horizontalalignment="left")
    pyplot.text(x, y, s, **options)


def Config(**options):
    names = [
        "title",
        "xlabel",
        "ylabel",
        "xscale",
        "yscale",
        "xticks",
        "yticks",
        "axis",
        "xlim",
        "ylim",
    ]

    for name in names:
        if name in options:
            getattr(pyplot, name)(options[name])

    # looks like this is not necessary: matplotlib understands text loc specs
    loc_dict = {
        "upper right": 1,
        "upper left": 2,
        "lower left": 3,
        "lower right": 4,
        "right": 5,
        "center left": 6,
        "center right": 7,
        "lower center": 8,
        "upper center": 9,
        "center": 10,
    }

    loc = options.get("loc", 0)
    # loc = loc_dict.get(loc, loc)

    legend = options.get("legend", True)
    if legend:
        pyplot.legend(loc=loc)


def Show(**options):
    clf = options.pop("clf", True)
    Config(**options)
    pyplot.show()
    if clf:
        Clf()


def Plotly(**options):
    clf = options.pop("clf", True)
    Config(**options)

    url = plotly.plot(pyplot.gcf())
    if clf:
        Clf()
    return url


def Save(root=None, formats=None, **options):
    clf = options.pop("clf", True)
    Config(**options)

    if formats is None:
        formats = ["pdf", "eps"]

    try:
        formats.remove("plotly")
        Plotly(clf=False)
    except ValueError:
        pass

    if root:
        for fmt in formats:
            SaveFormat(root, fmt)
    if clf:
        Clf()


def SaveFormat(root, fmt="eps"):
    filename = "%s.%s" % (root, fmt)
    print("Writing", filename)
    pyplot.savefig(filename, format=fmt, dpi=300)


# provide aliases for calling functons with lower-case names
preplot = PrePlot
subplot = SubPlot
clf = Clf
figure = Figure
plot = Plot
scatter = Scatter
pmf = Pmf
pmfs = Pmfs
hist = Hist
hists = Hists
diff = Diff
cdf = Cdf
cdfs = Cdfs
contour = Contour
pcolor = Pcolor
config = Config
show = Show
save = Save


def main():
    color_iter = _Brewer.ColorGenerator(7)
    for color in color_iter:
        print(color)


if __name__ == "__main__":
    main()
