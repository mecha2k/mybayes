from __future__ import print_function

import csv


def read_csv(filename, constructor):
    fp = open(filename)
    reader = csv.reader(fp)

    header = next(reader)
    names = [s.lower() for s in header]

    objs = [make_object(t, names, constructor) for t in reader]
    fp.close()

    return objs


def write_csv(filename, header, data):
    fp = open(filename, "w")
    writer = csv.writer(fp)
    writer.writerow(header)

    for t in data:
        writer.writerow(t)
    fp.close()


def print_cols(cols):
    for i, col in enumerate(cols):
        print(i, col[0], col[1])


def make_col_dict(cols, names):
    col_dict = {}
    for name, col in zip(names, cols):
        col_dict[name] = col
    return col_dict


def make_object(row, names, constructor):
    obj = constructor()
    for name, val in zip(names, row):
        func = constructor.convert.get(name, int)
        try:
            val = func(val)
        except:
            pass
        setattr(obj, name, val)
    obj.clean()
    return obj