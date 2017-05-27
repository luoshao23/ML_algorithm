import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class matchrow(object):
    """docstring for matchrow"""

    def __init__(self, row, allnum=False):
        if allnum:
            self.data = [float(row[i]) for i in xrange(len(row) - 1)]
        else:
            self.data = row[:-1]
        self.match = int(row[-1])


def loadmatch(f, allnum=False):
    rows = []
    with open(f) as of:
        for line in of:
            rows.append(matchrow(line.strip().split(','), allnum))
    return rows


def plotagematches(rows):
    xdm, ydm = [r.data[0] for r in rows if r.match == 1],\
        [r.data[1] for r in rows if r.match == 1]
    xdn, ydn = [r.data[0] for r in rows if r.match == 0],\
        [r.data[1] for r in rows if r.match == 0]

    plt.plot(xdm, ydm, 'bo')
    plt.plot(xdn, ydn, 'b+')
    plt.savefig('aa')
