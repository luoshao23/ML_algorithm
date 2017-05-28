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
    plt.figure()
    plt.plot(xdm, ydm, 'bo')
    plt.plot(xdn, ydn, 'b+')
    plt.savefig('aa')


def lineartrain(rows):
    averages = {}
    counts = {}

    for row in rows:
        cl = row.match

        averages.setdefault(cl, [0.0] * len(row.data))
        counts.setdefault(cl, 0)

        for i in xrange(len(row.data)):
            averages[cl][i] += float(row.data[i])

        counts[cl] += 1

    for cl, avg in averages.items():
        for i in xrange(len(avg)):
            avg[i] /= counts[cl]

    return averages


def dotproduct(v1, v2):
    return sum([v1[i] * v2[i] for i in xrange(len(v1))])


def dpclassify(point, avgs):
    b = (dotproduct(avgs[1], avgs[1]) - dotproduct(avgs[0], avgs[0])) / 2
    y = dotproduct(point, avgs[0]) - dotproduct(point, avgs[1]) + b
    if y > 0:
        return 0
    else:
        return 1


def yesno(v):
    if v == 'yes':
        return 1
    elif v == 'no':
        return -1
    else:
        return 0


def matchcount(interest1, interest2):
    l1 = interest1.split(':')
    l2 = interest2.split(':')
    x = 0
    for v in l1:
        if v in l2:
            x += 1
    return x

def milesdistance(a1, a2):
    return 0

def loadnumerical():
    oldrows = loadmatch('matchmaker.csv')
    newrows = []
    for row in oldrows:
        d = row.data
        data = [float(d[0]), yesno(d[1]), yesno(d[2]),float(d[5]), yesno(d[6]),yesno(d[7]),matchcount(d[3],d[8]),milesdistance(d[4],d[9]),row.match]
        newrows.append(matchrow(data))
    return newrows

def scaledata(rows):
    low = [99999999.0]*len(rows[0].data)
    high = [-99999999.0]*len(rows[0].data)

    for row in rows:
        d = row.data
        for i in xrange(len(d)):
            if d[i]<low[i]: low[i] = d[i]
            if d[i]>high[i]: high[i] = d[i]

    def scaleinput(d):
        return [(d.data[i]-low[i])/(high[i] - low[i]) for i in xrange(len(low))]

    newrows = [matchrow(scaleinput(row.data)+[row.match]) for row in rows]

    return newrows, scaleinput
