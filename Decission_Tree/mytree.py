from math import log
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np
from pandas import DataFrame

# my_data = [['slashdot', 'USA', 'yes', 18, 213.2, 'None'],
#            ['google', 'France', 'yes', 23, 121.2, 'Premium'],
#            ['digg', 'USA', 'yes', 24, 21.32, 'Basic'],
#            ['kiwitobes', 'France', 'yes', 23, 1.2, 'Basic'],
#            ['google', 'UK', 'no', 21, .2, 'Premium'],
#            ['(direct)', 'New Zealand', 'no', 12, 71.2, 'None'],
#            ['(direct)', 'UK', 'no', 21, -21.2, 'Basic'],
#            ['google', 'USA', 'no', 24, 241.2, 'Premium'],
#            ['slashdot', 'France', 'yes', 19, 20, 'None'],
#            ['digg', 'USA', 'no', 18, 1.0, 'None'],
#            ['google', 'UK', 'no', 18, 2, 'None'],
#            ['kiwitobes', 'UK', 'no', 19, 44, 'None'],
#            ['digg', 'New Zealand', 'yes', 12, 27, 'Basic'],
#            ['slashdot', 'UK', 'no', 21, 86, 'None'],
#            ['google', 'UK', 'yes', 18, 2, 'Basic'],
#            ['kiwitobes', 'France', 'yes', 19, 0.0, 'Basic']]

my_data = [[213.2, 'None'],
           [121.2, 'Premium'],
           [21.32, 'Basic'],
           [1.2, 'Basic'],
           [.2, 'Premium'],
           [71.2, 'None'],
           [-21.2, 'Basic'],
           [241.2, 'Premium'],
           [20, 'None'],
           [1.0, 'None'],
           [2, 'None'],
           [44, 'None'],
           [27, 'Basic'],
           [86, 'None'],
           [2, 'Basic'],
           [0.0, 'Basic']]
data = np.array(DataFrame(my_data))
# my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
#            ['google', 'France', 'yes', 23, 'None'],
#            ['digg', 'USA', 'yes', 24, 'None'],
#            ['kiwitobes', 'France', 'yes', 23, 'None'],
#            ['google', 'UK', 'no', 21, 'None'],
#            ['(direct)', 'New Zealand', 'no', 12, 'None'],
#            ['(direct)', 'UK', 'no', 21, 'None'],
#            ['google', 'USA', 'no', 24, 'None'],
#            ['slashdot', 'France', 'yes', 19, 'None'],
#            ['digg', 'USA', 'no', 18, 'None'],
#            ['google', 'UK', 'no', 18, 'None'],
#            ['kiwitobes', 'UK', 'no', 19, 'None'],
#            ['digg', 'New Zealand', 'yes', 12, 'None'],
#            ['slashdot', 'UK', 'no', 21, 'None'],
#            ['google', 'UK', 'yes', 18, 'None'],
#            ['kiwitobes', 'France', 'yes', 19, 'None']]


class decisionnode(object):
    """docstring for decisionnode"""

    def __init__(self, col=-1, value=None, tb=None, fb=None, results=None):
        self.col = col
        self.value = value
        self.tb = tb
        self.fb = fb
        self.results = results


def divideset(rows, column, value):
    split_func = None

    if isinstance(value, int) or isinstance(value, float):
        split_func = lambda x: x[column] >= value
    else:
        split_func = lambda x: x[column] == value
    set1 = [row for row in rows if split_func(row)]
    set2 = [row for row in rows if not split_func(row)]

    return (set1, set2)


def uniquecounts(rows, col=-1):
    results = {}
    for row in rows:
        r = row[col]
        results.setdefault(r, 0)
        results[r] += 1
    return results


def entropy(rows):
    ent = 0.0
    log2 = lambda x: log(x) / log(2)
    res = uniquecounts(rows)
    for k, v in res.items():
        p = float(v) / len(rows)
        ent -= p * log2(p)
    return ent


def variance(rows, col=-1):
    if len(rows) == 0:
        return 0
    data = [float(row[col]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean)**2 for d in data]) / len(data)
    return variance


def buildtree(rows, scoref=entropy, min_len=2):
    if len(rows) <= min_len:
        return decisionnode(results=uniquecounts(rows))

    current_score = scoref(rows)
    if current_score <= 0.0:
        return decisionnode(results=uniquecounts(rows))

    best_gain = 0.0
    best_criteria = None
    best_set = None

    for col in xrange(len(rows[0]) - 1):
        if isinstance(rows[0][col], float):
            uniq_value = split_float(rows, col)
        else:
            uniq_value = uniquecounts(rows, col)
        for val in uniq_value.keys():
            set1, set2 = divideset(rows, col, val)
            p = float(len(set1)) / len(rows)
            gain = current_score - (p * scoref(set1) + (1 - p) * scoref(set2))
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, val)
                best_set = (set1, set2)

    if best_gain > 0:
        tbr = buildtree(best_set[0])
        fbr = buildtree(best_set[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1], tb=tbr, fb=fbr)
    else:
        return decisionnode(results=uniquecounts(rows))


def printtree(tree, indent='|', depth=0):
    if tree.results is not None:
        print str(tree.results) + '##'
    else:
        if isinstance(tree.value, int) or isinstance(tree.value, float):
            print 'depth %s ~ %s>=%s?' % (str(depth), str(tree.col), str(tree.value))
        else:
            print 'depth %s ~ %s is %s?' % (str(depth), str(tree.col), str(tree.value))
        print indent + 'T->'
        printtree(tree.tb, indent + ' |', depth + 1)
        print indent + 'F->'
        printtree(tree.fb, indent + ' |', depth + 1)


def split_float(rows, column):
    col = [row[column] for row in rows]
    val = np.median(col)
    return {val: 1}


def classify(obs, tree):
    pass


def prune(tree, mingain):
    pass


def mdclassify(obs, tree):
    pass
