from math import log
from PIL import Image, ImageDraw

my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]


class decisionnode(object):
    """docstring for decisionnode"""

    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):

        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def divideset(rows, column, value):
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


def uniquecounts(rows):
    results = {}
    for row in rows:
        r = row[-1]
        results.setdefault(r, 0)
        results[r] += 1
    return results


def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        imp += p1 * (1 - p1)
    return imp


def entropy(rows):
    total = len(rows)
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / total
        ent -= p * log2(p)
    return ent


def buildtree(rows, scoref=entropy):
    if len(rows) == 0:
        return decisionnode()
    current_score = scoref(rows)

    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in xrange(column_count):
        column_values = {}
        for row in rows:
            column_values.setdefault(row[col], 1)
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


def printtree(tree, indent=''):
    if tree.results != None:
        print str(tree.results)
    else:
        print '%s:%s?' % (str(tree.col), str(tree.value))
        print indent + 'T->'
        printtree(tree.tb, indent + '--')
        print indent + 'F->'
        printtree(tree.fb, indent + '--')


def getwidth(trees):
    if trees.tb is None and trees.fb is None:
        return 1
    else:
        return getwidth(trees.tb) + getwidth(trees.fb)


def getdepth(trees):
    if trees.tb is None and trees.fb is None:
        return 0
    else:
        return max(getdepth(trees.tb), getdepth(trees.fb)) + 1


def drawtrees(trees, jpeg='trees.jpg', widdelta=100, depdelta=100):
    w = getwidth(trees) * widdelta
    h = getdepth(trees) * depdelta + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, trees, w / 2, 20, widdelta, depdelta)
    img.save(jpeg, 'JPEG')


def drawnode(draw, trees, x, y, widdelta=100, depdelta=100):

    if trees.results is None:
        wf = getwidth(trees.fb) * widdelta
        wt = getwidth(trees.tb) * widdelta

        left = x - (wf + wt) / 2
        right = x + (wf + wt) / 2
        if isinstance(trees.value, int) or isinstance(trees.value, float):
            draw.text((x - 20, y - 10), '%s:>=%s?\n' %
                      (str(trees.col), str(trees.value)), (0, 0, 0))
        else:
            draw.text((x - 20, y - 10), '%s:==%s?\n' %
                      (str(trees.col), str(trees.value)), (0, 0, 0))

        draw.line((x, y, left + wf / 2, y + depdelta), fill=(255, 0, 0))
        draw.line((x, y, right - wt / 2, y + depdelta), fill=(255, 0, 0))

        drawnode(draw, trees.fb, left + wf / 2,
                 y + depdelta, widdelta, depdelta)
        drawnode(draw, trees.tb, right - wt / 2,
                 y + depdelta, widdelta, depdelta)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in trees.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))


def classify(obs, tree):
    if tree.results is not None:
        return tree.results
    else:
        v = obs[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
    return classify(obs, branch)


def prune(tree, mingain):
    if tree.tb.results is None:
        prune(tree.tb, mingain)
    if tree.fb.results is None:
        prune(tree.fb, mingain)

    if tree.tb.results is not None and tree.fb.results is not None:
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb)) / 2
        if delta < mingain:
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)

def mdclassify(obs, tree):
    if tree.results is not None:
        return tree.results
    else:
        v=obs[tree.col]
        if v is None:
            tr, fr = mdclassify(obs, tree.tb), mdclassify(obs, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())

            tw = float(tcount)/(tcount+fcount)
            fw = float(fcount)/(tcount+fcount)
            result = {}
            for k,v in tr.items():
                result.setdefault(k, v*tw)
            for k,v in fr.items():
                result.setdefault(k, 0)
                result[k] += v*fw
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v>=tree.value: branch = tree.tb
                else: branch = tree.fb
            else:
                if v == tree.value: branch = tree.tb
                else: branch = tree.fb
            return mdclassify(obs, branch)

def variance(rows):
    if len(rows)==0:
        return 0
    data = [float(row[-1]) for row in rows]
    mean = sum(data)/len(data)
    variance = sum([(d-mean)**2 for d in data])/len(data)
    return variance