from random import random, randint
import math
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

weightdomain = [(0, 20)] * 4


def wineprice(rating, age):
    peak_age = rating - 50

    price = float(rating) / 2
    if age > peak_age:
        price = price * (5 - (age - peak_age))
    else:
        price = price * (5 * float(age + 1) / peak_age)
    if price < 0:
        price = 0.0
    return price


def wineset1():
    rows = []
    for i in xrange(300):
        rating = random() * 50 + 50
        age = random() * 50

        price = wineprice(rating, age)
        price *= (random() * 0.2 + 0.9)

        rows.append((rating, age, price))
    rows = np.array(rows)
    return rows


def wineset2():
    rows = []
    for i in xrange(300):
        rating = random() * 50 + 50
        age = random() * 50
        aisle = float(randint(1, 20))
        bottlesize = [375.0, 750.0, 1500.0, 3000.0][randint(0, 3)]
        price = wineprice(rating, age)
        price *= (bottlesize / 750)
        price *= (random() * 0.2 + 0.9)
        rows.append((rating, age, aisle, bottlesize, price))
    rows = np.array(rows)
    return rows


def wineset3():
    rows = wineset1()
    for row in rows:
        if random() < 0.5:
            row[-1] *= 0.5
    return rows


def euclidean(v1, v2):
    d = 0.0
    for i in xrange(len(v1)):
        d += (v1[i] - v2[i])**2

    return math.sqrt(d)


def getdistances(data, vec1):
    distancelist = []
    for i in xrange(len(data)):
        vec2 = data[i][:-1]
        distancelist.append((euclidean(vec1, vec2), i))
    distancelist.sort()
    return distancelist


def knnestimate(data, vec1, k=5):
    dlist = getdistances(data, vec1)
    avg = 0.0

    for i in xrange(k):
        idx = dlist[i][1]
        avg += data[idx][-1]
    avg = avg / k
    return avg


def inverseweight(dist, num=1.0, const=0.1):
    return num / (dist + const)


def subtractweight(dist, const=1.0):
    if dist > const:
        return 0
    else:
        return const - dist


def gaussian(dist, sigma=5.0):
    return math.exp(-dist**2 / (2 * sigma**2))


def weightedknn(data, vec1, k=5, weightf=gaussian):
    dlist = getdistances(data, vec1)
    avg = 0.0
    totalweight = 0.0

    for i in xrange(k):
        dist = dlist[i][0]
        idx = dlist[i][1]
        weight = weightf(dist)
        avg += weight * data[idx][-1]
        totalweight += weight
    if totalweight == 0:
        return 0
    avg = avg / totalweight
    return avg


def dividedata(data, test=0.05):
    trainset = []
    testset = []
    for row in data:
        if random() < test:
            testset.append(row)
        else:
            trainset.append(row)
    return trainset, testset


def testalgorithm(algf, trainset, testset):
    error = 0.0
    for row in testset:
        guess = algf(trainset, row[:-1])
        error += (row[-1] - guess)**2
    return error / len(testset)


def crossvalidate(algf, data, trials=100, test=0.05):
    error = 0.0
    for i in xrange(trials):
        trainset, testset = dividedata(data, test)
        error += testalgorithm(algf, trainset, testset)
    return error / trials


def rescale(data, scale=None):
    if scale is not None and len(scale) == data.shape[1] - 1:
        scaleddata = data * (scale + [1])
    else:
        scaleddata = data / (np.mean(data, 0) + 0.0001)
        scaleddata[:, -1] = data[:, -1]

    return scaleddata


def createcostfunction(algf, data):
    def costf(scale):
        sdata = rescale(data, scale)
        return crossvalidate(algf, sdata, trials=20)
    return costf


def probguess(data, vec1, low, high, k=5, weightf=gaussian):
    dlist = getdistances(data, vec1)
    nweight = 0.0
    tweight = 0.0

    for i in xrange(k):
        dist = dlist[i][0]
        idx = dlist[i][1]
        weight = weightf(dist)
        v = data[idx][-1]

        if v>=low and v<=high:
            nweight += weight
        tweight += weight
    if tweight == 0:
        return 0
    return nweight/tweight

def cumulativegraph(data,vec1,high,k=5,weightf=gaussian):
    t1 = np.arange(0.0, high, 0.1)
    cprob = np.array([probguess(data, vec1, 0, v, k, weightf) for v in t1])
    data = go.Scatter(x=t1, y=cprob)
    fig = go.Figure(data=[data])
    py.plot(fig, filename='wineguess')

def probabilitygraph(data, vec1, high, k=5, weightf=gaussian, ss=5.0):
    t1 = np.arange(0.0, high, 0.1)
    probs = np.array([probguess(data, vec1, v, v+0.1, k, weightf) for v in t1])

    smoothed = []
    for i in xrange(len(probs)):
        sv = 0.0
        for j in xrange(len(probs)):
            dist = abs(i-j)*0.1
            weight = gaussian(dist, sigma=ss)
            sv += weight*probs[j]
        smoothed.append(sv)
    smoothed = np.array(smoothed)

    data = go.Scatter(x=t1, y=smoothed)
    fig = go.Figure(data=[data])
    py.plot(fig, filename='wineguess_smoothed')

data = wineset1()
