import time
import random
import math
import operator

people = [('Seymour','BOS'),
          ('Franny','DAL'),
          ('Zooey','CAK'),
          ('Walt','MIA'),
          ('Buddy','ORD'),
          ('Les','OMA')]

destination = 'LGA'

flights = {}

with open('schedule.txt') as f:
	for line in f:
		origin, dest, depart, arrive, price = line.strip().split(',')
		flights.setdefault((origin, dest), [])

		flights[(origin, dest)].append((depart, arrive, int(price)))

def getminutes(t):
	x = time.strptime(t, '%H:%M')
	return x.tm_hour * 60 + x.tm_min

def printschedule(r):
	for x in xrange(len(r)/2):
		name = people[x][0]
		origin = people[x][1]
		out = flights[(origin, destination)][r[2*x]]
		ret = flights[(destination, origin)][r[2*x+1]]
		print '%10s%10s %5s-%5s $%3s  %5s-%5s $%3s' %(name, origin,
														out[0], out[1], out[2],
														ret[0], ret[1], ret[2])

def schedulecost(sol):
	totalprice = 0
	latestarrival = 0
	earliestdep = 24*60

	for d in xrange(len(sol)/2):
		origin = people[d][1]
		out = flights[(origin, destination)][sol[2*d]]
		ret = flights[(destination, origin)][sol[2*d+1]]

		totalprice += out[2]+ret[2]

		if latestarrival < getminutes(out[1]): latestarrival = getminutes(out[1])
		if earliestdep > getminutes(ret[0]): earliestdep = getminutes(ret[0])

	totalwait = 0
	for d in xrange(len(sol)/2):
		origin = people[d][1]
		out = flights[(origin, destination)][sol[2*d]]
		ret = flights[(destination, origin)][sol[2*d+1]]

		totalwait += latestarrival-getminutes(out[1])
		totalwait += getminutes(ret[0]) - earliestdep

	if latestarrival<earliestdep: totalprice += 50

	return totalprice+totalwait

def printsol(sol, cost):
	print "best sol is %r and the cost is %f" %(sol, float(cost))

def randomopt(domain, costf=schedulecost):
	best = 999999999
	bestr = None
	for i in xrange(10000):
		r = [random.randint(domain[i][0], domain[i][1]) for i in xrange(len(domain))]

		cost = costf(r)

		if cost<best:
			best=cost
			bestr=r

	printsol(bestr, best)
	return bestr

def hillclimb(domain, costf=schedulecost):
	sol = [random.randint(domain[i][0], domain[i][1]) for i in xrange(len(domain))]

	while 1:
		neighbors = []
		for j in xrange(len(domain)):
			if sol[j]>domain[j][0]:
				neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])
			if sol[j]<domain[j][1]:
				neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])

		current = costf(sol)
		best = current
		for j in xrange(len(neighbors)):
			cost = costf(neighbors[j])
			if cost < best:
				best = cost
				sol = neighbors[j]
		if best == current: break

	printsol(sol, best)
	return sol

def annealingopt(domain, costf=schedulecost, T=10000.0, cool=0.95, step=1):
	tmpcost = None
	sol = []
	for k in xrange(100):
		tmpsol = [random.randint(domain[i][0], domain[i][1]) for i in xrange(len(domain))]
		if tmpcost is None or costf(tmpsol) < tmpcost:
			sol = tmpsol

	while T>0.1:
		i = random.randint(0,len(domain)-1)

		dir = random.randint(-step, step)

		solb= sol[:]  ## shallow copy
		solb[i] += dir

		if solb[i] > domain[i][1]: solb[i] = domain[i][1]
		elif solb[i] < domain[i][0]: solb[i] = domain[i][0]

		ea=costf(sol)
		eb=costf(solb)

		if (eb<ea or random.random()<math.exp(-(eb-ea)/T)):
			sol = solb
		T = T*cool

	printsol(sol, costf(sol))
	return sol

def genopt(domain, costf=schedulecost, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100, threshold=0.0):

	def mutate(vec):
		i = random.randint(0,len(domain)-1)
		if random.random()<0.5:
			new = vec[0:i]+[vec[i]-step]+vec[i+1:]
		else:
			new = vec[0:i]+[vec[i]+step]+vec[i+1:]

		if new[i] > domain[i][1]: new[i] = domain[i][1]
		elif new[i] < domain[i][0]: new[i] = domain[i][0]

		return new

	def crossover(r1, r2):
		i = random.randint(0,len(domain)-1)
		return r1[0:i]+r2[i:]

	pop = []
	for i in xrange(popsize):
		vec = [random.randint(domain[i][0], domain[i][1]) for i in xrange(len(domain))]
		pop.append(vec)

	topelite = int(elite*popsize)
	tmptopcost = None

	for i in xrange(maxiter):
		scores = [(costf(v), v) for v in pop]
		scores.sort(key=operator.itemgetter(0))

		if maxiter>10 and tmptopcost is not None and abs(scores[0][0]-tmptopcost)/float(tmptopcost)<threshold:
			break
		ranked = [v for (s,v) in scores]

		pop = ranked[0:topelite]

		while len(pop)<popsize:
			if random.random()<mutprob:
				c = random.randint(0, topelite)
				pop.append(mutate(ranked[c]))
			else:
				c1 = random.randint(0, topelite)
				c2 = random.randint(0, topelite)
				pop.append(crossover(ranked[c1], ranked[c2]))
		tmptopcost = scores[0][0]

	printsol(scores[0][1], scores[0][0])

	return scores[0][1]