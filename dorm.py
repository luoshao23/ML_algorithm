import random
import math

dorms=['Zeus','Athena','Hercules','Bacchus','Pluto']

# People, along with their first and second choices
prefs=[('Toby', ('Bacchus', 'Hercules')),
       ('Steve', ('Zeus', 'Pluto')),
       ('Karen', ('Athena', 'Zeus')),
       ('Sarah', ('Zeus', 'Pluto')),
       ('Dave', ('Athena', 'Bacchus')),
       ('Jeff', ('Hercules', 'Pluto')),
       ('Fred', ('Pluto', 'Athena')),
       ('Suzie', ('Bacchus', 'Hercules')),
       ('Laura', ('Bacchus', 'Hercules')),
       ('James', ('Hercules', 'Athena'))]

domain = [(0, len(dorms)*2-i-1) for i in xrange(len(dorms)*2)]

def printsol(vec):
	slots = []

	for i in xrange(len(dorms)):
		slots += [i,i]
	for i in xrange(len(vec)):
		x = int(vec[i])

		dorm = dorms[slots[x]]
		print prefs[i][0], dorm

		del slots[x]

def dormcost(vec):
	cost = 0
	slots = []

	for i in xrange(len(dorms)):
		slots += [i,i]

	for i in xrange(len(vec)):
		x = int(vec[i])
		dorm = dorms[slots[x]]
		pref = prefs[i][1]

		if pref[0] == dorm: cost += 0
		elif pref[1] == dorm: cost += 1
		else: cost += 3

		del slots[x]
	return cost
