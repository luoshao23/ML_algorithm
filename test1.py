def test(iter, r):
	pool = tuple(iter)
	n = len(pool)

	if not n and r:
		return
	indices = [0] * r
	yield tuple(pool[i] for i in indices)

	while True:
		for i in reversed(xrange(r)):
			if indices[i] != n-1:
				break
		else:
			return
		indices[i:] = [indices[i] + 1]*(r-i)

		yield tuple(pool[i] for i in indices)
