def test(iter, r):
	pool = tuple(iter)
	n = len(pool)
	if not n and r:
		return
	indices = [0] * r
	yield tuple(pool[i] for i in indices)
	yield tuple(pool[-1])