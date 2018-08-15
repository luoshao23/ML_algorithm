import numpy as np

states = ['rainny', 'sunny', 'cloudy', 'snowy']
actions = ['up', 'down', 'none']
tol = 1e-3
gamma = 0.5

X_num = len(states)
A_num = len(actions)

X = range(X_num)
A = range(A_num)

pi = np.random.rand(X_num, A_num)
pi /= pi.sum(1, keepdims=1)

P = np.random.rand(A_num, X_num, X_num)
P /= P.sum(-1, keepdims=1)
R = np.random.rand(A_num, X_num, X_num)

v = {s: 0 for s in states}
print(v)

while True:
    vtmp = {}
    for ix, x in zip(X, states):
        vtmp[x] = sum([pi[ix, a] * sum([P[a,ix,xn] * (R[a,ix,xn] + gamma * v[states[xn]]) for xn in X]) for a in A])

    if max([vtmp[s] - v[s] for s in states]) < tol:
        break
    else:
        v = vtmp

print(v)