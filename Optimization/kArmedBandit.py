import numpy as np


def kArmedBandit(func, k=5, max_iter=100, epsilon=0.5, temp=10, method='kGreedy'):
    r = 0
    avg_reward = [0] * k
    counts = [0] * k
    actions = []
    if method == 'kGreedy':
        def fetch_one():
            if np.random.random() < epsilon:
                i = np.random.choice(k)
            else:
                i = np.argmax(avg_reward)
            return i
    elif method == 'softmax':
        def fetch_one():
            p = np.exp(np.array(avg_reward) / temp)
            p /= np.sum(p)
            p = np.cumsum(p)
            i = np.searchsorted(p, np.random.random())
            return i

    for _ in range(max_iter):
        i = fetch_one()
        actions.append(i)
        v = func(i)
        r += v
        counts[i] += 1
        avg_reward[i] += 1. / counts[i] * (v - avg_reward[i])
    e, n = np.unique(actions, return_counts=True)
    print(actions)

    return r, e[np.argmax(n)]


if __name__ == "__main__":
    def r(i):
        func = [10, 1, 2, 3]
        return func[i]
    r, strategy = kArmedBandit(r, 4, method='softmax')
    print(r, strategy)
