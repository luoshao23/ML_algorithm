import numpy as np
import networkx as nx


def evalTarget(i, x):
    return x * qty[i] * (x / disc[i]) ** gamma


def add1(G, u, nums):
    ui, uw = u
    if ui >= nums:
        return
    if uw >= 0.3:
        G.add_edge(u, (ui+1, uw), weight=evalTarget(ui, uw))
        add1(G, (ui+1, uw), nums)
    if uw >= 0.4:
        G.add_edge(u, (ui+1, uw-0.1), weight=evalTarget(ui, uw-0.1))
        add1(G, (ui+1, uw-0.1), nums)


def add2(G, nums):
    for ind in range(nums):
        for di in np.arange(0.3, 0.8, 0.1):
            for dj in available(di):
                G.add_edge((ind, di), (ind+1, dj),
                           weight=evalTarget(ind, dj))


def available(d):
    while d > 0.3:
        yield d
        d -= 0.1

if __name__ == "__main__":
    high = 0.7
    gamma = -0.5
    qty = np.asarray([502, 463, 421, 583, 653, 518, 221, 169, ])
    disc = np.asarray([0.567, 0.567, 0.567, 0.567, 0.567, 0.567, 0.567, 0.567, ])
    nums = len(qty)
    low = high - 0.1 * (nums - 1)

    date = list(range(nums + 1))

    G = nx.DiGraph()
    # add1(G, (0, high), nums)
    add2(G, nums)
    print(G.edges)
    print(nx.dag_longest_path_length(G))
    path = nx.dag_longest_path(G)
    print(path)
    print([p[1] for p in path[1:]])
