# coding: utf-8


def connected_components(G):

    seen = set()
    for v in G:
        if v not in seen:
            c = set(_plain_bfs(G, v))
            yield c
            seen.update(c)


def _plain_bfs(G, source):

    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(G[v])


def topsort(G):

    indegree_map = {v: d for v, d in G.in_degree.items() if d > 0}
    nocoming = [v for v, d in G.in_degree.items() if d == 0]
    # S = []
    while nocoming:
        node = nocoming.pop()
        for adj in G[node]:
            indegree_map[adj] -= 1
            if indegree_map[adj] == 0:
                nocoming.append(adj)
                del indegree_map[adj]
        yield node

    if indegree_map:
        raise RuntimeError(
            "Graph contains a cycle or graph changed during iteration")


def dag_longest_path(G, weight='weight', default_weight=1):
    _MIN = -2**32
    dist = {v: (_MIN, v) for v in topsort(G)}
    for v in topsort(G):
        for u, data in G[v].items():
            if dist[u][0] < dist[v][0] + data.get(weight, default_weight):
                dist[u] = (dist[v][0] + data.get(weight, default_weight), v)
    u = None
    v = max(dist, key=lambda x: dist[x][0])

    path = []
    while u != v:
        path.append(v)
        u = v
        v = dist[v][1]
    path.reverse()
    return path
