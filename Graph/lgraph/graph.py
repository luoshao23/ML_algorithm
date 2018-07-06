class Graph(object):

    """docstring for Graph"""

    __slots__ = ("nodes", "edges", "adj")

    def __init__(self, edges=[]):
        from collections import defaultdict
        self.edges = {}
        self.adj = defaultdict(dict)
        self.nodes = {}
        if edges:
            self.add_edges_from(edges)
            # for edge in edges:
            #     edge = tuple(set(edge))
            #     self.edges[edge] = {}
            #     p, q = edge
            #     self.nodes[p] = {}
            #     self.nodes[q] = {}
            #     self.adj[p][q] = {}
            #     self.adj[q][p] = {}

    def __getitem__(self, key):
        return self.adj[key]

    def __iter__(self):
        return iter(self.nodes)

    def add_node(self, node):
        self.nodes[node] = {}
        if node not in self.adj:
            self.adj[node] = {}

    def add_nodes_from(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_edge(self, *edge):
        if len(edge) <= 1:
            raise TypeError('at least 2 arguments!')
        edge = tuple(set(edge))
        self.add_nodes_from(edge)
        if len(edge) == 2:

            self.edges[edge] = {}

            p, q = edge
            self.adj[p][q] = {}
            self.adj[q][p] = {}

    def add_edges_from(self, edges):

        for edge in edges:
            self.add_edge(*edge)


class DiGraph(Graph):

    """docstring for DiGraph"""

    __slots__ = ("nodes", "edges", "adj")

    def __init__(self, edges=[]):
        from collections import defaultdict
        self.edges = {}
        self.adj = defaultdict(dict)
        self.nodes = {}
        if edges:
            self.add_edges_from(edges)

    def add_node(self, node):
        self.nodes[node] = {}
        if node not in self.adj:
            self.adj[node] = {}

    def add_nodes_from(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_edge(self, *edge, **attr):
        """
        edge(u,v) is u->v, which stores v as u's adjacency.
        """
        if len(edge) <= 1:
            raise TypeError('at least 2 arguments!')

        self.add_nodes_from(edge)
        if len(edge) == 2:
            self.edges[edge] = {}
            p, q = edge
            self.adj[p][q] = {}

    def add_edges_from(self, edges):
        for edge in edges:
            self.add_edge(*edge)

    @property
    def in_degree(self):
        in_degree = {u: 0 for u in self.nodes}

        for u in self.nodes:
            for v in self.adj[u]:
                in_degree[v] += 1
        return in_degree


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
        raise RuntimeError("Graph contains a cycle or graph changed during iteration")
    # return S


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


# def findLink(input, output, args):
#     # print 'Start'
#     GWhole = Graph()
#     # print 'add_edge start'

#     for line in input:
#         line = line.toDict()
#         xtra_card_nbr = int(line['xtra_card_nbr'])
#         xtra_card_nbr_2 = int(line['xtra_card_nbr_2'])
#         GWhole.add_edge(xtra_card_nbr, xtra_card_nbr_2)
#     # print 'add_edge finished'
#     # empty=set([])
#     connectedComponents = connected_components(GWhole)
#     del GWhole  # delete graph

#     value_list = {}
#     # print 'turn connectedComponents'

#     for nodeset in connectedComponents:
#         nodeset = list(nodeset)

#         for i in xrange(len(nodeset)):

#             value_list['XTRA_CARD_NBR'] = nodeset[0]
#             value_list['EC_CARD_LINKED'] = nodeset[i]
#             output.push(value_list)
