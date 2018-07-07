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

    def add_edge(self, u, v, **attr):
        """
        edge(u,v) is u->v, which stores v as u's adjacency.
        """
        self.add_nodes_from((u, v))
        self.edges[(u, v)] = {}
        dictdata = self.adj[u].get(v, {})
        dictdata.update(attr)
        self.adj[u][v] = dictdata
        self.adj[v][u] = dictdata

    def add_edges_from(self, edges, **attr):
        """
        the attr in edges is private for one edge, while `attr` is public for all edges added in this time.
        """
        for edge in edges:
            ne = len(edge)
            if ne == 3:
                u, v, dd = edge
            elif ne == 2:
                u, v = edge
                dd = {}
            else:
                raise ValueError(
                    "Edge tuple %s must be a 2-tuple or 3-tuple." % (edge,))
            dd.update(attr)
            self.add_edge(u, v, **dd)


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

    def add_edge(self, u, v, **attr):
        """
        edge(u,v) is u->v, which stores v as u's adjacency.
        """
        self.add_nodes_from((u, v))
        self.edges[(u, v)] = {}
        dictdata = self.adj[u].get(v, {})
        dictdata.update(attr)
        self.adj[u][v] = dictdata

    def add_edges_from(self, edges, **attr):
        """
        the attr in edges is private for one edge, while `attr` is public for all edges added in this time.
        """
        for edge in edges:
            ne = len(edge)
            if ne == 3:
                u, v, dd = edge
            elif ne == 2:
                u, v = edge
                dd = {}
            else:
                raise ValueError(
                    "Edge tuple %s must be a 2-tuple or 3-tuple." % (edge,))
            dd.update(attr)
            self.add_edge(u, v, **dd)

    def add_weighted_edges_from(self, edges, weight='weight', **attr):
        self.add_edges_from(((u, v, {weight: d}) for u, v, d in edges), **attr)

    @property
    def in_degree(self):
        in_degree = {u: 0 for u in self.nodes}

        for u in self.nodes:
            for v in self.adj[u]:
                in_degree[v] += 1
        return in_degree
