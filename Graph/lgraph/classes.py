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
