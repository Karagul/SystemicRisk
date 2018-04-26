import networkx as nx
import numpy as np


class GraphInit:

    def __init__(self, graph):
        self.graph = graph

    def get_nedges(self):
        n_edges = len(self.graph.edges())
        return n_edges

    def set_loans(self, di_strength):
        n_edges = self.get_nedges()
        G = nx.Graph()
        if di_strength.shape[0] != n_edges:
            print("shape incoherence")
            return 1
        else:
            count = 0
            for edge in self.graph.edges():
                G.add_edge(edge[0], edge[1], weight=di_strength[count])
                count += 1
        self.graph = G

    def get_weights_matrix(self):
        return np.asarray(nx.to_numpy_matrix(self.graph))

    def get_loans_matrix(self):
        return np.maximum(0, self.get_weights_matrix())

    def generate_dibernouilli(self, p):
        bers = 2 * (np.random.binomial(1, p, self.get_nedges()) - p)
        return bers

    def generate_loans_values(self, vals, distrib):
        values = np.random.choice(a=vals, size=self.get_nedges(), p=distrib)
        return values