from final_project_part1 import DirectedWeightedGraph, dijkstra, bellman_ford
from Part2 import a_star


class Graph(DirectedWeightedGraph):
 


    def get_adj_nodes(self, node):
        return self.adjacent_nodes(node)

    def get_num_of_nodes(self):
        return self.number_of_nodes()


class WeightedGraph(Graph):

    pass


class HeuristicGraph(WeightedGraph):


    def __init__(self):
        super().__init__()
        self.heuristic = {}

    def set_heuristic(self, heuristic):
        self.heuristic = heuristic

    def get_heuristic(self):
        return self.heuristic


class SPAlgorithm:


    def calc_sp(self, graph, source, dest):
        raise NotImplementedError()


class Dijkstra(SPAlgorithm):
    def calc_sp(self, graph, source, dest):
        dist = dijkstra(graph, source)
        return dist.get(dest, float("inf"))


class Bellman_Ford(SPAlgorithm):
    def calc_sp(self, graph, source, dest):
        dist = bellman_ford(graph, source)
        return dist.get(dest, float("inf"))


class A_Star(SPAlgorithm):


    def __init__(self):
        self.last_pred = {}
        self.last_path = []

    def calc_sp(self, graph, source, dest):
        # TODO:
  
        h = graph.get_heuristic()

        pred, path = a_star(graph, source, dest, h)

        self.last_pred = pred
        self.last_path = path

        if path == []:
            return float("inf")

        total = 0.0
        for i in range(len(path) - 1):
            total += graph.w(path[i], path[i + 1])

        return total

    def get_last_path(self):
        return self.last_path

    def get_last_pred(self):
        return self.last_pred


class ShortPathFinder:


    def __init__(self):
        self.graph = None
        self.algorithm = None

    def set_graph(self, graph):
        self.graph = graph

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def calc_short_path(self, source, dest):
        if self.graph is None:
            raise ValueError("Graph has not been set.")
        if self.algorithm is None:
            raise ValueError("Algorithm has not been set.")
        return self.algorithm.calc_sp(self.graph, source, dest)


