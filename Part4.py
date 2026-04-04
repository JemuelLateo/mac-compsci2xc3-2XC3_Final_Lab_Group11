from final_project_part1 import DirectedWeightedGraph, dijkstra, bellman_ford
from Part2 import a_star
from Part3 import (
    load_stations,
    load_connections,
    build_london_graph,
    build_heuristic,
)


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

    @classmethod
    def from_london_data(cls, stations_file, connections_file, destination):
        stations = load_stations(stations_file)
        connections = load_connections(connections_file)

        base_graph = build_london_graph(stations, connections)
        heuristic = build_heuristic(stations, destination)

        graph = cls()
        graph.adj = base_graph.adj
        graph.weights = base_graph.weights
        graph.set_heuristic(heuristic)


        graph.stations = stations
        graph.connections = connections
        graph.destination = destination

        return graph


class SPAlgorithm:

    def calc_sp(self, graph, source, dest):
        raise NotImplementedError()


class Dijkstra(SPAlgorithm):


    def __init__(self):
        self.last_distance = None

    def calc_sp(self, graph, source, dest):
        dist = dijkstra(graph, source)
        self.last_distance = dist.get(dest, float("inf"))
        return self.last_distance


class Bellman_Ford(SPAlgorithm):

    def __init__(self):
        self.last_distance = None

    def calc_sp(self, graph, source, dest):
        dist = bellman_ford(graph, source)
        self.last_distance = dist.get(dest, float("inf"))
        return self.last_distance


class A_Star(SPAlgorithm):

    def __init__(self):
        self.last_pred = {}
        self.last_path = []
        self.last_distance = None

    def calc_sp(self, graph, source, dest):
        h = graph.get_heuristic()
        pred, path = a_star(graph, source, dest, h)

        self.last_pred = pred
        self.last_path = path

        if not path:
            self.last_distance = float("inf")
            return self.last_distance

        total = 0.0
        for i in range(len(path) - 1):
            total += graph.w(path[i], path[i + 1])

        self.last_distance = total
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


if __name__ == "__main__":
    
    # London data demo
    STATIONS_FILE = "london_stations.csv"
    CONNECTIONS_FILE = "london_connections.csv"

    source = 1
    destination = 50

    # heuristic
    G = HeuristicGraph.from_london_data(
        STATIONS_FILE,
        CONNECTIONS_FILE,
        destination
    )

    finder = ShortPathFinder()
    finder.set_graph(G)

    # Dijkstra
    dij = Dijkstra()
    finder.set_algorithm(dij)
    dij_dist = finder.calc_short_path(source, destination)
    print("Dijkstra distance:", dij_dist)

    # Bellman-Ford
    bf = Bellman_Ford()
    finder.set_algorithm(bf)
    bf_dist = finder.calc_short_path(source, destination)
    print("Bellman-Ford distance:", bf_dist)

    # A*
    ast = A_Star()
    finder.set_algorithm(ast)
    ast_dist = finder.calc_short_path(source, destination)
    print("A* distance:", ast_dist)
    print("A* path:", ast.get_last_path())