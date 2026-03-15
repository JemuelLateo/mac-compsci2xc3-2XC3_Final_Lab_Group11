import networkx as nx
import matplotlib.pyplot as plt

from final_project_part1 import *

def print_graph(G):
    for node in G.adj:
        for neighbour in G.adj[node]:
            print(f"{node} -> {neighbour}, weight: {G.weights[(node, neighbour)]}")