import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import random 

from tqdm import tqdm
from final_project_part1 import *

def print_graph(G):
    for node in G.adj:
        for neighbour in G.adj[node]:
            print(f"{node} -> {neighbour}, weight: {G.weights[(node, neighbour)]}")

def accuracy(approx_dist, exact_dist):
    correct = 0
    for node in exact_dist:
        if approx_dist[node] == exact_dist[node]:
            correct += 1
    return correct / len(exact_dist)


num_nodes = 50
num_edges = 150
max_weight = 100
k_values = [i for i in range(0,20)]
trials = 50
source = 0

graphs = [create_random_graph(num_nodes, num_edges, max_weight) for _ in range(trials)]

dijkstra_accuracy = []
bellman_accuracy = []

for k in tqdm(k_values):
    avg_b = 0
    avg_d = 0

    for G in graphs:

        approx_d = dijkstra_approx(G, source, k)
        approx_b = bellman_ford_approx(G, source, k)

        exact_d = dijkstra(G, source)
        exact_b = bellman_ford(G, source)

        avg_d += accuracy(approx_d, exact_d)
        avg_b += accuracy(approx_b, exact_b)

    dijkstra_accuracy.append(avg_d / trials)
    bellman_accuracy.append(avg_b / trials)

print(f"dikstra_accuracy: {dijkstra_accuracy}")
print(f"belman_accuracyu: {bellman_accuracy}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, dijkstra_accuracy, label="Dijkstra Approx", marker='o')
plt.plot(k_values, bellman_accuracy, label="Bellman-Ford Approx", marker='s')
plt.xlabel("k (max relaxations)")
plt.ylabel("Average Accuracy (%)")
plt.title("Accuracy vs k relaxation with sparse graph")
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend()
plt.xticks(k_values)
plt.ylim(0, 1.1) 
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("img/experiment11.png")
plt.show()