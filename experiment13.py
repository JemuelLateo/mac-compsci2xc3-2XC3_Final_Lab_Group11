import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from final_project_part1 import *

num_nodes = 50
max_weight = 100
k_values = [i for i in range(0, 20)]
trials = 50
source = 0

graphs = [create_random_complete_graph(num_nodes, max_weight) for _ in range(trials)]

approx_dijkstra_times = []
approx_bellman_times = []

exact_dijkstra_times = []
exact_bellman_times = []

# this was added to allocate memory and cache before hand 
# or else the first iteration of the loop will take significantly longer than normal
for G in graphs:
    dijkstra(G, source)
    bellman_ford(G, source)

for k in tqdm(k_values):
    avg_approx_d_time = 0
    avg_approx_b_time = 0

    avg_exact_d_time = 0
    avg_exact_b_time = 0
    for G in graphs:

        start = time.perf_counter()
        dijkstra_approx(G, source, k)
        avg_approx_d_time += time.perf_counter() - start

        start = time.perf_counter()
        bellman_ford_approx(G, source, k)
        avg_approx_b_time += time.perf_counter() - start

        start = time.perf_counter()
        dijkstra(G, source)
        avg_exact_d_time += time.perf_counter() - start

        start = time.perf_counter()
        bellman_ford(G, source)
        avg_exact_b_time += time.perf_counter() - start

    approx_dijkstra_times.append(avg_approx_d_time / trials)
    approx_bellman_times.append(avg_approx_b_time / trials)

    exact_dijkstra_times.append(avg_exact_d_time / trials)
    exact_bellman_times.append(avg_exact_b_time / trials)

print(f"approx_dikstra_time: {approx_dijkstra_times}")
print(f"approx_belman_time: {approx_bellman_times}")

print(f"exact_dikstra_time: {exact_dijkstra_times}")
print(f"exact_belman_time: {exact_bellman_times}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, exact_dijkstra_times, label="Dijkstra Exact", marker='o')
plt.plot(k_values, approx_dijkstra_times, label="Dijkstra Approx", marker='s')
plt.xlabel("k (max relaxations)")
plt.ylabel("Average Runtime (seconds)")
plt.title("Runtime vs k Relaxations")
plt.legend()
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("img/experiment13_d.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(k_values, exact_bellman_times, label="Bellman_Ford Exact", marker='o')
plt.plot(k_values, approx_bellman_times, label="Bellman-Ford Approx", marker='s')
plt.xlabel("k (max relaxations)")
plt.ylabel("Average Runtime (seconds)")
plt.title("Runtime vs k Relaxations")
plt.legend()
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("img/experiment13_b.png")
plt.show()
