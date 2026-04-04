from final_project_part1 import *

# Manual graph with 4 nodes and negative edge weights
# 0 --(1)--> 1
# 1 --(-3)--> 2
# 2 --(2)--> 3
# 0 --(10)--> 3

G = create_random_complete_graph(4, 10)  # or however you create a graph

# Override edges manually
G.weights[(0,1)] = 1
G.weights[(1,2)] = -3
G.weights[(2,3)] = 2
G.weights[(0,3)] = 10

result = mystery(G)

# Expected shortest paths from node 0:
# 0 -> 0 = 0
# 0 -> 1 = 1
# 0 -> 2 = 1 + (-3) = -2
# 0 -> 3 = 1 + (-3) + 2 = 0  (better than direct 0->3 = 10)

print("Shortest paths from node 0:")
for j in range(4):
    print(f"0 -> {j}: {result[0][j]}")
