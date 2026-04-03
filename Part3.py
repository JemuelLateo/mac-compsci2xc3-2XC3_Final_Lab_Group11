"""
Part 3: Comparing Dijkstra vs A* on the London Underground Network

"""

import csv
import math
import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from final_project_part1 import DirectedWeightedGraph, dijkstra
import min_heap



# A* Algorithm (from Part 2)


def reconstruct_path(pred, s, d):
    if s == d:
        return [s]
    if d not in pred:
        return []
    path = []
    current = d
    while current != s:
        path.append(current)
        current = pred[current]
    path.append(s)
    path.reverse()
    return path


def a_star(G, s, d, h):
    pred = {}
    dist = {}
    Q = min_heap.MinHeap([])
    closed = set()
    nodes = list(G.adj.keys())

    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")

    dist[s] = 0
    Q.decrease_key(s, dist[s] + h.get(s, 0))

    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value

        if current_node in closed:
            continue
        closed.add(current_node)

        if current_node == d:
            break

        for neighbour in G.adj[current_node]:
            if neighbour in closed:
                continue
            tentative_g = dist[current_node] + G.w(current_node, neighbour)
            if tentative_g < dist[neighbour]:
                dist[neighbour] = tentative_g
                pred[neighbour] = current_node
                f_value = tentative_g + h.get(neighbour, 0)
                Q.decrease_key(neighbour, f_value)

    path = reconstruct_path(pred, s, d)
    return pred, path



# Unified Plot Style (matches team style: blue + orange, circles/squares, dashed grid)
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.dpi': 150,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

COLOR_BLUE   = '#1f77b4'
COLOR_ORANGE = '#ff7f0e'
MARKER_BLUE  = 'o'
MARKER_ORANGE = 's'
GRID_STYLE = {'color': 'gray', 'linestyle': '--', 'alpha': 0.5}


# 1. Data Loading

def load_stations(filename):
    """Load station data from CSV. Returns dict: station_id -> {name, latitude, longitude, ...}"""
    stations = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['id'])
            stations[sid] = {
                'name': row['name'].strip('"'),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'zone': row['zone'],
                'total_lines': int(row['total_lines']),
                'rail': int(row['rail'])
            }
    return stations


def load_connections(filename):
    """Load connection data from CSV. Returns list of (station1, station2, line, time)."""
    connections = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            connections.append((
                int(row['station1']), int(row['station2']),
                int(row['line']), int(row['time'])
            ))
    return connections


# 2. Graph Construction


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance (km) between two lat/lon points using Haversine formula."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_london_graph(stations, connections):
    """Build a DirectedWeightedGraph from station/connection data.
    Edge weights = Haversine distance between connected stations.
    Edges are bidirectional (underground goes both ways)."""
    G = DirectedWeightedGraph()
    for sid in stations:
        G.add_node(sid)
    for s1, s2, line, t in connections:
        dist = haversine(stations[s1]['latitude'], stations[s1]['longitude'],
                         stations[s2]['latitude'], stations[s2]['longitude'])
        G.add_edge(s1, s2, dist)
        G.add_edge(s2, s1, dist)
    return G


def build_heuristic(stations, destination):
    """Build A* heuristic: straight-line distance from each station to destination."""
    h = {}
    dest_lat = stations[destination]['latitude']
    dest_lon = stations[destination]['longitude']
    for sid in stations:
        h[sid] = haversine(stations[sid]['latitude'], stations[sid]['longitude'],
                           dest_lat, dest_lon)
    return h



# 3. Helper Functions


def get_station_lines(connections):
    """Return dict: station_id -> set of line numbers the station is on."""
    station_lines = {}
    for s1, s2, line, t in connections:
        station_lines.setdefault(s1, set()).add(line)
        station_lines.setdefault(s2, set()).add(line)
    return station_lines


def build_edge_lines(connections):
    """Return dict: (station1, station2) -> set of line numbers for that edge."""
    edge_lines = {}
    for s1, s2, line, t in connections:
        edge_lines.setdefault((s1, s2), set()).add(line)
        edge_lines.setdefault((s2, s1), set()).add(line)
    return edge_lines


def count_transfers(path, edge_lines_cache):
    """Count minimum number of line transfers along a path using greedy approach."""
    if len(path) <= 1:
        return 0
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    current_lines = edge_lines_cache.get(edges[0], set())
    transfers = 0
    for i in range(1, len(edges)):
        available = edge_lines_cache.get(edges[i], set())
        common = current_lines & available
        if common:
            current_lines = common
        else:
            transfers += 1
            current_lines = available
    return transfers


def dijkstra_single_target(G, source, dest):
    """Point-to-point Dijkstra with early termination (for fair comparison with A*)."""
    pred = {}
    dist = {}
    Q = min_heap.MinHeap([])
    for node in G.adj:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        if current_node == dest:
            break
        for neighbour in G.adj[current_node]:
            new_dist = dist[current_node] + G.w(current_node, neighbour)
            if new_dist < dist[neighbour]:
                Q.decrease_key(neighbour, new_dist)
                dist[neighbour] = new_dist
                pred[neighbour] = current_node
    return pred, dist.get(dest, float("inf"))


# 4. Experiment


def run_experiment(G, stations, connections, sample_size=10000):
    """Run Dijkstra vs A* on sampled station pairs. Returns list of result dicts."""
    station_ids = sorted(stations.keys())
    station_lines = get_station_lines(connections)
    edge_lines_cache = build_edge_lines(connections)

    # Generate all pairs and sample
    all_pairs = [(s, d) for s in station_ids for d in station_ids if s != d]
    random.seed(42)
    if sample_size and sample_size < len(all_pairs):
        pairs = random.sample(all_pairs, sample_size)
    else:
        pairs = all_pairs

    print(f"Running experiment: {len(pairs)} pairs...")
    results = []

    for idx, (s, d) in enumerate(pairs):
        if (idx + 1) % 2000 == 0:
            print(f"  Progress: {idx + 1}/{len(pairs)} ({100*(idx+1)/len(pairs):.1f}%)")

        # Build heuristic for this destination
        h = build_heuristic(stations, d)

        # Time Dijkstra (point-to-point)
        repeat = 5
        dij_times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _, dij_dist = dijkstra_single_target(G, s, d)
            dij_times.append(time.perf_counter() - t0)

        # Time A*
        astar_times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _, astar_path = a_star(G, s, d, h)
            astar_times.append(time.perf_counter() - t0)
        dij_time = np.mean(dij_times)
        astar_time = np.mean(astar_times)

        # Calculate A* path distance
        astar_dist = 0
        if len(astar_path) > 1:
            for i in range(len(astar_path) - 1):
                astar_dist += G.w(astar_path[i], astar_path[i + 1])

        # Categorize pair
        lines_s = station_lines.get(s, set())
        lines_d = station_lines.get(d, set())
        category = 'same_line' if lines_s & lines_d else 'different_line'

        # Count transfers
        transfers = count_transfers(astar_path, edge_lines_cache)

        # Speedup ratio
        speedup = dij_time / astar_time if astar_time > 0 else float('inf')

        results.append({
            'source': s, 'dest': d,
            'source_name': stations[s]['name'],
            'dest_name': stations[d]['name'],
            'dijkstra_time': dij_time,
            'astar_time': astar_time,
            'dijkstra_dist': dij_dist,
            'astar_dist': astar_dist,
            'category': category,
            'transfers': transfers,
            'speedup': speedup
        })

    print(f"Experiment complete. {len(results)} pairs tested.")
    return results



# 5. Plotting (Unified Style)


def plot_figure1(results):
    """Figure 1: Average Runtime vs Number of Transfers"""
    tg_dij = {}
    tg_ast = {}
    for r in results:
        t = r['transfers']
        tg_dij.setdefault(t, []).append(r['dijkstra_time'] * 1000)
        tg_ast.setdefault(t, []).append(r['astar_time'] * 1000)

    sorted_t = sorted(tg_dij.keys())
    dij_means = [np.mean(tg_dij[t]) for t in sorted_t]
    ast_means = [np.mean(tg_ast[t]) for t in sorted_t]

    fig, ax = plt.subplots()
    ax.plot(sorted_t, dij_means, color=COLOR_BLUE, marker=MARKER_BLUE,
            linewidth=2, markersize=8, label='Dijkstra')
    ax.plot(sorted_t, ast_means, color=COLOR_ORANGE, marker=MARKER_ORANGE,
            linewidth=2, markersize=8, label='A*')
    ax.set_xlabel('Number of Transfers')
    ax.set_ylabel('Average Runtime (ms)')
    ax.set_title('Average Runtime vs Number of Transfers')
    ax.set_xticks(sorted_t)
    ax.legend()
    ax.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.savefig('figure1_runtime_vs_transfers.png', dpi=150)
    plt.close()
    print("Saved: figure1_runtime_vs_transfers.png")


def plot_figure2(results):
    """Figure 2: A* Win Rate vs Dijkstra Win Rate by Number of Transfers"""
    tg = {}
    for r in results:
        t = r['transfers']
        tg.setdefault(t, {'astar_wins': 0, 'dij_wins': 0, 'total': 0})
        tg[t]['total'] += 1
        if r['speedup'] > 1.0:
            tg[t]['astar_wins'] += 1
        else:
            tg[t]['dij_wins'] += 1

    sorted_t = sorted(tg.keys())
    astar_rates = [100 * tg[t]['astar_wins'] / tg[t]['total'] for t in sorted_t]
    dij_rates = [100 * tg[t]['dij_wins'] / tg[t]['total'] for t in sorted_t]

    fig, ax = plt.subplots()
    ax.plot(sorted_t, astar_rates, color=COLOR_BLUE, marker=MARKER_BLUE,
            linewidth=2, markersize=8, label='A* Win Rate')
    ax.plot(sorted_t, dij_rates, color=COLOR_ORANGE, marker=MARKER_ORANGE,
            linewidth=2, markersize=8, label='Dijkstra Win Rate')
    ax.set_xlabel('Number of Transfers')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('A* vs Dijkstra Win Rate by Number of Transfers')
    ax.set_xticks(sorted_t)
    ax.legend()
    ax.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.savefig('figure2_win_rate_vs_transfers.png', dpi=150)
    plt.close()
    print("Saved: figure2_win_rate_vs_transfers.png")


def plot_figure3(results):
    """Figure 3: A* Speedup - Same Line vs Different Line by Transfers"""
    same = {}
    diff = {}
    for r in results:
        t = r['transfers']
        if r['category'] == 'same_line':
            same.setdefault(t, []).append(r['speedup'])
        else:
            diff.setdefault(t, []).append(r['speedup'])

    # Only include transfer counts with enough samples in both categories
    all_t = sorted(set(same.keys()) & set(diff.keys()))
    all_t = [t for t in all_t if len(same.get(t, [])) >= 5 and len(diff.get(t, [])) >= 5]

    same_means = [np.mean(same[t]) for t in all_t]
    diff_means = [np.mean(diff[t]) for t in all_t]

    fig, ax = plt.subplots()
    ax.plot(all_t, same_means, color=COLOR_BLUE, marker=MARKER_BLUE,
            linewidth=2, markersize=8, label='Endpoint Shared Line')
    ax.plot(all_t, diff_means, color=COLOR_ORANGE, marker=MARKER_ORANGE,
            linewidth=2, markersize=8, label='Different Line')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal Performance')
    ax.set_xlabel('Number of Transfers')
    ax.set_ylabel('Mean Speedup Ratio (Dijkstra / A*)')
    ax.set_title('A* Speedup by Transfers: Endpoint Line Relationship')
    ax.set_xticks(all_t)
    ax.legend()
    ax.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.savefig('figure3_category_comparison.png', dpi=150)
    plt.close()
    print("Saved: figure3_category_comparison.png")


def plot_figure4(results):
    """Figure 4: Dijkstra vs A* Average Runtime by Shortest Path Distance"""
    valid = [(r['dijkstra_dist'], r['dijkstra_time'] * 1000, r['astar_time'] * 1000)
             for r in results if r['dijkstra_dist'] < float('inf')]

    max_dist = max(v[0] for v in valid)
    bin_size = 5
    bins = np.arange(0, max_dist + bin_size, bin_size)
    bin_centers = []
    dij_means = []
    ast_means = []

    for i in range(len(bins) - 1):
        in_bin = [(dt, at) for d, dt, at in valid if bins[i] <= d < bins[i + 1]]
        if len(in_bin) > 10:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            dij_means.append(np.mean([x[0] for x in in_bin]))
            ast_means.append(np.mean([x[1] for x in in_bin]))

    fig, ax = plt.subplots()
    ax.plot(bin_centers, dij_means, color=COLOR_BLUE, marker=MARKER_BLUE,
            linewidth=2, markersize=8, label='Dijkstra')
    ax.plot(bin_centers, ast_means, color=COLOR_ORANGE, marker=MARKER_ORANGE,
            linewidth=2, markersize=8, label='A*')
    ax.set_xlabel('Shortest Path Distance (km)')
    ax.set_ylabel('Average Runtime (ms)')
    ax.set_title('Average Runtime vs Shortest Path Distance')
    ax.legend()
    ax.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.savefig('figure4_runtime_vs_distance.png', dpi=150)
    plt.close()
    print("Saved: figure4_runtime_vs_distance.png")


def plot_figure5(results):
    """Figure 5: Dijkstra vs A* Average Runtime - Same Line vs Different Line"""
    same_dij = {}
    same_ast = {}
    diff_dij = {}
    diff_ast = {}

    for r in results:
        t = r['transfers']
        if r['category'] == 'same_line':
            same_dij.setdefault(t, []).append(r['dijkstra_time'] * 1000)
            same_ast.setdefault(t, []).append(r['astar_time'] * 1000)
        else:
            diff_dij.setdefault(t, []).append(r['dijkstra_time'] * 1000)
            diff_ast.setdefault(t, []).append(r['astar_time'] * 1000)

    # Use different_line data (larger sample)
    sorted_t = sorted(diff_dij.keys())
    sorted_t = [t for t in sorted_t if len(diff_dij.get(t, [])) >= 5]

    dij_means = [np.mean(diff_dij[t]) for t in sorted_t]
    ast_means = [np.mean(diff_ast[t]) for t in sorted_t]

    fig, ax = plt.subplots()
    ax.plot(sorted_t, dij_means, color=COLOR_BLUE, marker=MARKER_BLUE,
            linewidth=2, markersize=8, label='Dijkstra')
    ax.plot(sorted_t, ast_means, color=COLOR_ORANGE, marker=MARKER_ORANGE,
            linewidth=2, markersize=8, label='A*')
    ax.set_xlabel('Number of Transfers')
    ax.set_ylabel('Average Runtime (ms)')
    ax.set_title('Average Runtime vs Transfers (Different Line Pairs)')
    ax.set_xticks(sorted_t)
    ax.legend()
    ax.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.savefig('figure5_runtime_diff_line.png', dpi=150)
    plt.close()
    print("Saved: figure5_runtime_diff_line.png")



# 6. Statistics


def print_statistics(results):
    """Print summary statistics of the experiment."""
    speedups = [r['speedup'] for r in results]
    dij_times = [r['dijkstra_time'] * 1000 for r in results]
    astar_times = [r['astar_time'] * 1000 for r in results]

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE 2 - SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal pairs tested: {len(results)}")
    print(f"\nDijkstra average time: {np.mean(dij_times):.4f} ms")
    print(f"A* average time:      {np.mean(astar_times):.4f} ms")
    print(f"\nMean speedup (Dijkstra/A*):   {np.mean(speedups):.4f}")
    print(f"Median speedup:               {np.median(speedups):.4f}")

    astar_wins = sum(1 for s in speedups if s > 1.0)
    dij_wins = sum(1 for s in speedups if s < 1.0)
    print(f"\nA* faster: {astar_wins} pairs ({100*astar_wins/len(results):.1f}%)")
    print(f"Dijkstra faster: {dij_wins} pairs ({100*dij_wins/len(results):.1f}%)")

    # By category
    print("\n--- By Category ---")
    for cat in ['same_line', 'different_line']:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            cat_speedups = [r['speedup'] for r in cat_results]
            print(f"  {cat}: {len(cat_results)} pairs, mean speedup = {np.mean(cat_speedups):.4f}")

    # By transfers
    print("\n--- By Number of Transfers ---")
    transfer_groups = {}
    for r in results:
        transfer_groups.setdefault(r['transfers'], []).append(r['speedup'])
    for t in sorted(transfer_groups.keys()):
        sp = transfer_groups[t]
        print(f"  {t} transfers: {len(sp)} pairs, mean speedup = {np.mean(sp):.4f}")

    # Correctness check
    print("\n--- Correctness Check ---")
    mismatches = sum(1 for r in results if abs(r['dijkstra_dist'] - r['astar_dist']) > 0.001)
    print(f"Distance mismatches: {mismatches} / {len(results)}")
    if mismatches == 0:
        print("A* and Dijkstra produce identical shortest path distances.")

    # Top cases
    print("\n--- Top 10 Pairs Where A* is Fastest ---")
    for r in sorted(results, key=lambda x: x['speedup'], reverse=True)[:10]:
        print(f"  {r['source_name']} -> {r['dest_name']}: "
              f"speedup={r['speedup']:.2f}x, transfers={r['transfers']}")

    print("\n--- Top 10 Pairs Where Dijkstra is Fastest ---")
    for r in sorted(results, key=lambda x: x['speedup'])[:10]:
        print(f"  {r['source_name']} -> {r['dest_name']}: "
              f"speedup={r['speedup']:.2f}x, transfers={r['transfers']}")



# 7. Main


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    stations = load_stations("london_stations.csv")
    connections = load_connections("london_connections.csv")
    print(f"Loaded {len(stations)} stations and {len(connections)} connections.")

    # Build graph
    print("Building London Underground graph...")
    G = build_london_graph(stations, connections)
    print(f"Graph built with {G.number_of_nodes()} nodes.")

    # Run experiment (10000 sampled pairs for efficiency; set to None for all pairs)
    results = run_experiment(G, stations, connections, sample_size=10000)

    # Print statistics
    print_statistics(results)

    # Generate all 5 figures
    print("\nGenerating plots...")
    plot_figure1(results)
    plot_figure2(results)
    plot_figure3(results)
    plot_figure4(results)
    plot_figure5(results)

    print("\nAll experiments and figures complete.")
