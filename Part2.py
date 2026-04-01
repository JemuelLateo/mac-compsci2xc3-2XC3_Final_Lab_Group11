import min_heap

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