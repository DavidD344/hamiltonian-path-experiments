def build_adj_list(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def has_hamiltonian_path_bt(n, adj):
    # caso trivial
    if n <= 1:
        return True

    visited = [False] * n

    def dfs(v, count):
        if count == n:
            return True
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                if dfs(u, count + 1):
                    return True
                visited[u] = False
        return False

    for start in range(n):
        visited[start] = True
        if dfs(start, 1):
            return True
        visited[start] = False

    return False

n = 10
edges = [
    (8, 9),
    (0, 1),
    (0, 4),
    (1, 2),
    (1, 3),
    (2, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (6, 7)
]

adj = build_adj_list(n, edges)

print("SIM" if has_hamiltonian_path_bt(n, adj) else "NAO")
