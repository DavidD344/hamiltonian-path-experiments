from collections import deque

def build_adj_list(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj
def is_connected(n, adj):
    vis = [False] * n
    q = deque([0])
    vis[0] = True
    while q:
        v = q.popleft()
        for u in adj[v]:
            if not vis[u]:
                vis[u] = True
                q.append(u)
    return all(vis)


def has_hamiltonian_path_bb(n, adj):
    # -------- casos triviais --------
    if n <= 1:
        return True

    deg = [len(adj[i]) for i in range(n)]

    # -------- pré-podas --------
    # vértice isolado
    if any(d == 0 for d in deg):
        return False

    # mais de dois vértices grau 1
    endpoints = [i for i in range(n) if deg[i] == 1]
    if len(endpoints) > 2:
        return False

    # desconexo
    if not is_connected(n, adj):
        return False

    visited = [False] * n
    residual_deg = deg[:]  # grau residual (incremental)

    def dfs(v, count):
        if count == n:
            return True

        # candidatos não visitados
        candidates = [u for u in adj[v] if not visited[u]]

        # heurística: menor grau residual primeiro
        candidates.sort(key=lambda x: residual_deg[x])

        for u in candidates:
            visited[u] = True

            # atualiza graus residuais
            changed = []
            for w in adj[u]:
                if not visited[w]:
                    residual_deg[w] -= 1
                    changed.append(w)

            # poda local barata:
            # se algum vértice não visitado ficou com grau residual 0,
            # não dá mais para encaixar depois
            dead = False
            for w in changed:
                if residual_deg[w] == 0:
                    dead = True
                    break

            if not dead and dfs(u, count + 1):
                return True

            # rollback
            for w in changed:
                residual_deg[w] += 1
            visited[u] = False

        return False

    # se houver exatamente dois vértices grau 1, eles são extremidades
    starts = endpoints if len(endpoints) == 2 else range(n)

    for s in starts:
        for i in range(n):
            visited[i] = False
        visited[s] = True
        if dfs(s, 1):
            return True

    return False
# ===== TESTE COM O EXEMPLO DO ENUNCIADO =====
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
print("SIM" if has_hamiltonian_path_bb(n, adj) else "NAO")
