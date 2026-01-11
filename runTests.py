import csv
import random
import time
from collections import deque

# ----------------------------
# Graph utils
# ----------------------------
def build_adj_list(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def gen_random_graph(n, p, rng):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return build_adj_list(n, edges), len(edges)

def is_connected(n, adj):
    # assume n >= 1
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


# ----------------------------
# 1) Backtracking puro (instrumentado)
# ----------------------------
def has_hamiltonian_path_bt(n, adj, time_limit_s=None):
    if n <= 1:
        return True, 1, 0, 0.0, "ok"

    visited = [False] * n
    iterations = 0
    expansions = 0
    t0 = time.perf_counter()

    def timed_out():
        return (time_limit_s is not None) and ((time.perf_counter() - t0) > time_limit_s)

    def dfs(v, count):
        nonlocal iterations, expansions
        iterations += 1
        if timed_out():
            return None
        if count == n:
            return True

        for u in adj[v]:
            expansions += 1
            if not visited[u]:
                visited[u] = True
                res = dfs(u, count + 1)
                if res is True:
                    return True
                if res is None:
                    return None
                visited[u] = False
        return False

    for start in range(n):
        visited[start] = True
        res = dfs(start, 1)
        if res is True:
            return True, iterations, expansions, time.perf_counter() - t0, "ok"
        if res is None:
            return None, iterations, expansions, time.perf_counter() - t0, "timeout"
        visited[start] = False

    return False, iterations, expansions, time.perf_counter() - t0, "ok"


# ----------------------------
# 2) Branch-and-Bound (instrumentado, otimizado)
#    - pré-podas: isolado, >2 grau1, desconexo
#    - heurística: menor grau residual primeiro
#    - poda local incremental: evita "dead end" sem varrer o grafo todo
# ----------------------------
def has_hamiltonian_path_bb(n, adj, time_limit_s=None):
    if n <= 1:
        return True, 1, 0, 0.0, "ok"

    deg = [len(adj[i]) for i in range(n)]

    if any(d == 0 for d in deg):
        return False, 0, 0, 0.0, "pruned_isolated"

    endpoints = [i for i in range(n) if deg[i] == 1]
    if len(endpoints) > 2:
        return False, 0, 0, 0.0, "pruned_deg1"

    if not is_connected(n, adj):
        return False, 0, 0, 0.0, "pruned_disconnected"

    visited = [False] * n
    residual = deg[:]  # grau residual (atualizado incrementalmente)
    iterations = 0
    expansions = 0
    t0 = time.perf_counter()

    def timed_out():
        return (time_limit_s is not None) and ((time.perf_counter() - t0) > time_limit_s)

    def dfs(v, count):
        nonlocal iterations, expansions
        iterations += 1
        if timed_out():
            return None
        if count == n:
            return True

        candidates = [u for u in adj[v] if not visited[u]]
        candidates.sort(key=lambda x: residual[x])  # menor grau residual primeiro

        for u in candidates:
            expansions += 1
            visited[u] = True

            # update incremental do residual dos vizinhos não visitados de u
            changed = []
            dead = False
            for w in adj[u]:
                if not visited[w]:
                    residual[w] -= 1
                    changed.append(w)
                    if residual[w] == 0 and (count + 1) < (n - 1):
                        dead = True

            if not dead:
                res = dfs(u, count + 1)
                if res is True:
                    return True
                if res is None:
                    return None

            # rollback
            for w in changed:
                residual[w] += 1
            visited[u] = False

        return False

    starts = endpoints if len(endpoints) == 2 else range(n)
    for s in starts:
        for i in range(n):
            visited[i] = False
            residual[i] = deg[i]
        visited[s] = True

        res = dfs(s, 1)
        if res is True:
            return True, iterations, expansions, time.perf_counter() - t0, "ok"
        if res is None:
            return None, iterations, expansions, time.perf_counter() - t0, "timeout"

    return False, iterations, expansions, time.perf_counter() - t0, "ok"


# ----------------------------
# Experimentos + CSV (simples)
# ----------------------------
def run_experiments(
    ns=(10, 20, 30, 40, 50),
    graphs_per_setting=4,
    p_sparse=0.15,
    p_mid=0.30,
    p_dense=0.60,
    seed=42,
    time_limit_s=2.0,
    out_csv="results_hamilton.csv",
):
    rng = random.Random(seed)

    fieldnames = [
        "n", "m", "density", "p", "graph_id",
        "bt_answer", "bt_status", "bt_iterations", "bt_expansions", "bt_time_s",
        "bb_answer", "bb_status", "bb_iterations", "bb_expansions", "bb_time_s",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n in ns:
            for density_name, p in (("sparse", p_sparse), ("mid", p_mid), ("dense", p_dense)):
                for graph_id in range(graphs_per_setting):
                    adj, m = gen_random_graph(n, p, rng)

                    # ---------- Backtracking ----------
                    bt_ans, bt_it, bt_ex, bt_t, bt_status = has_hamiltonian_path_bt(
                        n, adj, time_limit_s=time_limit_s
                    )
                    bt_answer = "TIMEOUT" if bt_ans is None else ("YES" if bt_ans else "NO")

                    # ---------- Branch-and-Bound ----------
                    bb_ans, bb_it, bb_ex, bb_t, bb_status = has_hamiltonian_path_bb(
                        n, adj, time_limit_s=time_limit_s
                    )
                    bb_answer = "TIMEOUT" if bb_ans is None else ("YES" if bb_ans else "NO")

                    writer.writerow({
                        "n": n,
                        "m": m,
                        "density": density_name,
                        "p": p,
                        "graph_id": graph_id,

                        "bt_answer": bt_answer,
                        "bt_status": bt_status,
                        "bt_iterations": bt_it,
                        "bt_expansions": bt_ex,
                        "bt_time_s": bt_t,

                        "bb_answer": bb_answer,
                        "bb_status": bb_status,
                        "bb_iterations": bb_it,
                        "bb_expansions": bb_ex,
                        "bb_time_s": bb_t,
                    })

                    print(f"[n={n} {density_name} graph={graph_id}] done")

    print(f"\nOK: CSV saved to {out_csv}")


if __name__ == "__main__":
    run_experiments(
        ns=(10, 20, 30, 40, 50),
        graphs_per_setting=10,
        p_sparse=0.12,
        p_mid=0.20,
        p_dense=0.50,
        seed=42,
        time_limit_s=2.0,
        out_csv="results_hamilton.csv",
    )
