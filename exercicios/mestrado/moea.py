# nsga_parallel.py
import numpy as np
import matplotlib.pyplot as plt
import os
from math import inf
import concurrent.futures
import time
import multiprocessing
from functools import partial

# ======== Parâmetros (edite se quiser) ========
N_VALUES = [20, 50, 100, 200]
IND_SIZE = 30
GENERATIONS = 100
RUNS = 10
OUTDIR = "nsga_results"
BASE_SEED = 42  # muda isso para outra reprodução diferente
MAX_WORKERS = None  # None -> usa cpu_count()

os.makedirs(OUTDIR, exist_ok=True)

# ======== Funções Objetivo (ZDT1) ========
def f1_ind(ind): return ind[0]
def g_ind(ind): return 1 + 9 * np.sum(ind[1:]) / (len(ind) - 1)
def f2_ind(ind): return g_ind(ind) * (1 - np.sqrt(ind[0] / g_ind(ind)))

# ======== Funções Auxiliares ========
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def pareto_fronts(objs):
    n = len(objs)
    dominated_count = [0]*n
    dominates_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if dominates(objs[i], objs[j]):
                dominates_list[i].append(j)
            elif dominates(objs[j], objs[i]):
                dominated_count[i] += 1
    fronts = []
    current = [i for i in range(n) if dominated_count[i] == 0]
    while current:
        fronts.append(current)
        next_front = []
        for p in current:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        current = next_front
    return fronts

def crowding_distance(front, objs):
    n = len(front)
    distances = {i: 0.0 for i in front}
    if n <= 2:
        for i in front:
            distances[i] = float('inf')
        return np.array([distances[i] for i in front])
    obj_count = len(objs[0])
    for m in range(obj_count):
        front_sorted = sorted(front, key=lambda idx: objs[idx][m])
        vals = [objs[idx][m] for idx in front_sorted]
        vmin, vmax = vals[0], vals[-1]
        denom = vmax - vmin if vmax != vmin else 1.0
        distances[front_sorted[0]] = float('inf')
        distances[front_sorted[-1]] = float('inf')
        for k in range(1, n - 1):
            prev_val = vals[k - 1]
            next_val = vals[k + 1]
            distances[front_sorted[k]] += (next_val - prev_val) / denom
    return np.array([distances[i] for i in front])

# ======== Operadores ========
def generate_pop(ind_size, pop_size):
    return np.random.rand(pop_size, ind_size)

def crossover(p1, p2):
    alpha = np.random.rand(*p1.shape)
    return alpha * p1 + (1 - alpha) * p2

def mutation(x, rate=0.1, delta=0.1):
    x = x.copy()
    for i in range(len(x)):
        if np.random.rand() < rate:
            x[i] = np.clip(x[i] + np.random.uniform(-delta, delta), 0, 1)
    return x

# ======== Métricas ========
def hypervolume(front_objs, ref=(1, 1)):
    if len(front_objs) == 0:
        return 0.0
    pts = sorted(front_objs, key=lambda x: x[0])
    hv = 0.0
    prev_f1 = ref[0]
    for f1_, f2_ in reversed(pts):
        width = prev_f1 - f1_
        height = ref[1] - f2_
        width = max(width, 0)
        height = max(height, 0)
        hv += width * height
        prev_f1 = f1_
    return hv

def spacing(front_objs):
    if len(front_objs) <= 1:
        return 0.0
    dists = []
    for i, a in enumerate(front_objs):
        others = [b for j, b in enumerate(front_objs) if j != i]
        min_dist = min(np.linalg.norm(np.array(a) - np.array(b)) for b in others)
        dists.append(min_dist)
    d_mean = np.mean(dists)
    return np.sqrt(np.sum((dists - d_mean) ** 2) / (len(dists) - 1))

# ======== NSGA-II (uma execução) ========
def run_nsga(pop_size, ind_size, generations, ref=(1, 10)):
    pop = generate_pop(ind_size, pop_size)
    hv_values, sp_values = [], []
    for gen in range(generations):
        objs = [[f1_ind(ind), f2_ind(ind)] for ind in pop]
        fronts = pareto_fronts(objs)
        best_front_objs = [objs[i] for i in fronts[0]] if fronts else []
        hv_values.append(hypervolume(best_front_objs, ref))
        sp_values.append(spacing(best_front_objs))
        # Geração de filhos
        children = [mutation(crossover(pop[np.random.randint(pop_size)], pop[np.random.randint(pop_size)]))
                    for _ in range(pop_size)]
        union = np.vstack((pop, np.array(children)))
        union_objs = [[f1_ind(ind), f2_ind(ind)] for ind in union]
        union_fronts = pareto_fronts(union_objs)
        new_pop_idx = []
        for front in union_fronts:
            if len(new_pop_idx) + len(front) <= pop_size:
                new_pop_idx.extend(front)
            else:
                remaining = pop_size - len(new_pop_idx)
                distances = crowding_distance(front, union_objs)
                order = np.argsort(distances)[::-1]
                picked = [front[i] for i in order[:remaining]]
                new_pop_idx.extend(picked)
                break
        pop = union[new_pop_idx]
    final_objs = [[f1_ind(ind), f2_ind(ind)] for ind in pop]
    final_fronts = pareto_fronts(final_objs)
    final_best = [final_objs[i] for i in final_fronts[0]] if final_fronts else []
    return final_best, np.array(hv_values), np.array(sp_values)

# ======== Wrapper para execução em processo separado (recebe tupla de args) ========
def run_single_task(task):
    """
    task: tuple (N, IND_SIZE, GENERATIONS, seed)
    retorna: (N, hv_array, sp_array)
    """
    N, ind_size, generations, seed = task
    # seed específica por processo/run:
    np.random.seed(int(seed))
    # executar NSGA-II
    _, hv, sp = run_nsga(N, ind_size, generations)
    return (N, hv, sp)

# ======== Função principal: organizar tasks, paralelizar, agregar e plotar ========
def main(n_values=N_VALUES, ind_size=IND_SIZE, generations=GENERATIONS, runs=RUNS, base_seed=BASE_SEED, outdir=OUTDIR, max_workers=MAX_WORKERS):
    # construir lista de tasks (cada run é independente)
    tasks = []
    idx = 0
    for N in n_values:
        for r in range(runs):
            seed = base_seed + idx
            tasks.append((N, ind_size, generations, seed))
            idx += 1

    cpu_count = multiprocessing.cpu_count()
    workers = max_workers if (max_workers is not None) else cpu_count
    print(f"Executando {len(tasks)} tasks em até {workers} workers (CPU: {cpu_count}) ...")

    # dicionário para armazenar hv/sp por N
    hv_runs_by_N = {N: [] for N in n_values}
    sp_runs_by_N = {N: [] for N in n_values}

    start_time = time.time()
    # Executa em paralelo
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_task, t): t for t in tasks}
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            N_res, hv_res, sp_res = fut.result()
            hv_runs_by_N[N_res].append(hv_res)
            sp_runs_by_N[N_res].append(sp_res)
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                print(f"  {completed}/{len(tasks)} tasks concluídas...")

    total_time = time.time() - start_time
    print(f"Paralelização finalizada em {total_time:.1f}s")

    results = {}
    for N in n_values:
        hv_array = np.array(hv_runs_by_N[N])  # shape (runs, generations)
        sp_array = np.array(sp_runs_by_N[N])
        hv_mean = np.mean(hv_array, axis=0)
        hv_std = np.std(hv_array, axis=0)
        sp_mean = np.mean(sp_array, axis=0)
        sp_std = np.std(sp_array, axis=0)
        results[N] = (hv_mean, hv_std, sp_mean, sp_std)

    # ======== Plot convergência média (hipervolume) ========
    plt.figure(figsize=(10,6))
    gens = np.arange(generations)
    for N, (hv_mean, hv_std, _, _) in results.items():
        plt.plot(gens, hv_mean, label=f'N={N}')
        plt.fill_between(gens, hv_mean - hv_std, hv_mean + hv_std, alpha=0.2)
    plt.title("Convergência Média do Hipervolume (paralelizado)")
    plt.xlabel("Geração")
    plt.ylabel("Hipervolume Médio ± 1σ")
    plt.legend()
    plt.grid(True)
    hv_path = os.path.join(outdir, "hv_convergencia_media_parallel.png")
    plt.savefig(hv_path, dpi=300)
    plt.close()
    print(f"Gráfico hipervolume salvo em: {hv_path}")

    # ======== Plot convergência média (spacing) ========
    plt.figure(figsize=(10,6))
    for N, (_, _, sp_mean, sp_std) in results.items():
        plt.plot(gens, sp_mean, label=f'N={N}')
        plt.fill_between(gens, sp_mean - sp_std, sp_mean + sp_std, alpha=0.2)
    plt.title("Convergência Média do Spacing (paralelizado)")
    plt.xlabel("Geração")
    plt.ylabel("Spacing Médio ± 1σ")
    plt.legend()
    plt.grid(True)
    sp_path = os.path.join(outdir, "spacing_convergencia_media_parallel.png")
    plt.savefig(sp_path, dpi=300)
    plt.close()
    print(f"Gráfico spacing salvo em: {sp_path}")

    print(f"✅ Todos os resultados salvos em: {outdir}")
    return results

# ======== Execução protegida (necessário para multiprocessing em Windows) ========
if __name__ == "__main__":
    results = main()
