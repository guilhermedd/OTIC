import numpy as np
import matplotlib.pyplot as plt
import os

def dominates(ind_a_objs, ind_b_objs):
    """Verifica se a solução A domina a solução B (problema de minimização)."""
    return all(a <= b for a, b in zip(ind_a_objs, ind_b_objs)) and any(a < b for a, b in zip(ind_a_objs, ind_b_objs))

import multiprocessing

# Supondo que sua função 'dominates' já existe
# def dominates(ind_a_objs, ind_b_objs): ...

def _calculate_dominance_for_one(args):
    """
    Função auxiliar que calcula a dominância para um único indivíduo.
    Projetada para ser usada com multiprocessing.Pool.
    """
    pop_idx, population_objs = args
    pop_size = len(population_objs)
    
    dominates_over_p = []
    gets_dominated_p = 0
    
    # Compara o indivíduo 'pop_idx' com todos os outros
    for compare_idx in range(pop_size):
        if pop_idx == compare_idx:
            continue
            
        if dominates(population_objs[pop_idx], population_objs[compare_idx]):
            dominates_over_p.append(compare_idx)
        elif dominates(population_objs[compare_idx], population_objs[pop_idx]):
            gets_dominated_p += 1
            
    return dominates_over_p, gets_dominated_p

def pareto_fronts(population_objs):
    """Versão paralelizada da função pareto_fronts."""
    pop_size = len(population_objs)
    fronts = [[]]
    tasks = [(i, population_objs) for i in range(pop_size)]

    with multiprocessing.Pool() as pool:
        results = pool.map(_calculate_dominance_for_one, tasks)
    dominates_over, gets_dominated = zip(*results)
    
    gets_dominated = list(gets_dominated)

    for i in range(pop_size):
        if gets_dominated[i] == 0:
            fronts[0].append(i)
            
    i = 0
    while fronts[i]:
        next_front = []
        for pop_idx in fronts[i]:
            for compare_idx in dominates_over[pop_idx]:
                gets_dominated[compare_idx] -= 1
                if gets_dominated[compare_idx] == 0:
                    next_front.append(compare_idx)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts

def crowding_distance(points):
    """Calcula a distância de aglomeração para um conjunto de pontos na mesma frente."""
    n_points = len(points)
    if n_points <= 2:
        return [float('inf')] * n_points

    distances = [0.0] * n_points
    indexed_points = list(enumerate(points))
    n_obj = len(points[0])

    for i in range(n_obj):
        indexed_points.sort(key=lambda x: x[1][i])
        min_val = indexed_points[0][1][i]
        max_val = indexed_points[-1][1][i]

        if max_val == min_val:
            continue

        distances[indexed_points[0][0]] = float('inf')
        distances[indexed_points[-1][0]] = float('inf')

        for j in range(1, n_points - 1):
            distances[indexed_points[j][0]] += (indexed_points[j + 1][1][i] - indexed_points[j - 1][1][i]) / (max_val - min_val)

    return distances

def f1(x: np.ndarray) -> float:
    """Primeira função objetivo do ZDT1."""
    return x[0]

def g(x: np.ndarray) -> float:
    """Função auxiliar g do ZDT1."""
    return 1 + 9 * np.sum(x[1:]) / (len(x) - 1)

def f2(x: np.ndarray) -> float:
    """Segunda função objetivo do ZDT1."""
    g_x = g(x)
    f1_x = f1(x)
    return g_x * (1 - np.sqrt(f1_x / g_x))

def generate_pop(individual_size: int, population_size: int) -> np.ndarray:
    """Gera a população inicial com valores entre 0 e 1."""
    return np.random.uniform(0, 1, (population_size, individual_size))

def crossover(parent_1, parent_2):
    """Crossover aritmético."""
    alpha = np.random.uniform(0, 1, size=parent_1.shape)
    return alpha * parent_1 + (1 - alpha) * parent_2

def mutation(offspring: np.ndarray, mutation_rate=0.1, delta=0.1) -> np.ndarray:
    """Mutação que adiciona um pequeno valor aleatório."""
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] += np.random.uniform(-delta, delta)
            offspring[i] = np.clip(offspring[i], 0, 1) 
    return offspring

def plot_generations(generation_obj_values, filename='geracoes.png'):
    """Plota a última geração, colorindo por frente de Pareto."""
    final_gen_objs = generation_obj_values[-1]
    fronts = pareto_fronts(final_gen_objs)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(fronts)))

    for i, front in enumerate(fronts):
        f1_vals = [final_gen_objs[idx][0] for idx in front]
        f2_vals = [final_gen_objs[idx][1] for idx in front]
        plt.scatter(f1_vals, f2_vals, color=colors[i], label=f'Frente {i+1}')
    
    plt.xlabel('f1(x) - Minimizar')
    plt.ylabel('f2(x) - Minimizar')
    plt.title('Frentes de Pareto da Última Geração')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    # plt.show()

def main():
    POPULATION_SIZE = 1000
    INDIVIDUAL_SIZE = 100
    GENERATIONS = 50

    population = generate_pop(INDIVIDUAL_SIZE, POPULATION_SIZE)
    generation_obj_values = []

    for i in range(GENERATIONS):
        print(f'Geração {i+1}/{GENERATIONS}')
        
        current_pop_objs = [ [f1(ind), f2(ind)] for ind in population ]
        generation_obj_values.append(current_pop_objs)

        children_pop = []
        for _ in range(POPULATION_SIZE):
            p1_idx, p2_idx = np.random.choice(range(POPULATION_SIZE), 2, replace=False)
            offspring = crossover(population[p1_idx], population[p2_idx])
            offspring = mutation(offspring)
            children_pop.append(offspring)
        
        combined_pop = np.vstack((population, np.array(children_pop)))
        
        combined_pop_objs = [ [f1(ind), f2(ind)] for ind in combined_pop ]
        
        fronts_idx = pareto_fronts(combined_pop_objs)

        new_population_indices = []
        for front in fronts_idx:
            if len(new_population_indices) + len(front) <= POPULATION_SIZE:
                new_population_indices.extend(front)
            else:
                population_diff = POPULATION_SIZE - len(new_population_indices)
                
                chosen_indices = np.random.choice(front, size=population_diff, replace=False)
                
                new_population_indices.extend(chosen_indices.tolist())
                break
        
        population = combined_pop[new_population_indices]

    plot_generations(generation_obj_values)

if __name__ == '__main__':
    main()