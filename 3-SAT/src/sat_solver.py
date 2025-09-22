from typing import Literal
from multiprocessing import Process, Array
import numpy as np
import os


class SAT_Solver():
    def __init__(
        self,
        num_threads: int,
        data_file: str,
        bounds: tuple,
        stop_on_first_answer: bool,
        POP: int,
        DIM: int,
        GEN: int,
        COD: Literal['BIN', 'REAL', 'INT', 'INT-PERM'],
        elitismo: bool = True,
        metodo_selecao: Literal['roleta', 'torneio'] = 'roleta',
        K: int = 3,
        KP: float = 0.75
    ) -> None:

        self.data_file = data_file
        self.num_threads = num_threads
        self.bounds = bounds
        self.stop_on_first_answer = stop_on_first_answer
        self.population_size = POP
        self.dimension = DIM
        self.generations = GEN
        self.population_type = COD

        self.elitismo = elitismo
        self.metodo_selecao = metodo_selecao
        self.K = K
        self.KP = KP

        # array 1D contínuo (population_size * dimension)
        self.population = Array('i' if COD != 'BIN' else 'B', POP * DIM, lock=False)

    def _worker(self, idx):
        low, high = self.bounds
        if self.population_type in ('REAL', 'FLOAT'):
            pop = np.random.uniform(low, high, self.dimension)
        elif self.population_type == 'BIN':
            pop = np.random.randint(0, 2, self.dimension)
        elif self.population_type == 'INT':
            pop = np.random.randint(low, high, self.dimension)
        elif self.population_type == 'INT-PERM':
            pop = np.random.permutation(np.arange(low, high))
        else:
            raise ValueError("Invalid population type.")

        for j in range(self.dimension):
            self.population[idx * self.dimension + j] = int(pop[j])

    def get_initial_population(self):
        done_pop = 0
        while done_pop < self.population_size:
            processes = []
            for i in range(self.num_threads):
                idx = i + done_pop
                if idx >= self.population_size:
                    break
                process = Process(target=self._worker, args=(idx,))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            done_pop += len(processes)

    def get_data(self):
        self.SAT = []
        with open(self.data_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('p') or line.startswith('%') or line.startswith('0'):
                continue
            self.SAT.append([int(x) for x in line.split()[:-1]])

    def evaluate(self, idx, good_individuals):
        start = idx * self.dimension
        end = start + self.dimension
        individual = self.population[start:end]

        correct_clauses = 0
        for clause in self.SAT:
            proposition_1 = individual[clause[0] - 1] if clause[0] > 0 else not individual[-clause[0] - 1]
            proposition_2 = individual[clause[1] - 1] if clause[1] > 0 else not individual[-clause[1] - 1]
            proposition_3 = individual[clause[2] - 1] if clause[2] > 0 else not individual[-clause[2] - 1]
            if (proposition_1 or proposition_2 or proposition_3):
                correct_clauses += 1
        good_individuals[idx] = correct_clauses

    def get_correct_individuals(self):
        good_individuals = Array('i', self.population_size, lock=False)
        done_pop = 0
        best_individual = None
        done = False

        while done_pop < self.population_size:
            processes = []
            for i in range(self.num_threads):
                idx = i + done_pop
                if idx >= self.population_size:
                    break
                process = Process(target=self.evaluate, args=(idx, good_individuals))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            arr = np.fromiter(good_individuals, dtype=np.int32)
            good_individuals_idx = np.where(arr == len(self.SAT))[0]

            if len(good_individuals_idx) > 0:
                return self.get_individual(good_individuals_idx[0]), arr[good_individuals_idx[0]], arr[best_individual_idx] == len(self.SAT)
            else:
                best_individual_idx = np.argmax(arr)
                best_individual = (self.get_individual(best_individual_idx), arr[best_individual_idx])

            done_pop += self.num_threads

        return best_individual[0], best_individual[1], arr[best_individual_idx] == len(self.SAT)

    def roulete(self, scores, num_selecionados):
        scores = np.array(scores, dtype=np.float64)

        # Se todos os scores são zero, escolhe aleatoriamente
        if scores.sum() == 0:
            return np.random.choice(len(scores), num_selecionados, replace=False).tolist()

        # Evitar zeros para não quebrar a roleta
        scores = scores - scores.min() + 1e-6

        selecionados = []
        for _ in range(num_selecionados):
            total = scores.sum()
            if total == 0:  # fallback caso zere durante o processo
                restantes = [i for i in range(len(scores)) if scores[i] > -1]
                selecionados += np.random.choice(restantes, num_selecionados - len(selecionados), replace=False).tolist()
                break

            probs = scores / total
            idx = np.random.choice(len(scores), p=probs)
            selecionados.append(idx)
            scores[idx] = 0  # remove da roleta

        return selecionados


    def torneio_estocastico(self, scores, num_selecionados, K=3, KP=0.75):
        selecionados = []
        for _ in range(num_selecionados):
            competidores = np.random.choice(len(scores), K, replace=False)
            compet_scores = scores[competidores]
            ordem = np.argsort(-compet_scores)
            competidores = competidores[ordem]
            for idx in competidores:
                if np.random.rand() < KP:
                    selecionados.append(idx)
                    break
        return selecionados

    def criar_pop_intermediaria(self, scores):
        if self.metodo_selecao == "roleta":
            selecionados = self.roulete(scores, self.population_size)
        else:
            selecionados = self.torneio_estocastico(scores, self.population_size, self.K, self.KP)

        nova_pop = []
        for i in range(0, len(selecionados), 2):
            p1 = self.get_individual(selecionados[i])
            p2 = self.get_individual(selecionados[(i+1) % len(selecionados)])
            filho = [(p1[j] + p2[j]) // 2 for j in range(self.dimension)]
            nova_pop.append(filho)

        for i, ind in enumerate(nova_pop):
            self.change_individual(i, ind)

    def mutate(self):
        for i in range(self.population_size):
            start = i * self.dimension
            for j in range(self.dimension):
                if np.random.rand() < 0.01:
                    self.population[start + j] = 1 - self.population[start + j]

    def get_individual(self, idx):
        start = idx * self.dimension
        return self.population[start: start + self.dimension]

    def change_individual(self, idx, vals):
        start = idx * self.dimension
        for j, val in enumerate(vals):
            self.population[start + j] = val

    def main(self):
        self.get_data()
        self.get_initial_population()

        best_individual, score, done = None, None, False

        executions = 0
        while (not best_individual or executions < self.generations) and not done:
            # Avaliar todos os indivíduos
            scores_array = Array('i', self.population_size, lock=False)
            processes = []
            for i in range(self.population_size):
                process = Process(target=self.evaluate, args=(i, scores_array))
                processes.append(process)
                process.start()
                
            for process in processes:
                process.join()
            scores = np.frombuffer(scores_array, dtype=np.int32)

            # Guardar o melhor para elitismo
            if self.elitismo:
                melhor = best_individual

            # Criar nova população
            self.criar_pop_intermediaria(scores)

            # Restaurar o melhor indivíduo
            if self.elitismo and melhor is not None:
                self.change_individual(0, melhor)

            # Mutação
            self.mutate()

            # Avaliar novamente para pegar melhor indivíduo
            best_individual, score, done = self.get_correct_individuals()

            executions += 1
            print("Geração:", executions, "Melhor Score:", score)

        print("Melhor indivíduo final:", best_individual)
        print("Score final:", score)

