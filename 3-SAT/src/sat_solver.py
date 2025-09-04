from typing import Literal
from multiprocessing import Process, Array, Pipe

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
        COD: Literal['BIN', 'REAL', 'INT', 'INT-PERM']
    ) -> None:

        self.data_file = data_file

        self.num_threads = num_threads
        self.bounds = bounds
        self.stop_on_first_answer = stop_on_first_answer
        self.population_size = POP
        self.dimension = DIM
        self.generations = GEN
        self.population_type = COD

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
            raise ValueError("Invalid population type. Choose from 'REAL', 'FLOAT', 'INT', 'BIN', or 'INT-PERM'")

        for j in range(self.dimension):
            self.population[idx * self.dimension + j] = int(pop[j])  # <-- cast here


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

            # Join after starting all threads
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
                return self.get_individual(good_individuals_idx), arr[good_individuals_idx]
            else:
                best_individual_idx = np.argmax(arr)
                best_individual = (self.get_individual(best_individual_idx), arr[best_individual_idx])

            done_pop += self.num_threads

        return best_individual


    def mutate(self):
        for i in range(self.population_size):
            start = i * self.dimension
            for j in range(self.dimension):
                if np.random.rand() < 0.01:
                    self.population[start + j] = 1 - self.population[start + j]

    def get_individual(self, idx):
        start = idx * self.dimension
        return self.population[start : start + self.dimension]

    def change_individual(self, idx, vals):
        start = idx * self.dimension

        val_idx = 0
        for i in range(start, start + self.dimension):
            self.population[i] = vals[val_idx]
            val_idx += 1

    def main(self):
        self.get_data()
        self.get_initial_population()

        best_individual, score = self.get_correct_individuals()
        
        executions = 0
        while not best_individual or executions < self.generations:
            self.mutate()
            
            best_individual, score = self.get_correct_individuals()

            executions += 1
            print("best_individuals", best_individual)
            print("scores", score)
            print("executions", executions)
        print("best_individuals final", best_individual)
        print("scores final", score)

