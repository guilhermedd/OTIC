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

        self.num_threads            = num_threads
        self.bounds                 = bounds
        self.stop_on_first_answer   = stop_on_first_answer
        self.population_size        = POP
        self.dimension              = DIM
        self.generations            = GEN
        self.population_type        = COD
        
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
            raise ValueError("Invalid population type. Choose from 'REAL', 'FLOAT', 'INT', 'BIN', or 'INT-PERM'.")

        for j in range(self.dimension):
            self.population[idx * self.dimension + j] = pop[j]

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
            done_pop += self.num_threads
        
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
            proposition_1 = individual[clause[0]] if clause[0] > 0 else not individual[-clause[0]]
            proposition_2 = individual[clause[1]] if clause[1] > 0 else not individual[-clause[1]]
            proposition_3 = individual[clause[2]] if clause[2] > 0 else not individual[-clause[2]]
            if (proposition_1 or proposition_2 or proposition_3):
                correct_clauses += 1

        good_individuals[idx] = correct_clauses

    def get_correct_individuals(self):
        good_individuals = Array('i', self.population_size, lock=False)

        done_pop = 0
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
                return self.population[good_individuals_idx]

            done_pop += self.num_threads
            
        for i in good_individuals:
            print(i)

        return []
    
    def mutate(self):
        pass

    def main(self):
        self.get_initial_population()
        self.get_data()
        
        good_individuals = self.get_correct_individuals()

        executions = 0
        while not good_individuals or executions < self.generations:
            self.mutate()
            good_individuals = self.get_correct_individuals()
            executions += 1