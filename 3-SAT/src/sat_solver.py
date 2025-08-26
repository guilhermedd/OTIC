from typing import Literal
from multiprocessing import Process, Array, Pipe

import numpy as np
import os

from src.generic_genetic import generic_genetic

class SAT_Solver(generic_genetic):
    def __init__(
        self,
        num_threads: int,
        data_file: str,
        population_size: int,
        dimension: int,
        bounds: tuple[float, float],
        population_type: Literal['f', 'i', 'B']) -> None:
        
        super().__init__(
            num_threads=num_threads,
            population_size=population_size,
            dimension=dimension,
            bounds=bounds,
            population_type=population_type
        )
        
        self.data_file = data_file

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

        for clause in self.SAT:
            proposition_1 = individual[clause[0]] if clause[0] > 0 else not individual[-clause[0]]
            proposition_2 = individual[clause[1]] if clause[1] > 0 else not individual[-clause[1]]
            proposition_3 = individual[clause[2]] if clause[2] > 0 else not individual[-clause[2]]
            if not (proposition_1 or proposition_2 or proposition_3):
                good_individuals[idx] = 0
                return False

        good_individuals[idx] = 1
        return True

    def check_satisfiability(self):
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

            arr = np.frombuffer(good_individuals.get_obj(), dtype=np.int32)
            good_individuals_idx = np.where(arr == 1)[0]
            if len(good_individuals_idx) > 0:
                return good_individuals_idx

            done_pop += self.num_threads

        return []
    
    def mutate(self):
        pass

    def main(self):
        self.get_initial_population()
        self.get_data()
        
        good_individuals = self.check_satisfiability()
        
        while not good_individuals:
            self.mutate()
            good_individuals = self.check_satisfiability()