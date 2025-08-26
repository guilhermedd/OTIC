from multiprocessing import Process, Array
from typing import Literal

import numpy as np
import os

class generic_genetic:
    def __init__(
        self, 
        num_threads: int,
        population_size: int,
        dimension: int,
        bounds: tuple[float, float],
        population_type: Literal['f', 'i', 'B']
    ) -> None:

        self.num_threads     = num_threads
        self.population_size = population_size
        self.dimension       = dimension
        self.bounds          = bounds
        self.population_type = population_type
        
        # array 1D contínuo (population_size * dimension)
        self.population = Array(population_type, population_size * dimension, lock=False)

    def _worker(self, idx):
        low, high = self.bounds
        if self.population_type in ('f', 'd'):   # float
            pop = np.random.uniform(low, high, self.dimension)
        elif self.population_type == 'i':        # int
            pop = np.random.randint(low, high, self.dimension)
        elif self.population_type == 'B':        # binary
            pop = np.random.randint(0, 2, self.dimension)
        else:
            raise ValueError("Invalid population type. Choose from 'f', 'd', 'i', or 'B'.")
        
        # escreve no array compartilhado (1D flatten)
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

            