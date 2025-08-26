from dotenv import load_dotenv
import os

from src.sat_solver import SAT_Solver

if __name__ == "__main__":
    load_dotenv()
    
    NUM_THREADS     = int(os.getenv('NUM_THREADS'))
    DATA_FILE       = os.getenv('DATA_FILE')
    POPULATION_SIZE = int(os.getenv('POPULATION_SIZE'))
    DIMENSION       = int(os.getenv('DIMENSION'))
    BOUNDS          = (float(os.getenv('LOWER_BOUND')), float(os.getenv('UPPER_BOUND')))
    POPULATION_TYPE = os.getenv('POPULATION_TYPE')

    solver = SAT_Solver(
        num_threads=NUM_THREADS,
        data_file=DATA_FILE,
        population_size=POPULATION_SIZE,
        dimension=DIMENSION,
        bounds=BOUNDS,
        population_type=POPULATION_TYPE
    )
    
    print(solver.SAT)