import argparse
from src.sat_solver import SAT_Solver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAT Solver with multiprocessing")

    parser.add_argument("--num_threads", type=int, default=10, help="Number of threads to use")
    parser.add_argument("--data_file", type=str, default="data/100.txt", help="Path to the CNF/SAT data file")
    parser.add_argument("--lower_bound", type=float, default=0.0, help="Lower bound for initialization")
    parser.add_argument("--upper_bound", type=float, default=1.0, help="Upper bound for initialization")
    parser.add_argument("--stop-on-first-answer", type=bool, default=False, help="Stop on first valid answer")
    parser.add_argument("--population_size", type=int, default=30, help="Number of individuals in the population")
    parser.add_argument("--dimension", type=int, default=100, help="Number of variables (dimension of each individual)")
    parser.add_argument("--generations", type=int, default=100_000, help="Number of generations to run")
    parser.add_argument("--population_type", type=str, choices=['BIN', 'REAL', 'INT', 'INT-PERM'], default='BIN', help="Type of population: 'f'=float32, 'd'=float64, 'i'=int, 'B'=binary")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs to perform")

    args = parser.parse_args()

    for i in range(1):
        solver = SAT_Solver(
            num_threads=args.num_threads,
            data_file=args.data_file,
            bounds=(args.lower_bound, args.upper_bound),
            stop_on_first_answer=args.stop_on_first_answer,
            POP=args.population_size,
            DIM=args.dimension,
            GEN=args.generations,
            COD=args.population_type,
        )

        solver.main()

"""
RUN:
python3 main.py \
--num_threads 20 \
--data_file data/100.txt \
--population_size 30 \
--dimension 100 \
--lower_bound -5 \
--upper_bound 10 \
--population_type BIN \
--stop-on-first-answer True \
--generations 5 \
--runs 1
"""

