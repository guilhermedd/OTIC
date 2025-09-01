import argparse
from src.sat_solver import SAT_Solver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAT Solver with multiprocessing")

    parser.add_argument("--num_threads", type=int, required=True,
                        help="Number of threads to use")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the CNF/SAT data file")
    parser.add_argument("--population_size", type=int, required=True,
                        help="Number of individuals in the population")
    parser.add_argument("--dimension", type=int, required=True,
                        help="Number of variables (dimension of each individual)")
    parser.add_argument("--lower_bound", type=float, default=0.0,
                        help="Lower bound for initialization")
    parser.add_argument("--upper_bound", type=float, default=1.0,
                        help="Upper bound for initialization")
    parser.add_argument("--population_type", type=str, choices=['BIN', 'REAL', 'INT', 'INT-PERM'], required=True,
                        help="Type of population: 'f'=float32, 'd'=float64, 'i'=int, 'B'=binary")
    parser.add_argument("--stop-on-first-answer", type=bool, default=False,
                        help="Stop on first valid answer")
    parser.add_argument("--generations", type=int, required=True,
                        help="Number of generations to run")
    parser.add_argument("--runs", type=int, default=1, required=True,
                        help="Number of independent runs to perform")

    args = parser.parse_args()
    
    for i in range(args.runs):
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