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
    parser.add_argument("--population_type", type=str, choices=['BIN', 'REAL', 'INT', 'INT-PERM'], default='BIN',
                        help="Type of population: 'BIN'=binary, 'REAL'=float, 'INT'=int, 'INT-PERM'=int permutation")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs to perform")
    parser.add_argument("--elitism", type=bool, default=True, help="Preserve best individual in next generation")
    parser.add_argument("--selection_method", type=str, choices=["roleta", "torneio"], default="roleta",
                        help="Selection method: 'roleta' or 'torneio'")
    parser.add_argument("--K", type=int, default=3, help="Number of competitors for stochastic tournament")
    parser.add_argument("--KP", type=float, default=0.75, help="Winning probability for stochastic tournament")

    args = parser.parse_args()

    for run in range(args.runs):
        print(f"\n===== Run {run+1}/{args.runs} =====")
        solver = SAT_Solver(
            num_threads=args.num_threads,
            data_file=args.data_file,
            bounds=(args.lower_bound, args.upper_bound),
            stop_on_first_answer=args.stop_on_first_answer,
            POP=args.population_size,
            DIM=args.dimension,
            GEN=args.generations,
            COD=args.population_type,
            elitismo=args.elitism,
            metodo_selecao=args.selection_method,
            K=args.K,
            KP=args.KP
        )

        solver.main()