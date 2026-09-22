"""Run all dispatch policies on one normalized JSPLIB instance."""
import argparse

from mas_jssp.experiments import METHODS, evaluate, make_scenario, verify_controls
from mas_jssp.utils.loaders import load_jssp_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("n_machines", type=int, nargs="?", help="legacy optional consistency check")
    parser.add_argument("--scenario", choices=["static", "due_dates", "arrivals"], default="static")
    parser.add_argument("--due-factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=float, default=2.0)
    args = parser.parse_args()
    jobs = load_jssp_instance(args.path)
    count = max(o.machine_id for j in jobs for o in j.operations) + 1
    if args.n_machines is not None and args.n_machines != count:
        parser.error("n_machines does not match the instance")
    jobs = make_scenario(jobs, args.scenario, args.due_factor, args.seed)
    rows = []
    for method in METHODS:
        row, _ = evaluate(jobs, method, args.k)
        rows.append(row)
        tardy = "N/A" if row["total_tardiness"] is None else f"{row['total_tardiness']:.2f}"
        print(f"{method:>18}: Cmax={row['makespan']:.2f} tardiness={tardy} "
              f"runtime={row['runtime_seconds']:.4f}s valid={row['valid']}")
    verify_controls(rows)


if __name__ == "__main__":
    main()
