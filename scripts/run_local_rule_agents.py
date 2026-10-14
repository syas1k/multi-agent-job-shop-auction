"""Compare local rule agents with fixed ATC on identical generated scenarios."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from statistics import fmean

from mas_jssp.agents.local_rule_auction import run_local_rules
from mas_jssp.experiments import evaluate
from mas_jssp.learning_data import generate_scenario, scenario_hash
from mas_jssp.utils.loaders import load_jssp_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--seed-base", type=int, default=410000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--fixed-k", type=float, default=2.0)
    parser.add_argument("--urgent-k", type=float, default=0.5)
    parser.add_argument("--relaxed-k", type=float, default=3.0)
    parser.add_argument("--urgent-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    if len({p.resolve() for p in args.instances}) != len(args.instances):
        parser.error("Duplicate instance paths")
    # Validate policy parameters before reserving the output directory.
    from mas_jssp.agents.local_rule_auction import LocalMachineAgent
    LocalMachineAgent(0, "local", args.fixed_k, args.urgent_k,
                      args.relaxed_k, args.urgent_threshold)
    bases = [(p, load_jssp_instance(p)) for p in args.instances]
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src" / "mas_jssp").rglob("*.py")) + [Path(__file__).resolve()]
    config = {
        "status": "exploratory pilot, not a held-out confirmatory experiment",
        "python": platform.python_version(),
        "seed_base": args.seed_base, "seeds": args.seeds,
        "fixed_k": args.fixed_k, "urgent_k": args.urgent_k,
        "relaxed_k": args.relaxed_k, "urgent_threshold": args.urgent_threshold,
        "instances": [{"name": p.stem, "path": str(p),
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p, _ in bases],
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(
            p.read_bytes()).hexdigest() for p in sources},
        "runtime_note": "Local modes include same-queue shadow diagnostic; not a fair speed benchmark.",
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    example = None
    for path, base in bases:
        for seed in range(args.seed_base, args.seed_base + args.seeds):
            jobs = generate_scenario(base, path.stem, seed)
            started = time.perf_counter()
            baseline, _ = evaluate(jobs, "ATC_LOCAL", args.fixed_k)
            baseline_runtime = time.perf_counter() - started
            common = {"instance": path.stem, "seed": seed,
                      "scenario_sha256": scenario_hash(jobs)}
            work = sum(o.duration for j in jobs for o in j.operations)
            rows.append({**common, "method": "ATC_LOCAL",
                         "makespan": baseline["makespan"],
                         "total_tardiness": baseline["total_tardiness"],
                         "normalized_tardiness": baseline["total_tardiness"] / work,
                         "valid": baseline["valid"], "runtime_seconds": baseline_runtime})
            for mode in ("fixed", "local"):
                started = time.perf_counter()
                result = run_local_rules(
                    jobs, mode, args.fixed_k, args.urgent_k, args.relaxed_k,
                    args.urgent_threshold, capture_trace=(example is None and mode == "local"))
                runtime = time.perf_counter() - started
                trace = result.pop("trace")
                if mode == "fixed":
                    # Exact control: not merely equal objective values.
                    if result["schedule_sha256"] != baseline["schedule_sha256"]:
                        raise AssertionError("Fixed distributed control differs from ATC")
                if example is None and mode == "local":
                    example = {**common, "trace": trace}
                rows.append({**common, "method": "LOCAL_FIXED" if mode == "fixed" else "LOCAL_RULES",
                             **result, "runtime_seconds": runtime})
    summary = {}
    for method in ("ATC_LOCAL", "LOCAL_FIXED", "LOCAL_RULES"):
        selected = [r for r in rows if r["method"] == method]
        summary[method] = {
            "scenarios": len(selected),
            **{key: fmean(r[key] for r in selected) for key in
               ("makespan", "total_tardiness", "normalized_tardiness")},
            "all_valid": all(r["valid"] for r in selected),
        }
    local = [r for r in rows if r["method"] == "LOCAL_RULES"]
    summary["diagnostics"] = {
        "changed_winners_on_own_queues": sum(
            r["changed_winners_vs_fixed_on_same_queue"] for r in local),
        "contested_auctions": sum(r["contested_auctions"] for r in local),
        "note": "Same-queue shadow comparison; not an estimate of causal improvement.",
    }
    for filename, payload in (("runs.json", rows), ("summary.json", summary),
                              ("example_trace.json", example)):
        (args.output / filename).write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
