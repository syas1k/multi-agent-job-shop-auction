"""Reproducible paired comparisons. Synthetic arrivals are not original JSPLIB data."""
import copy
import hashlib
import random
import time

from .agents.agents import JobAgent, MachineAgent, ManagerAgent
from .environment.environment import Environment, MachineState
from .environment.heuristics import run_dispatch
from .metrics.metrics import machine_utilization, makespan, total_tardiness
from .metrics.validate import validate_schedule

METHODS = ("FIFO", "EDD", "SPT", "MWKR", "CR", "ATC_LOCAL",
           "CENTRAL_ORIGINAL", "AUCTION_ORIGINAL", "AUCTION_ATC")


def make_scenario(jobs, scenario, due_factor=2.0, seed=0):
    jobs = copy.deepcopy(jobs)
    if scenario not in ("static", "due_dates", "arrivals"):
        raise ValueError(f"Unknown scenario {scenario}")
    if due_factor <= 0:
        raise ValueError("due_factor must be positive")
    rng = random.Random(seed)
    machine_ids = {o.machine_id for j in jobs for o in j.operations}
    total_work = sum(o.duration for j in jobs for o in j.operations)
    # Finite workload, not a steady-state queueing experiment.
    release_horizon = total_work / max(1, len(machine_ids))
    for job in jobs:
        job.release_time = rng.uniform(0.0, release_horizon) if scenario == "arrivals" else 0.0
        processing = sum(op.duration for op in job.operations)
        job.due_date = (None if scenario == "static"
                        else job.release_time + due_factor * processing)
    return jobs


def schedule_fingerprint(jobs):
    intervals = [(j.job_id, o.op_index, o.machine_id, o.start, o.end)
                 for j in sorted(jobs, key=lambda j: j.job_id) for o in j.operations]
    return hashlib.sha256(repr(intervals).encode()).hexdigest()


def evaluate(jobs, method, k=2.0):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    copied = copy.deepcopy(jobs)
    machines = [MachineState(mid) for mid in
                sorted({o.machine_id for j in copied for o in j.operations})]
    env = Environment(copied, machines)
    manager = None
    started = time.perf_counter()
    if method.startswith("AUCTION"):
        rule = "ORIGINAL" if method == "AUCTION_ORIGINAL" else "ATC_LOCAL"
        manager = ManagerAgent([JobAgent(j, rule, k) for j in copied],
                               [MachineAgent(m.machine_id) for m in machines])
        manager.run_until_done(env)
    else:
        run_dispatch(env, "ORIGINAL" if method == "CENTRAL_ORIGINAL" else method, k)
    runtime = time.perf_counter() - started
    violations = validate_schedule(copied)
    if violations or not env.is_done():
        raise AssertionError(f"{method}: invalid schedule: {violations}")
    span = makespan(copied)
    completions = [max((o.end for o in j.operations), default=j.release_time) for j in copied]
    has_due = all(j.due_date is not None for j in copied)
    row = {
        "method": method, "makespan": span,
        "total_tardiness": total_tardiness(copied) if has_due else None,
        "tardy_fraction": (sum(c > j.due_date for c, j in zip(completions, copied)) / len(copied)
                          if has_due else None),
        "mean_flow_time": sum(c - j.release_time for c, j in zip(completions, copied)) / len(copied),
        "mean_utilization": sum(machine_utilization(machines, span).values()) / len(machines),
        "runtime_seconds": runtime,
        "auctions": manager.auction_count if manager else 0,
        "bids": manager.bid_count if manager else 0,
        "valid": True, "schedule_sha256": schedule_fingerprint(copied),
    }
    return row, copied


def verify_controls(rows):
    by_method = {r["method"]: r for r in rows}
    for a, b in (("CENTRAL_ORIGINAL", "AUCTION_ORIGINAL"), ("ATC_LOCAL", "AUCTION_ATC")):
        if a in by_method and b in by_method:
            if by_method[a]["schedule_sha256"] != by_method[b]["schedule_sha256"]:
                raise AssertionError(f"Central/agent equivalence failed: {a}, {b}")
