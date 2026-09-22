"""Shared dispatch indices. See docs/methods_and_sources.md for attribution."""
import math

from .environment import Environment, Operation

RULES = ("FIFO", "EDD", "SPT", "MWKR", "CR", "ATC_LOCAL", "ORIGINAL")


def priority(env: Environment, op: Operation, rule: str,
             mean_processing: float = 1.0, k: float = 2.0) -> float:
    """Higher is better. Resolve ties by (job_id, op_index)."""
    if rule not in RULES:
        raise ValueError(f"Unknown rule: {rule}")
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    job = env.jobs[op.job_id]
    remaining = sum(o.duration for o in job.operations[op.op_index:])
    due = job.due_date if job.due_date is not None else math.inf
    if rule == "FIFO":
        return -job.ready_at  # arrival into current machine queue, not job ID
    if rule == "EDD":
        return -due
    if rule == "SPT":
        return -op.duration
    if rule == "MWKR":
        return remaining
    if rule == "CR":
        return -(due - env.time) / remaining
    if rule == "ORIGINAL":
        urgency = 0.0 if job.due_date is None else 1.0 / max(due - env.time, 1e-6)
        return remaining + 100.0 * urgency
    # ATC adapted with local due date = due - downstream processing (unit weights).
    # log(index) preserves ranking without exponential underflow.
    # Missing due dates: no urgency, so reduce to SPT.
    slack = 0.0 if job.due_date is None else max(due - env.time - remaining, 0.0)
    return -math.log(op.duration) - slack / (k * mean_processing)
