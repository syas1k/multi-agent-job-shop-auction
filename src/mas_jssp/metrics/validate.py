"""Independent validation of complete non-preemptive schedules."""
from math import isfinite

from ..environment.environment import Job


def validate_schedule(jobs: list[Job]) -> list[str]:
    violations = []
    by_machine = {}
    for job in jobs:
        previous_end = job.release_time
        for i, op in enumerate(job.operations):
            label = f"Job {job.job_id}/op{i}"
            if op.job_id != job.job_id or op.op_index != i:
                violations.append(f"{label}: invalid identity")
            if op.start is None or op.end is None:
                violations.append(f"{label}: missing interval")
                continue
            if not all(isfinite(v) for v in (op.start, op.end, op.duration)):
                violations.append(f"{label}: non-finite interval")
                continue
            if op.duration <= 0 or abs(op.end - op.start - op.duration) > 1e-9:
                violations.append(f"{label}: invalid duration")
            if op.start < previous_end - 1e-9:
                violations.append(f"{label}: precedence/release violation")
            previous_end = op.end
            by_machine.setdefault(op.machine_id, []).append(op)
    for mid, ops in by_machine.items():
        ordered = sorted(ops, key=lambda o: o.start)
        for a, b in zip(ordered, ordered[1:]):
            if b.start < a.end - 1e-9:
                violations.append(f"Machine {mid}: overlapping operations")
    return violations
