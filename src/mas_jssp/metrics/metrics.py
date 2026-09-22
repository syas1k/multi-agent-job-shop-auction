"""Metrics on completed schedules."""
from ..environment.environment import Job, MachineState


def makespan(jobs: list[Job]) -> float:
    return max((op.end for j in jobs for op in j.operations if op.end is not None),
               default=0.0)


def total_tardiness(jobs: list[Job]) -> float:
    return sum(max(0.0, max((op.end for op in j.operations), default=j.release_time)
                   - j.due_date)
               for j in jobs if j.due_date is not None and j.is_finished)


def machine_utilization(machines: list[MachineState], horizon: float) -> dict[int, float]:
    if horizon <= 0:
        return {m.machine_id: 0.0 for m in machines}
    return {m.machine_id: sum(max(0.0, min(end, horizon) - max(start, 0.0))
                             for start, end in m.busy_intervals) / horizon
            for m in machines}
