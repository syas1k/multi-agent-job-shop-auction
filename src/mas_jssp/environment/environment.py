"""Event-driven, non-preemptive JSSP environment; algorithms only start ready operations."""
from dataclasses import dataclass, field
from math import isfinite


@dataclass
class Operation:
    job_id: int
    op_index: int
    machine_id: int
    duration: float
    start: float | None = None
    end: float | None = None


@dataclass
class Job:
    job_id: int
    operations: list[Operation]
    due_date: float | None = None
    release_time: float = 0.0

    @property
    def next_op_index(self) -> int:
        """First unassigned operation. end is a scheduled completion time."""
        return next((i for i, op in enumerate(self.operations) if op.end is None),
                    len(self.operations))

    @property
    def is_finished(self) -> bool:
        """All operations assigned; Environment.is_done checks physical completion."""
        return self.next_op_index == len(self.operations)

    def ready_operation(self) -> Operation | None:
        idx = self.next_op_index
        return None if idx == len(self.operations) else self.operations[idx]

    @property
    def ready_at(self) -> float:
        idx = self.next_op_index
        return self.release_time if idx == 0 else self.operations[idx - 1].end


@dataclass
class MachineState:
    machine_id: int
    free_at: float = 0.0
    is_broken: bool = False
    broken_until: float = 0.0  # repair events not implemented
    busy_intervals: list[tuple[float, float]] = field(default_factory=list)


class Environment:
    def __init__(self, jobs: list[Job], machines: list[MachineState]):
        if len({j.job_id for j in jobs}) != len(jobs):
            raise ValueError("Duplicate job IDs")
        if len({m.machine_id for m in machines}) != len(machines):
            raise ValueError("Duplicate machine IDs")
        self.jobs = {j.job_id: j for j in jobs}
        self.machines = {m.machine_id: m for m in machines}
        self.time = 0.0
        for j in jobs:
            if not isfinite(j.release_time) or j.release_time < 0:
                raise ValueError("release_time must be finite and nonnegative")
            if j.due_date is not None and not isfinite(j.due_date):
                raise ValueError("due_date must be finite")
            for i, op in enumerate(j.operations):
                if op.job_id != j.job_id or op.op_index != i:
                    raise ValueError("Invalid operation identity/order")
                if op.machine_id not in self.machines:
                    raise ValueError("Unknown machine")
                if not isfinite(op.duration) or op.duration <= 0:
                    raise ValueError("duration must be finite and positive")
                if op.start is not None or op.end is not None:
                    raise ValueError("Environment requires unscheduled jobs")

    def can_start(self, operation: Operation) -> bool:
        job = self.jobs.get(operation.job_id)
        machine = self.machines.get(operation.machine_id)
        return bool(job is not None and machine is not None
                    and job.ready_operation() is operation
                    and operation.start is None and operation.end is None
                    and job.ready_at <= self.time
                    and not machine.is_broken and machine.free_at <= self.time)

    def ready_operations(self) -> list[Operation]:
        return [op for j in self.jobs.values()
                if (op := j.ready_operation()) is not None and self.can_start(op)]

    def is_done(self) -> bool:
        return all(j.is_finished and j.ready_at <= self.time for j in self.jobs.values())

    def step(self, operation: Operation) -> Operation:
        if not self.can_start(operation):
            raise ValueError("Operation cannot start at the current simulation time")
        machine = self.machines[operation.machine_id]
        operation.start = self.time
        operation.end = self.time + operation.duration
        machine.free_at = operation.end
        machine.busy_intervals.append((operation.start, operation.end))
        return operation

    def advance_time_to_next_event(self) -> None:
        candidates = [m.free_at for m in self.machines.values() if m.free_at > self.time]
        candidates.extend(j.release_time for j in self.jobs.values()
                          if j.release_time > self.time)
        if candidates:
            self.time = min(candidates)
        elif not self.is_done():
            raise RuntimeError("No future event: blocked schedule; repairs are not implemented")
