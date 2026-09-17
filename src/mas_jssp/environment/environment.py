"""
src/mas_jssp/environment/environment.py

Детерминированное, событийно-ориентированное ядро симуляции JSSP.
Не знает ничего про агентов и аукцион: просто отвечает на вопрос
"что произойдёт, если операция X будет назначена на станок M в момент t".
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Operation:
    job_id: int
    op_index: int          # порядковый номер операции внутри job (порядок фиксирован)
    machine_id: int
    duration: float
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class Job:
    job_id: int
    operations: list[Operation]
    due_date: Optional[float] = None  # нужно для EDD и tardiness

    @property
    def next_op_index(self) -> int:
        """Индекс первой ещё не выполненной операции (или len(), если job завершена)."""
        for i, op in enumerate(self.operations):
            if op.end is None:
                return i
        return len(self.operations)

    @property
    def is_finished(self) -> bool:
        return self.next_op_index == len(self.operations)

    def ready_operation(self) -> Optional[Operation]:
        """Операция, готовая к назначению прямо сейчас, либо None, если job уже вся выполнена."""
        idx = self.next_op_index
        return None if idx == len(self.operations) else self.operations[idx]

    @property
    def ready_at(self) -> float:
        """Момент, когда job освобождается для следующей операции — это момент
        окончания её ПРЕДЫДУЩЕЙ операции (или 0.0, если это первая операция job).
        Без этого следующая операция может назначиться раньше, чем реально
        освободилась job — именно такой баг был в выводе."""
        idx = self.next_op_index
        if idx == 0:
            return 0.0
        prev = self.operations[idx - 1]
        return prev.end if prev.end is not None else float("inf")


@dataclass
class MachineState:
    machine_id: int
    free_at: float = 0.0          # момент освобождения станка
    is_broken: bool = False       # задел под динамику (отказы станков)
    broken_until: float = 0.0


class Environment:
    """
    Владеет состоянием: время, станки, jobs.
    Не решает, ЧТО назначить — только применяет решение и продвигает время.
    """

    def __init__(self, jobs: list[Job], machines: list[MachineState]):
        self.jobs = {j.job_id: j for j in jobs}
        self.machines = {m.machine_id: m for m in machines}
        self.time: float = 0.0

    # --- запросы состояния, которые понадобятся и эвристикам, и агентам ---

    def ready_operations(self) -> list[Operation]:
        """Операции, чей станок свободен И чья job уже закончила предыдущую операцию."""
        result = []
        for job in self.jobs.values():
            op = job.ready_operation()
            if op is None:
                continue
            if job.ready_at > self.time:
                continue  # job ещё выполняет предыдущую операцию — рано
            m = self.machines[op.machine_id]
            if not m.is_broken and m.free_at <= self.time:
                result.append(op)
        return result

    def is_done(self) -> bool:
        return all(j.is_finished for j in self.jobs.values())

    # --- единственная точка, где состояние реально меняется ---

    def step(self, operation: Operation) -> Operation:
        """
        Назначить конкретную операцию на её станок начиная с max(self.time, станок свободен).
        Кто выбрал именно эту операцию (эвристика, агент, аукцион) — Environment не касается.
        """
        machine = self.machines[operation.machine_id]
        job = self.jobs[operation.job_id]
        start = max(self.time, machine.free_at, job.ready_at)
        operation.start = start
        operation.end = start + operation.duration
        machine.free_at = operation.end
        return operation

    def advance_time_to_next_event(self) -> None:
        """
        Если сейчас нет готовых операций (все станки заняты/сломаны),
        продвинуть время до ближайшего момента освобождения станка.
        """
        candidates = [m.free_at for m in self.machines.values() if m.free_at > self.time]
        if candidates:
            self.time = min(candidates)
