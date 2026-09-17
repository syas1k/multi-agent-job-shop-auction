"""
src/mas_jssp/metrics/metrics.py

Метрики считаются на уже завершённой (или частично завершённой) симуляции,
т.е. после того как все нужные Operation получили start/end.
"""

from ..environment.environment import Job, MachineState


def makespan(jobs: list[Job]) -> float:
    ends = [op.end for j in jobs for op in j.operations if op.end is not None]
    return max(ends) if ends else 0.0


def total_tardiness(jobs: list[Job]) -> float:
    total = 0.0
    for j in jobs:
        if j.due_date is None or not j.is_finished:
            continue
        completion = max(op.end for op in j.operations)
        total += max(0.0, completion - j.due_date)
    return total


def machine_utilization(machines: list[MachineState], horizon: float) -> dict[int, float]:
    """Доля времени [0, horizon], в течение которого станок был занят.
    Пока считается только по free_at — для точного расчёта нужно копить busy-интервалы
    по каждой операции (добавить при необходимости)."""
    if horizon <= 0:
        return {m.machine_id: 0.0 for m in machines}
    return {m.machine_id: min(1.0, m.free_at / horizon) for m in machines}
