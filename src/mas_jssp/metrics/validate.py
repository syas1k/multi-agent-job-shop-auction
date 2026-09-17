"""
src/mas_jssp/metrics/validate.py

Автоматическая проверка корректности готового расписания. Ловит именно тот
класс багов, который руками легко пропустить в логе: нарушение порядка
операций внутри job и наложение операций друг на друга на одном станке.

Использование:
    violations = validate_schedule(jobs)
    if violations:
        for v in violations:
            print("НАРУШЕНИЕ:", v)
"""

from ..environment.environment import Job


def validate_schedule(jobs: list[Job]) -> list[str]:
    violations = []

    # 1. Порядок операций внутри job: следующая не может начаться раньше,
    #    чем закончилась предыдущая.
    for job in jobs:
        for i in range(1, len(job.operations)):
            prev, cur = job.operations[i - 1], job.operations[i]
            if prev.end is None or cur.start is None:
                continue  # ещё не всё выполнено — не наш случай
            if cur.start < prev.end - 1e-9:
                violations.append(
                    f"Job {job.job_id}: op{i} стартует в {cur.start}, "
                    f"но op{i-1} заканчивается только в {prev.end}"
                )

    # 2. На одном станке операции не должны пересекаться по времени.
    by_machine: dict[int, list] = {}
    for job in jobs:
        for op in job.operations:
            if op.start is not None:
                by_machine.setdefault(op.machine_id, []).append(op)

    for machine_id, ops in by_machine.items():
        ops_sorted = sorted(ops, key=lambda o: o.start)
        for a, b in zip(ops_sorted, ops_sorted[1:]):
            if b.start < a.end - 1e-9:
                violations.append(
                    f"Machine {machine_id}: Job {a.job_id}/op занимает "
                    f"[{a.start},{a.end}], а Job {b.job_id}/op начинается в "
                    f"{b.start} — пересечение"
                )

    return violations
