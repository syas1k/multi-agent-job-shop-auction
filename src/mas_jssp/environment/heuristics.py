"""
src/mas_jssp/environment/heuristics.py

Baseline-правила диспетчеризации. Не используют агентов и аукцион —
нужны только чтобы проверить, что Environment работает корректно,
и как точка сравнения для будущего аукционного механизма.
"""

from .environment import Environment, Operation


def run_fifo(env: Environment) -> None:
    """Из готовых операций всегда берём ту, чья job имеет меньший job_id
    (эмулирует порядок поступления)."""
    _run(env, key=lambda op: op.job_id)


def run_edd(env: Environment) -> None:
    """Приоритет — job с более ранним due_date."""
    _run(env, key=lambda op: env.jobs[op.job_id].due_date or float("inf"))


def _run(env: Environment, key) -> None:
    while not env.is_done():
        ready = env.ready_operations()
        if not ready:
            env.advance_time_to_next_event()
            continue
        # Группируем по станку: на каждом свободном станке выбираем
        # одну операцию по приоритету key, чтобы не назначать две сразу на один станок.
        by_machine: dict[int, list[Operation]] = {}
        for op in ready:
            by_machine.setdefault(op.machine_id, []).append(op)
        for ops in by_machine.values():
            chosen = min(ops, key=key)
            env.step(chosen)
