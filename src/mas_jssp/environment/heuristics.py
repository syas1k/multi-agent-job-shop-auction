"""Non-delay dispatching using the same environment as the agents."""
from .environment import Environment
from .priority import RULES, priority


def run_dispatch(env: Environment, rule: str, k: float = 2.0) -> None:
    if rule not in RULES:
        raise ValueError(f"Unknown rule: {rule}")
    while not env.is_done():
        ready = env.ready_operations()
        if not ready:
            env.advance_time_to_next_event()
            continue
        by_machine = {}
        for op in ready:
            by_machine.setdefault(op.machine_id, []).append(op)
        for machine_id in sorted(by_machine):
            ops = by_machine[machine_id]
            mean_p = sum(op.duration for op in ops) / len(ops)
            chosen = min(ops, key=lambda op: (
                -priority(env, op, rule, mean_p, k), op.job_id, op.op_index))
            env.step(chosen)


def run_fifo(env: Environment) -> None:
    run_dispatch(env, "FIFO")


def run_edd(env: Environment) -> None:
    run_dispatch(env, "EDD")
