"""
src/mas_jssp/agents/agents.py

Агенты не трогают Environment.step() напрямую в обход правил — они только
формируют ставки и выбирают победителя. Само применение решения (step())
остаётся единой точкой правды в Environment, как и в heuristics.py.
"""

from dataclasses import dataclass

from ..environment.environment import Environment, Job, Operation


@dataclass
class Bid:
    operation: Operation
    value: float  # чем больше, тем выше приоритет — интерпретацию задаёт JobAgent


class JobAgent:
    """Представляет одну job. Формирует ставку для своей готовой операции."""

    def __init__(self, job: Job):
        self.job = job

    def make_bid(self, env: Environment) -> Bid | None:
        op = self.job.ready_operation()
        if op is None:
            return None
        # Простейшая версия: приоритет = оставшаяся работа + срочность (slack до due_date).
        remaining = sum(o.duration for o in self.job.operations[self.job.next_op_index:])
        slack = (self.job.due_date - env.time) if self.job.due_date is not None else float("inf")
        urgency = 1.0 / max(slack, 1e-6) if slack != float("inf") else 0.0
        value = remaining + 100.0 * urgency
        return Bid(operation=op, value=value)


class MachineAgent:
    """Представляет один станок. Объявляет аукцион среди готовых операций для этого станка."""

    def __init__(self, machine_id: int):
        self.machine_id = machine_id

    def run_auction(self, bids: list[Bid]) -> Bid | None:
        """Простейший sealed-bid, один раунд, побеждает максимальная ставка."""
        own_bids = [b for b in bids if b.operation.machine_id == self.machine_id]
        if not own_bids:
            return None
        return max(own_bids, key=lambda b: b.value)


class ManagerAgent:
    """Координирует один раунд: собирает ставки со всех JobAgent,
    прогоняет аукцион на каждом свободном станке, применяет победителей."""

    def __init__(self, job_agents: list[JobAgent], machine_agents: list[MachineAgent]):
        self.job_agents = job_agents
        self.machine_agents = machine_agents

    def run_round(self, env: Environment) -> list[Operation]:
        bids = [b for ja in self.job_agents if (b := ja.make_bid(env)) is not None]
        winners = []
        for ma in self.machine_agents:
            winner = ma.run_auction(bids)
            if winner is not None:
                winners.append(env.step(winner.operation))
        return winners

    def run_until_done(self, env: Environment) -> None:
        while not env.is_done():
            winners = self.run_round(env)
            if not winners:
                env.advance_time_to_next_event()
