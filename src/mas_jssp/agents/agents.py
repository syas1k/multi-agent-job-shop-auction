"""Cooperative single-round priority bidding, without prices or budgets."""
from dataclasses import dataclass

from ..environment.environment import Environment, Job, Operation
from ..environment.priority import priority


@dataclass
class Bid:
    operation: Operation
    value: float


class JobAgent:
    def __init__(self, job: Job, rule: str = "ORIGINAL", k: float = 2.0):
        self.job = job
        self.rule = rule
        self.k = k

    def make_bid(self, env: Environment, mean_processing: float = 1.0) -> Bid | None:
        op = self.job.ready_operation()
        if op is None or not env.can_start(op):
            return None
        return Bid(op, priority(env, op, self.rule, mean_processing, self.k))


class MachineAgent:
    def __init__(self, machine_id: int):
        self.machine_id = machine_id

    def run_auction(self, bids: list[Bid]) -> Bid | None:
        own = [b for b in bids if b.operation.machine_id == self.machine_id]
        return min(own, key=lambda b: (-b.value, b.operation.job_id,
                                      b.operation.op_index)) if own else None


class ManagerAgent:
    def __init__(self, job_agents: list[JobAgent], machine_agents: list[MachineAgent]):
        self.job_agents = job_agents
        self.machine_agents = machine_agents
        self.auction_count = 0
        self.bid_count = 0

    def run_round(self, env: Environment) -> list[Operation]:
        by_machine = {}
        for op in env.ready_operations():
            by_machine.setdefault(op.machine_id, []).append(op)
        means = {m: sum(o.duration for o in ops) / len(ops)
                 for m, ops in by_machine.items()}
        bids = []
        for ja in self.job_agents:
            op = ja.job.ready_operation()
            if op is not None and op.machine_id in means:
                bid = ja.make_bid(env, means[op.machine_id])
                if bid is not None:
                    bids.append(bid)
        self.bid_count += len(bids)
        winners = []
        for ma in sorted(self.machine_agents, key=lambda a: a.machine_id):
            winner = ma.run_auction(bids)
            if winner is not None:
                self.auction_count += 1
                winners.append(env.step(winner.operation))
        return winners

    def run_until_done(self, env: Environment) -> None:
        while not env.is_done():
            if not self.run_round(env):
                env.advance_time_to_next_event()
