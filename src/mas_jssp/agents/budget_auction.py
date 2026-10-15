"""Автономные агенты на правилах с условными кредитами и разными торгами.

Здесь нет Q-learning. Правило оценки фиксировано и одинаково для протоколов.
Отличается именно оплата и, для ценовых часов, последовательность раундов.
"""
import copy
from dataclasses import asdict

from ..auction.protocols import Bid, PROTOCOLS, clear_auction
from ..environment.environment import Environment, MachineState
from ..experiments import schedule_fingerprint
from ..metrics.metrics import makespan, total_tardiness
from ..metrics.validate import validate_schedule


class CreditJobAgent:
    def __init__(self, job, credits_per_operation):
        # Агент получает собственный заказ. Других заказов и Environment у него нет.
        self.job = job
        self.initial_budget = credits_per_operation * len(job.operations)
        self.budget = self.initial_budget
        self.paid = 0.0

    def make_bid(self, now):
        op = self.job.ready_operation()
        if op is None or self.job.ready_at > now:
            return None
        remaining_ops = self.job.operations[op.op_index:]
        remaining_work = sum(o.duration for o in remaining_ops)
        # На старте резервируем примерно равную долю бюджета на каждую операцию.
        allowance = self.budget / len(remaining_ops)
        if self.job.due_date is None:
            urgency = 1.0
        else:
            slack = self.job.due_date - now - remaining_work
            # Большой запас уменьшает ставку. Отрицательный запас увеличивает,
            # но не более чем вдвое: правило должно оставаться ограниченным.
            urgency = ((1 + min(1.0, max(-slack, 0.0) / remaining_work))
                       / (1 + max(slack, 0.0) / remaining_work))
        amount = min(self.budget, allowance * urgency)
        return op, Bid(self.job.job_id, amount)

    def settle(self, payment):
        # Оплачивается только победа. Нельзя потратить чужой или будущий бюджет.
        if payment < 0 or payment > self.budget + 1e-9:
            raise AssertionError("Payment exceeds available budget")
        self.budget = max(0.0, self.budget - payment)
        self.paid += payment


class AuctionMachineAgent:
    def __init__(self, machine_id, protocol, price_step, max_rounds):
        self.machine_id = machine_id
        self.protocol = protocol
        self.price_step = price_step
        self.max_rounds = max_rounds

    def select(self, bids):
        # Станок не получает весь цех: только участников собственной очереди.
        return clear_auction(bids, self.protocol, self.price_step, self.max_rounds)


def run_budget_auction(jobs, protocol="first_price", credits_per_operation=1.0,
                       price_step=0.1, max_rounds=1000, capture_trace=False):
    """Выполнить всю партию; нулевой бюджет НЕ исключает заказ из обработки."""
    import math
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown protocol")
    if not math.isfinite(credits_per_operation) or credits_per_operation < 0:
        raise ValueError("Initial credits must be finite and nonnegative")
    # Проверяем параметры даже для задачи без конкуренции.
    clear_auction([Bid(0, 0)], protocol, price_step, max_rounds)
    if not jobs or any(not j.operations for j in jobs):
        raise ValueError("Nonempty jobs required")
    copied = copy.deepcopy(jobs)  # исходный датасет не портится после запуска
    mids = sorted({o.machine_id for j in copied for o in j.operations})
    env = Environment(copied, [MachineState(m) for m in mids])
    orders = {j.job_id: CreditJobAgent(j, credits_per_operation) for j in copied}
    machines = {m: AuctionMachineAgent(m, protocol, price_step, max_rounds) for m in mids}
    trace = []
    auctions = contested = rounds = messages = capped = exits = 0
    while not env.is_done():
        queues = {}
        for actor in orders.values():
            proposal = actor.make_bid(env.time)
            if proposal is None:
                continue
            op, bid = proposal
            if env.can_start(op):
                queues.setdefault(op.machine_id, []).append((op, bid))
        if not queues:
            env.advance_time_to_next_event()
            continue
        for mid, candidates in sorted(queues.items()):
            bids = [bid for _, bid in candidates]
            award = machines[mid].select(bids)
            winner_op = next(op for op, bid in candidates if bid.job_id == award.winner_id)
            actor = orders[award.winner_id]
            before = actor.budget
            actor.settle(award.payment)
            env.step(winner_op)
            auctions += 1
            contested += int(len(bids) > 1)
            rounds += award.rounds
            messages += award.messages
            capped += int(award.reason == "round_cap_id_tie")
            exits += int(award.reason == "simultaneous_exit_id_tie")
            if capture_trace:
                trace.append({
                    "time": env.time, "machine": mid, "operation": winner_op.op_index,
                    "bids_or_private_limits": [asdict(b) for b in bids],
                    "award": asdict(award), "budget_before": before,
                    "budget_after": actor.budget,
                    "start": winner_op.start, "end": winner_op.end,
                })
    violations = validate_schedule(copied)
    if violations:
        raise AssertionError(violations)
    work = sum(o.duration for j in copied for o in j.operations)
    tardiness = total_tardiness(copied) if all(j.due_date is not None for j in copied) else None
    return {
        "method": protocol, "valid": True, "makespan": makespan(copied),
        "total_tardiness": tardiness,
        "normalized_tardiness": None if tardiness is None else tardiness / work,
        "completed_jobs": len(copied), "auctions": auctions,
        "contested_auctions": contested, "rounds": rounds,
        "logical_messages": messages, "round_caps": capped,
        "simultaneous_exits": exits, "credits_spent": sum(a.paid for a in orders.values()),
        "remaining_budgets": {jid: a.budget for jid, a in orders.items()},
        "schedule_sha256": schedule_fingerprint(copied), "trace": trace,
    }
