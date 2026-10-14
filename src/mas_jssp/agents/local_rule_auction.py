"""Rule-based local agents. No policy receives the global Environment.

Jobs disclose duration and slack, machines announce local ATC conditions.
The broker routes messages and executes awards; it does not rank candidates.
"""
import copy
import math
from dataclasses import asdict, dataclass
from statistics import fmean

from ..environment.environment import Environment, Job, MachineState
from ..experiments import schedule_fingerprint
from ..metrics.metrics import makespan, total_tardiness
from ..metrics.validate import validate_schedule


@dataclass(frozen=True)
class Proposal:
    job_id: int
    op_index: int
    machine_id: int
    duration: float
    slack: float


@dataclass(frozen=True)
class Announcement:
    machine_id: int
    k: float
    mean_processing: float
    urgent_fraction: float


@dataclass(frozen=True)
class Offer:
    proposal: Proposal
    log_priority: float


class LocalJobAgent:
    """Own order only. Disclosure is explicit in Proposal."""

    def __init__(self, job: Job):
        self.job = job
        self.wins = 0
        self.losses = 0

    def receive_result(self, won):
        self.wins += int(won)
        self.losses += int(not won)

    def propose(self, now):
        op = self.job.ready_operation()
        if op is None or self.job.ready_at > now:
            return None
        remaining = sum(o.duration for o in self.job.operations[op.op_index:])
        return Proposal(self.job.job_id, op.op_index, op.machine_id, op.duration,
                        self.job.due_date - now - remaining)

    def bid(self, proposal, announcement):
        if proposal.job_id != self.job.job_id:
            raise ValueError("Cannot bid for another job")
        if proposal.machine_id != announcement.machine_id:
            raise ValueError("Wrong machine announcement")
        score = -math.log(proposal.duration) - max(proposal.slack, 0.0) / (
            announcement.k * announcement.mean_processing)
        return Offer(proposal, score)


class LocalMachineAgent:
    """Only own queue messages, with independent announcement history.

    Local rule: if at least half the queue has slack <= one mean operation,
    use urgent_k; otherwise use relaxed_k. This is an experimental heuristic.
    """

    def __init__(self, machine_id, policy="local", fixed_k=2.0,
                 urgent_k=0.5, relaxed_k=3.0, urgent_threshold=0.5):
        if policy not in ("local", "fixed"):
            raise ValueError("Unknown machine policy")
        if any(not math.isfinite(k) or k <= 0 for k in
               (fixed_k, urgent_k, relaxed_k)):
            raise ValueError("ATC parameters must be positive and finite")
        if not math.isfinite(urgent_threshold) or not 0 <= urgent_threshold <= 1:
            raise ValueError("Invalid urgent fraction threshold")
        self.machine_id = machine_id
        self.policy = policy
        self.fixed_k = fixed_k
        self.urgent_k = urgent_k
        self.relaxed_k = relaxed_k
        self.urgent_threshold = urgent_threshold
        self.k_counts = {}

    def announce(self, queue):
        if not queue or any(p.machine_id != self.machine_id for p in queue):
            raise ValueError("Expected own nonempty queue")
        mean_p = fmean(p.duration for p in queue)
        urgent = sum(p.slack <= mean_p for p in queue) / len(queue)
        k = self.fixed_k if self.policy == "fixed" else (
            self.urgent_k if urgent >= self.urgent_threshold else self.relaxed_k)
        key = str(k)
        self.k_counts[key] = self.k_counts.get(key, 0) + 1
        return Announcement(self.machine_id, k, mean_p, urgent)

    def award(self, offers):
        if not offers or any(o.proposal.machine_id != self.machine_id for o in offers):
            raise ValueError("Expected offers for own machine")
        return min(offers, key=lambda o: (
            -o.log_priority, o.proposal.job_id, o.proposal.op_index)).proposal


def run_local_rules(jobs, policy="local", fixed_k=2.0, urgent_k=0.5,
                    relaxed_k=3.0, urgent_threshold=0.5, capture_trace=False):
    if not jobs or any(not j.operations or j.due_date is None for j in jobs):
        raise ValueError("Nonempty jobs with deadlines required")
    copied = copy.deepcopy(jobs)
    mids = sorted({o.machine_id for j in copied for o in j.operations})
    env = Environment(copied, [MachineState(mid) for mid in mids])
    job_agents = {j.job_id: LocalJobAgent(j) for j in copied}
    machine_agents = {
        mid: LocalMachineAgent(mid, policy, fixed_k, urgent_k, relaxed_k,
                               urgent_threshold) for mid in mids
    }
    trace = []
    auctions = conflicts = changed_winners = messages = 0
    while not env.is_done():
        queues = {}
        # Routing uses availability, not priorities. Policies see no global state.
        for actor in job_agents.values():
            proposal = actor.propose(env.time)
            if proposal is None:
                continue
            machine = env.machines[proposal.machine_id]
            if machine.free_at <= env.time and not machine.is_broken:
                queues.setdefault(proposal.machine_id, []).append(proposal)
        if not queues:
            env.advance_time_to_next_event()
            continue
        for mid, queue in sorted(queues.items()):
            machine_actor = machine_agents[mid]
            announcement = machine_actor.announce(tuple(queue))
            offers = tuple(job_agents[p.job_id].bid(p, announcement) for p in queue)
            winner = machine_actor.award(offers)

            # Diagnostic only: fixed-k winner in the SAME current queue.
            # It is not a complete counterfactual baseline schedule.
            control_announcement = Announcement(
                mid, fixed_k, announcement.mean_processing, announcement.urgent_fraction)
            control_offers = tuple(
                job_agents[p.job_id].bid(p, control_announcement) for p in queue)
            control = machine_actor.award(control_offers)
            differs = winner.job_id != control.job_id
            changed_winners += int(differs)
            conflicts += int(len(queue) > 1)
            auctions += 1
            # Logical directed deliveries: proposal, announcement, bid, outcome
            # to/from every candidate. Diagnostic calls are not protocol messages.
            messages += 4 * len(queue)
            for proposal in queue:
                job_agents[proposal.job_id].receive_result(
                    proposal.job_id == winner.job_id)
            operation = env.jobs[winner.job_id].operations[winner.op_index]
            env.step(operation)
            if capture_trace:
                trace.append({
                    "time": env.time, "announcement": asdict(announcement),
                    "proposals": [asdict(p) for p in queue],
                    "offers": [asdict(o) for o in offers],
                    "winner_job": winner.job_id, "winner_operation": winner.op_index,
                    "start": operation.start, "end": operation.end,
                    "fixed_k_shadow_winner_job": control.job_id,
                    "changed_winner": differs,
                })
    violations = validate_schedule(copied)
    if violations:
        raise AssertionError(violations)
    work = sum(o.duration for j in copied for o in j.operations)
    tardiness = total_tardiness(copied)
    return {
        "policy": policy, "valid": True, "makespan": makespan(copied),
        "total_tardiness": tardiness, "normalized_tardiness": tardiness / work,
        "completed_jobs": len(copied),
        "auctions": auctions, "contested_auctions": conflicts,
        "changed_winners_vs_fixed_on_same_queue": changed_winners,
        "logical_message_deliveries": messages,
        "machine_k_counts": {mid: a.k_counts for mid, a in machine_agents.items()},
        "job_outcomes": {jid: {"wins": a.wins, "losses": a.losses}
                         for jid, a in job_agents.items()},
        "schedule_sha256": schedule_fingerprint(copied), "trace": trace,
    }
