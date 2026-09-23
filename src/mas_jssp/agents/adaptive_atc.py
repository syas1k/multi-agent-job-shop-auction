"""Tabular adaptive ATC coordinator over the existing auction agents.

An event-based, partially observed control problem: aggregate queue features are
not a sufficient Markov state. This is not a distributed MARL implementation.
"""
import copy
import json
import math
import random
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev

from .agents import JobAgent, MachineAgent, ManagerAgent
from ..environment.environment import Environment, MachineState
from ..metrics.metrics import makespan, total_tardiness
from ..metrics.validate import validate_schedule

K_GRID = tuple(i / 2 for i in range(1, 11))
STATE_VERSION = 1


def accrued_tardiness(env, time=None):
    """Integral of the number of overdue unfinished jobs up to time.

    Scheduled end times are clipped at time, so future completion is not
    prematurely counted. Finished jobs stop accumulating lateness.
    """
    time = env.time if time is None else time
    total = 0.0
    for job in env.jobs.values():
        if job.due_date is None:
            raise ValueError("Adaptive ATC requires a due date for every job")
        if job.release_time > time:
            continue
        completion = job.ready_at if job.is_finished else time
        total += max(0.0, min(time, completion) - job.due_date)
    return total


def observation(env):
    ready = env.ready_operations()
    if not ready:
        raise ValueError("No decision is available")
    ratios, deadlines = [], []
    for op in ready:
        job = env.jobs[op.job_id]
        remaining = sum(o.duration for o in job.operations[op.op_index:])
        ratios.append((job.due_date - env.time - remaining) / remaining)
        deadlines.append(job.due_date - env.time)
    mean_p = fmean(op.duration for op in ready)
    free = sum(not m.is_broken and m.free_at <= env.time for m in env.machines.values())
    features = (
        fmean(ratios),
        1 - free / len(env.machines),
        pstdev(deadlines) / mean_p,
        sum(r <= 0 for r in ratios) / len(ratios),
        len(ready) / max(1, free),
    )
    boundaries = ((-0.5, 0.0, 1.0), (0.33, 0.66), (2.0, 5.0),
                  (0.01, 0.5), (1.01, 2.01, 4.01))
    return tuple(bisect_right(bins, value) for bins, value in zip(boundaries, features))


class DispatchEpisode:
    def __init__(self, jobs):
        jobs = copy.deepcopy(jobs)
        if not jobs or any(not j.operations or j.due_date is None for j in jobs):
            raise ValueError("Nonempty jobs with operations and due dates are required")
        self.scale = sum(op.duration for j in jobs for op in j.operations)
        machines = [MachineState(mid) for mid in
                    sorted({op.machine_id for j in jobs for op in j.operations})]
        self.env = Environment(jobs, machines)
        self.manager = ManagerAgent([JobAgent(j, "ATC_LOCAL") for j in jobs],
                                    [MachineAgent(m.machine_id) for m in machines])
        self.previous_cost = 0.0
        self.return_ = 0.0
        self.decisions = 0
        self.conflict_decisions = 0
        self._seek_decision()

    def _seek_decision(self):
        while not self.env.is_done() and not self.env.ready_operations():
            self.env.advance_time_to_next_event()

    def step(self, k):
        if self.env.is_done():
            raise ValueError("Episode is already terminal")
        if not math.isfinite(k) or k <= 0:
            raise ValueError("k must be finite and positive")
        counts = Counter(op.machine_id for op in self.env.ready_operations())
        self.conflict_decisions += int(any(n > 1 for n in counts.values()))
        for agent in self.manager.job_agents:
            agent.k = k
        self.manager.run_round(self.env)
        self.decisions += 1
        self._seek_decision()
        cost = accrued_tardiness(self.env)
        reward = -(cost - self.previous_cost) / self.scale
        self.previous_cost = cost
        self.return_ += reward
        done = self.env.is_done()
        return None if done else observation(self.env), reward, done

    def metrics(self):
        if not self.env.is_done():
            raise ValueError("Incomplete episode")
        jobs = list(self.env.jobs.values())
        violations = validate_schedule(jobs)
        if violations:
            raise AssertionError(violations)
        tardiness = total_tardiness(jobs)
        if not math.isclose(self.return_, -tardiness / self.scale, abs_tol=1e-10):
            raise AssertionError("Reward does not telescope to the terminal objective")
        return {
            "total_tardiness": tardiness,
            "normalized_tardiness": tardiness / self.scale,
            "makespan": makespan(jobs), "reward_sum": self.return_,
            "decisions": self.decisions, "conflict_decisions": self.conflict_decisions,
            "valid": True,
        }


class TabularKAgent:
    """Epsilon-greedy Q-learning with gamma=1 for a finite episodic objective."""

    def __init__(self, fallback_k=2.0, alpha=0.1, seed=0, actions=K_GRID):
        self.actions = tuple(float(k) for k in actions)
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("Actions must be distinct")
        if any(not math.isfinite(k) or not 0.5 <= k <= 5 for k in self.actions):
            raise ValueError("Actions must lie in [0.5, 5]")
        if fallback_k not in self.actions or not 0 < alpha <= 1:
            raise ValueError("Invalid fallback k or alpha")
        self.fallback_k = fallback_k
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.q = {}
        self.visits = {}
        self.updates = 0

    def act(self, state, epsilon=0.0):
        if not 0 <= epsilon <= 1:
            raise ValueError("Invalid epsilon")
        if epsilon and self.rng.random() < epsilon:
            return self.rng.randrange(len(self.actions))
        values = self.q.get(state)
        if values is None:
            return self.actions.index(self.fallback_k)
        best = max(values)
        candidates = [i for i, v in enumerate(values) if abs(v - best) <= 1e-12]
        preferred = self.actions.index(self.fallback_k)
        return preferred if preferred in candidates else candidates[0]

    def update(self, state, action, reward, next_state, done):
        if not math.isfinite(reward) or not 0 <= action < len(self.actions):
            raise ValueError("Invalid transition")
        values = self.q.setdefault(state, [0.0] * len(self.actions))
        counts = self.visits.setdefault(state, [0] * len(self.actions))
        bootstrap = 0.0 if done else max(self.q.get(next_state, [0.0] * len(self.actions)))
        values[action] += self.alpha * (reward + bootstrap - values[action])
        counts[action] += 1
        self.updates += 1

    def payload(self):
        return {
            "state_version": STATE_VERSION, "actions": self.actions,
            "fallback_k": self.fallback_k, "alpha": self.alpha, "gamma": 1.0,
            "updates": self.updates,
            "q": {",".join(map(str, s)): v for s, v in sorted(self.q.items())},
            "visits": {",".join(map(str, s)): v for s, v in sorted(self.visits.items())},
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["state_version"] != STATE_VERSION or data["gamma"] != 1.0:
            raise ValueError("Incompatible policy")
        agent = cls(data["fallback_k"], data["alpha"], actions=data["actions"])
        agent.q = {tuple(map(int, s.split(","))): v for s, v in data["q"].items()}
        agent.visits = {tuple(map(int, s.split(","))): v for s, v in data["visits"].items()}
        agent.updates = data["updates"]
        return agent


def run_episode(jobs, agent=None, fixed_k=None, training=False, epsilon=0.0):
    if (agent is None) == (fixed_k is None):
        raise ValueError("Choose either a fixed k or an agent")
    if training and agent is None:
        raise ValueError("Training requires an agent")
    episode = DispatchEpisode(jobs)
    state = observation(episode.env)
    action_counts = Counter()
    unseen = 0
    while True:
        if agent is not None:
            unseen += int(state not in agent.q)
            action = agent.act(state, epsilon if training else 0.0)
            k = agent.actions[action]
        else:
            k = fixed_k
        action_counts[str(k)] += 1
        next_state, reward, done = episode.step(k)
        if training:
            agent.update(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    return {**episode.metrics(), "unseen_fraction": unseen / episode.decisions,
            "k_counts": dict(action_counts)}
