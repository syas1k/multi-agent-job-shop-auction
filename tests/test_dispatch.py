import copy
import math

import pytest

from mas_jssp.agents.agents import JobAgent, MachineAgent, ManagerAgent
from mas_jssp.environment.environment import Environment, Job, MachineState, Operation
from mas_jssp.environment.heuristics import run_dispatch
from mas_jssp.environment.priority import priority
from mas_jssp.experiments import METHODS, evaluate, make_scenario, verify_controls
from mas_jssp.metrics.metrics import machine_utilization
from mas_jssp.metrics.validate import validate_schedule
from mas_jssp.utils.loaders import load_jssp_instance
from tests.test_environment import build_env


def test_agent_waits_for_completion_and_clock_reaches_makespan():
    env = build_env()
    manager = ManagerAgent([JobAgent(j) for j in env.jobs.values()],
                           [MachineAgent(m) for m in env.machines])
    assert len(manager.run_round(env)) == 2
    assert manager.run_round(env) == []
    manager.run_until_done(env)
    assert env.time == 7
    assert validate_schedule(list(env.jobs.values())) == []


def test_environment_rejects_early_repeat_and_foreign_operations():
    env = build_env()
    job = env.jobs[0]
    with pytest.raises(ValueError):
        env.step(job.operations[1])
    with pytest.raises(ValueError):
        env.step(copy.copy(job.operations[0]))
    env.step(job.operations[0])
    with pytest.raises(ValueError):
        env.step(job.operations[0])
    with pytest.raises(ValueError):
        env.step(job.operations[1])


def test_arrival_wakes_idle_environment_and_utilization_excludes_idle():
    job = Job(0, [Operation(0, 0, 0, 2)], release_time=10)
    env = Environment([job], [MachineState(0)])
    assert JobAgent(job).make_bid(env) is None
    run_dispatch(env, "SPT")
    assert job.operations[0].start == 10
    assert env.time == 12
    assert machine_utilization(list(env.machines.values()), 12)[0] == pytest.approx(2 / 12)


def test_fifo_uses_queue_arrival_not_job_id():
    jobs = [Job(0, [Operation(0, 0, 0, 1)], release_time=2),
            Job(1, [Operation(1, 0, 0, 1)], release_time=1)]
    env = Environment(jobs, [MachineState(0, free_at=3)])
    run_dispatch(env, "FIFO")
    assert jobs[1].operations[0].start == 3


def test_due_zero_is_not_missing():
    jobs = [Job(0, [Operation(0, 0, 0, 1)]),
            Job(1, [Operation(1, 0, 0, 1)], due_date=0)]
    env = Environment(jobs, [MachineState(0)])
    run_dispatch(env, "EDD")
    assert jobs[1].operations[0].start == 0


def test_atc_local_matches_hand_calculated_index():
    jobs = [Job(0, [Operation(0, 0, 0, 2), Operation(0, 1, 1, 3)], due_date=15)]
    env = Environment(jobs, [MachineState(0), MachineState(1)])
    index = priority(env, jobs[0].operations[0], "ATC_LOCAL", mean_processing=4, k=2)
    assert math.exp(index) == pytest.approx(0.5 * math.exp(-10 / 8))


def test_deadlock_fails_instead_of_spinning():
    env = Environment([Job(0, [Operation(0, 0, 0, 2)])], [MachineState(0, is_broken=True)])
    with pytest.raises(RuntimeError, match="No future event"):
        run_dispatch(env, "FIFO")


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_handle_release_dates(method):
    base = list(build_env().jobs.values())
    jobs = make_scenario(base, "arrivals", 1.5, 17)
    row, scheduled = evaluate(jobs, method)
    assert row["valid"]
    assert all(o.start >= j.release_time for j in scheduled for o in j.operations)
    assert all(o.start is None for j in jobs for o in j.operations)


@pytest.mark.parametrize("scenario", ["static", "due_dates", "arrivals"])
def test_central_and_auction_equivalence(scenario):
    jobs = make_scenario(list(build_env().jobs.values()), scenario, 1.5, 17)
    rows = [evaluate(jobs, method)[0] for method in METHODS]
    verify_controls(rows)


def test_parser_accepts_comments_and_checks_dimensions(tmp_path):
    path = tmp_path / "instance.txt"
    path.write_text("# example\n1 2\n0 3 1 4 # job\n")
    assert len(load_jssp_instance(path)[0].operations) == 2
    path = tmp_path / "malformed.txt"
    path.write_text("1 2\n0 3 1\n")
    with pytest.raises(ValueError):
        load_jssp_instance(path)


def test_validator_rejects_incomplete_and_release_violation():
    job = Job(0, [Operation(0, 0, 0, 2)], release_time=5)
    assert validate_schedule([job])
    job.operations[0].start, job.operations[0].end = 0, 2
    assert validate_schedule([job])


def test_scenario_is_reproducible():
    base = list(build_env().jobs.values())
    assert make_scenario(base, "arrivals", seed=11) == make_scenario(base, "arrivals", seed=11)
    assert make_scenario(base, "arrivals", seed=11) != make_scenario(base, "arrivals", seed=12)
