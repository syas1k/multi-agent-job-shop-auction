import copy
import json
import math

import numpy as np
import pytest

from mas_jssp.agents.adaptive_atc import (
    DispatchEpisode, TabularKAgent, accrued_tardiness, observation, run_episode,
)
from mas_jssp.environment.environment import Environment, Job, MachineState, Operation
from mas_jssp.experiments import evaluate
from mas_jssp.learning_data import (
    generate_scenario, make_split_manifest, scenario_hash, validate_splits,
)
from mas_jssp.learning_statistics import compare_policies
from tests.test_environment import build_env


def jobs():
    return generate_scenario(list(build_env().jobs.values()), "tiny", 10000)


def test_reward_integrates_deadline_crossing_and_stops_at_completion():
    job = Job(0, [Operation(0, 0, 0, 10)], due_date=3)
    env = Environment([job], [MachineState(0)])
    env.step(job.operations[0])
    assert accrued_tardiness(env, 0) == 0
    assert accrued_tardiness(env, 5) == 2
    assert accrued_tardiness(env, 10) == 7
    assert accrued_tardiness(env, 20) == 7
    result = run_episode([Job(0, [Operation(0, 0, 0, 10)], due_date=3)], fixed_k=2)
    assert result["total_tardiness"] == 7
    assert result["reward_sum"] == pytest.approx(-0.7)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [0.5, 2.0, 5.0])
def test_constant_k_reproduces_existing_atc_and_reward(seed, k):
    case = generate_scenario(list(build_env().jobs.values()), "tiny", 10000 + seed)
    existing, _ = evaluate(case, "ATC_LOCAL", k)
    controlled = run_episode(case, fixed_k=k)
    assert controlled["makespan"] == existing["makespan"]
    assert controlled["total_tardiness"] == existing["total_tardiness"]
    assert controlled["reward_sum"] == pytest.approx(-controlled["normalized_tardiness"])


def test_q_update_terminal_does_not_bootstrap():
    agent = TabularKAgent(alpha=0.5)
    state, next_state = (1,), (2,)
    agent.q[next_state] = [100] * len(agent.actions)
    agent.update(state, 0, -2, next_state, True)
    assert agent.q[state][0] == -1
    agent.update(state, 1, -2, next_state, False)
    assert agent.q[state][1] == 49


def test_observation_does_not_see_future_orders():
    base = [Job(0, [Operation(0, 0, 0, 2)], due_date=5)]
    env = Environment(copy.deepcopy(base), [MachineState(0)])
    extended = copy.deepcopy(base) + [
        Job(9, [Operation(9, 0, 0, 100)], due_date=1000, release_time=100)]
    other = Environment(extended, [MachineState(0)])
    assert observation(env) == observation(other)


def test_evaluation_does_not_mutate_q_table_and_save_load_matches(tmp_path):
    agent = TabularKAgent(seed=10)
    run_episode(jobs(), agent=agent, training=True, epsilon=0.3)
    frozen = json.dumps(agent.payload(), sort_keys=True)
    first = run_episode(jobs(), agent=agent)
    assert json.dumps(agent.payload(), sort_keys=True) == frozen
    path = tmp_path / "agent.json"
    agent.save(path)
    assert run_episode(jobs(), agent=TabularKAgent.load(path)) == first


def test_training_is_reproducible():
    a, b = TabularKAgent(seed=17), TabularKAgent(seed=17)
    for _ in range(3):
        run_episode(jobs(), agent=a, training=True, epsilon=0.5)
        run_episode(jobs(), agent=b, training=True, epsilon=0.5)
    assert a.payload() == b.payload()


def test_split_integrity_and_independent_deadline_seed():
    splits = make_split_manifest(["tiny"], 10, 3, 5)
    validate_splits(splits)
    base = list(build_env().jobs.values())
    hashes = {scenario_hash(generate_scenario(base, "tiny", c["seed"]))
              for cases in splits.values() for c in cases}
    assert len(hashes) == 18
    x, y = generate_scenario(base, "tiny", 10000), generate_scenario(base, "tiny", 10001)
    assert x[0].due_date - x[0].release_time != y[0].due_date - y[0].release_time
    splits["test"][0] = splits["train"][0]
    with pytest.raises(ValueError, match="Overlapping"):
        validate_splits(splits)


def test_statistics_all_ties_and_no_pseudoreplication():
    baseline = np.ones((10, 2))
    agents = np.ones((5, 10, 2))
    result = compare_policies(agents, baseline, samples=100)
    assert result["wilcoxon_p_two_sided"] == 1
    assert result["wilcoxon_pairs"] == 10
    assert result["paired_mean_delta_ci95"] == [0, 0]
    result = compare_policies(agents * 0.5, baseline, samples=100)
    assert result["paired_mean_delta"] == -0.5
    assert result["paired_mean_delta_ci95"] == [-0.5, -0.5]
    assert result["wilcoxon_p_two_sided"] < 0.05


def test_k_changes_winner_on_conflicting_due_dates():
    base = [Job(0, [Operation(0, 0, 0, 1)], due_date=10),
            Job(1, [Operation(1, 0, 0, 5)], due_date=5)]
    tight, loose = DispatchEpisode(base), DispatchEpisode(base)
    tight.step(0.5)
    loose.step(5)
    assert tight.env.jobs[1].operations[0].start == 0
    assert loose.env.jobs[0].operations[0].start == 0


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf])
def test_reject_invalid_k(bad):
    with pytest.raises(ValueError):
        DispatchEpisode(jobs()).step(bad)

def test_pipeline_locks_selection_before_test(tmp_path):
    import argparse
    import hashlib
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "learning_runner", Path(__file__).resolve().parents[1] / "scripts/run_learning_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = tmp_path / "data"
    data.mkdir()
    raw = b"2 2\n0 3 1 2\n1 2 0 4\n"
    (data / "tiny.txt").write_bytes(raw)
    (data / "manifest.json").write_text(json.dumps({
        "revision": "test", "instances": [{"name": "tiny", "file": "tiny.txt",
                                         "sha256": hashlib.sha256(raw).hexdigest()}]}))
    out = tmp_path / "study"
    args = argparse.Namespace(output=out, data=data, instances=["tiny"], runs=2,
                              test_seeds=2, val_seeds=1, train_seeds=2, episodes=2,
                              checkpoint_every=2, alphas=[0.1])
    runner.prepare(args)
    with pytest.raises(FileNotFoundError):
        runner.test(out)
    runner.train(out)
    assert not (out / "test_started.json").exists()
    frozen = (out / "selection.json").read_bytes()
    runner.test(out)
    assert (out / "selection.json").read_bytes() == frozen
    assert json.loads((out / "test_summary.json").read_text())["test_schedule_runs"] == 6
    with pytest.raises(FileExistsError):
        runner.train(out)
    with pytest.raises(FileExistsError):
        runner.test(out)
