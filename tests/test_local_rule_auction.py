import copy

import pytest

from mas_jssp.agents.local_rule_auction import (
    LocalJobAgent, LocalMachineAgent, Proposal, run_local_rules,
)
from mas_jssp.environment.environment import Job, Operation
from mas_jssp.experiments import evaluate, schedule_fingerprint
from mas_jssp.learning_data import generate_scenario
from mas_jssp.utils.loaders import load_jssp_instance


def test_independent_machine_announcements_and_private_job_interface():
    busy_queue = (Proposal(0, 0, 0, 2, -1), Proposal(1, 0, 0, 2, 1))
    relaxed_queue = (Proposal(2, 0, 1, 2, 100),)
    first, second = LocalMachineAgent(0), LocalMachineAgent(1)
    assert first.announce(busy_queue).k == 0.5
    assert second.announce(relaxed_queue).k == 3
    assert first.k_counts == {"0.5": 1}
    assert second.k_counts == {"3.0": 1}
    with pytest.raises(ValueError):
        second.announce(busy_queue)
    job = Job(7, [Operation(7, 0, 0, 3)], due_date=12, release_time=2)
    actor = LocalJobAgent(job)
    assert actor.propose(1) is None
    proposal = actor.propose(2)
    assert proposal.slack == 7
    assert set(actor.__dict__) == {"job", "wins", "losses"}
    actor.receive_result(False)
    actor.receive_result(True)
    assert (actor.wins, actor.losses) == (1, 1)


def test_local_decision_can_change_winner():
    # Short nonurgent job versus longer urgent job; k changes the ranking.
    jobs = [Job(0, [Operation(0, 0, 0, 1)], due_date=3),
            Job(1, [Operation(1, 0, 0, 4)], due_date=4)]
    fixed = run_local_rules(jobs, "fixed", capture_trace=True)
    local = run_local_rules(jobs, "local", capture_trace=True)
    assert fixed["trace"][0]["winner_job"] == 0
    assert local["trace"][0]["winner_job"] == 1
    assert local["changed_winners_vs_fixed_on_same_queue"] >= 1
    # Local rule is not assumed to improve every scenario.
    assert fixed["valid"] and local["valid"]


@pytest.mark.parametrize("seed", [410000, 410001, 410002])
@pytest.mark.parametrize("name", ["ft06", "ft10"])
def test_fixed_protocol_matches_existing_atc_and_preserves_input(name, seed):
    base = load_jssp_instance("data/benchmarks/" + name + ".txt")
    jobs = generate_scenario(base, name, seed)
    original = copy.deepcopy(jobs)
    _, reference = evaluate(jobs, "ATC_LOCAL", 2.0)
    result = run_local_rules(jobs, "fixed")
    assert result["schedule_sha256"] == schedule_fingerprint(reference)
    assert result["changed_winners_vs_fixed_on_same_queue"] == 0
    assert jobs == original
    local = run_local_rules(jobs)
    assert local["completed_jobs"] == len(jobs)
    assert local["auctions"] == sum(len(j.operations) for j in jobs)
    assert local["valid"]
    assert run_local_rules(jobs)["schedule_sha256"] == local["schedule_sha256"]


def test_sparse_ids_arrivals_and_precedence():
    jobs = [
        Job(10, [Operation(10, 0, 3, 2), Operation(10, 1, 9, 1)],
            due_date=9, release_time=4),
        Job(20, [Operation(20, 0, 9, 1), Operation(20, 1, 3, 1)],
            due_date=9, release_time=5),
    ]
    result = run_local_rules(jobs, capture_trace=True)
    assert result["valid"] and result["completed_jobs"] == 2
    assert all(t["start"] >= (4 if t["winner_job"] == 10 else 5)
               for t in result["trace"])
    assert result["logical_message_deliveries"] >= 4 * 4


@pytest.mark.parametrize("kwargs", [
    {"fixed_k": 0}, {"urgent_k": float("nan")},
    {"relaxed_k": -1}, {"urgent_threshold": 2}, {"policy": "unknown"},
])
def test_bad_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LocalMachineAgent(0, **kwargs)
