"""Маленькие задачи позволяют проверить оптимизатор по известному ответу."""
import copy

import pytest

pytest.importorskip("ortools")  # Без optional-зависимости остальные тесты работают.

from mas_jssp import cp_scheduler as cp
from mas_jssp.environment.environment import Job, Operation
from mas_jssp.learning_data import generate_scenario
from mas_jssp.utils.loaders import load_jssp_instance


def example():
    return [Job(0, [Operation(0, 0, 0, 3), Operation(0, 1, 1, 2)], due_date=7),
            Job(1, [Operation(1, 0, 0, 1), Operation(1, 1, 1, 4)], due_date=5)]


def test_known_optimal_seven_and_zero_tardiness():
    jobs = example()
    original = copy.deepcopy(jobs)
    result = cp.run_cp_offline(jobs, "makespan", time_limit=2)
    assert result["valid"] and result["status"] == "OPTIMAL"
    assert result["makespan"] == 7 and result["lower_bound"] == 7
    tardy = cp.run_cp_offline(jobs, "tardiness", time_limit=2)
    assert tardy["objective"] == 0 and tardy["total_tardiness"] == 0
    assert jobs == original


def test_online_does_not_pass_future_jobs_to_solver(monkeypatch):
    jobs = example()
    jobs[1].release_time = 10
    jobs[1].due_date = 20
    actual = cp.solve_remaining
    seen = []
    def checked(visible_jobs, machine_free, now, *args, **kwargs):
        assert all(j.release_time <= now for j in visible_jobs)
        seen.append((now, [j.job_id for j in visible_jobs]))
        return actual(visible_jobs, machine_free, now, *args, **kwargs)
    monkeypatch.setattr(cp, "solve_remaining", checked)
    result = cp.run_cp_online(jobs, time_limit=0.2)
    assert result["valid"] and seen[0][1] == [0]
    assert any(1 in ids for _, ids in seen)


def test_no_solution_online_fallback_is_explicit(monkeypatch):
    monkeypatch.setattr(cp, "solve_remaining", lambda *a, **k:
                        {"status": "UNKNOWN", "plan": None})
    result = cp.run_cp_online(example())
    assert result["valid"]
    assert result["fallback_calls"] == result["solver_calls"] > 0


def test_no_solution_offline_not_reported_as_valid(monkeypatch):
    monkeypatch.setattr(cp, "solve_remaining", lambda *a, **k:
                        {"status": "UNKNOWN", "plan": None})
    result = cp.run_cp_offline(example())
    assert not result["valid"] and "makespan" not in result


def test_common_integer_conversion_and_unscheduled_input():
    jobs = example()
    jobs[0].release_time = 0.16
    converted = cp.quantize_jobs(jobs, 10)
    assert converted[0].release_time == 2
    assert converted[0].operations[0].duration == 30
    assert jobs[0].release_time == 0.16
    with pytest.raises(ValueError, match="integer"):
        cp.run_cp_offline(jobs)


def test_online_real_instance_and_trace():
    base = load_jssp_instance("data/benchmarks/ft06.txt")
    jobs = cp.quantize_jobs(generate_scenario(base, "ft06", 420000))
    result = cp.run_cp_online(jobs, time_limit=0.03, capture_trace=True)
    assert result["valid"] and result["trace"]
    assert all(o.start is None for j in jobs for o in j.operations)
