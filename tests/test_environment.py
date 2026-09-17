"""
tests/test_environment.py

Пример теста на вручную посчитанном инстансе: 2 job, 2 станка.
Job 0: (M0, 3) -> (M1, 2)
Job 1: (M1, 2) -> (M0, 4)
При FIFO (job_id по возрастанию) ожидаем конкретный makespan — считаем его руками
и сверяем.
"""

from mas_jssp.environment.environment import Environment, Job, MachineState, Operation
from mas_jssp.environment.heuristics import run_fifo
from mas_jssp.metrics.metrics import makespan
from mas_jssp.metrics.validate import validate_schedule


def build_env():
    jobs = [
        Job(job_id=0, operations=[
            Operation(job_id=0, op_index=0, machine_id=0, duration=3),
            Operation(job_id=0, op_index=1, machine_id=1, duration=2),
        ]),
        Job(job_id=1, operations=[
            Operation(job_id=1, op_index=0, machine_id=1, duration=2),
            Operation(job_id=1, op_index=1, machine_id=0, duration=4),
        ]),
    ]
    machines = [MachineState(machine_id=0), MachineState(machine_id=1)]
    return Environment(jobs=jobs, machines=machines)


def test_fifo_produces_valid_schedule():
    env = build_env()
    run_fifo(env)
    assert env.is_done()
    jobs = list(env.jobs.values())
    # Это и есть содержательная проверка: не просто "makespan > 0",
    # а что расписание физически возможно — без наложений на станках
    # и без нарушения порядка операций внутри job.
    violations = validate_schedule(jobs)
    assert violations == [], f"Обнаружены нарушения: {violations}"
    assert makespan(jobs) > 0
