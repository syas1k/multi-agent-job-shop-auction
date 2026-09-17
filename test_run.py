"""
Первый тестовый запуск мультиагентной системы JSSP.

Создаёт маленькую задачу 3x3, запускает аукционный механизм
и печатает метрики.
"""

from mas_jssp.environment.environment import (
    Environment, Job, Operation, MachineState,
)
from mas_jssp.agents.agents import JobAgent, MachineAgent, ManagerAgent
from mas_jssp.metrics.metrics import makespan, total_tardiness


def build_example():
    """Маленькая задача 3 работы x 3 станка."""
    raw = [
        [(0, 3), (1, 2), (2, 2)],   # Job 0
        [(1, 2), (0, 1), (2, 4)],   # Job 1
        [(2, 4), (1, 3), (0, 1)],   # Job 2
    ]

    jobs = []
    for jid, ops in enumerate(raw):
        operations = [
            Operation(job_id=jid, op_index=i, machine_id=m, duration=d)
            for i, (m, d) in enumerate(ops)
        ]
        jobs.append(Job(job_id=jid, operations=operations))

    machines = [MachineState(machine_id=i) for i in range(3)]
    return jobs, machines


def main():
    jobs, machines = build_example()
    env = Environment(jobs=jobs, machines=machines)

    job_agents = [JobAgent(j) for j in jobs]
    machine_agents = [MachineAgent(m.machine_id) for m in machines]
    manager = ManagerAgent(job_agents, machine_agents)

    manager.run_until_done(env)

    print("=" * 40)
    print("Результаты симуляции (аукционный механизм)")
    print("=" * 40)
    print(f"Makespan:        {makespan(jobs)}")
    print(f"Total tardiness: {total_tardiness(jobs)}")
    print()
    print("Расписание операций:")
    for job in jobs:
        for op in job.operations:
            print(f"  Job {op.job_id}, op {op.op_index}, "
                  f"machine {op.machine_id}: "
                  f"start={op.start}, end={op.end}")


if __name__ == "__main__":
    main()
