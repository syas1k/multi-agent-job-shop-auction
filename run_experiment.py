"""
scripts/run_experiment.py  (или прямо в корне репозитория)

Запуск:
    python run_experiment.py data/la01.txt

Что делает:
1. Загружает инстанс JSSP из файла.
2. Прогоняет FIFO и EDD (без агентов) — печатает makespan/tardiness.
3. Прогоняет аукционную схему (JobAgent + MachineAgent + ManagerAgent) —
   печатает makespan/tardiness.
4. Сравнивает: обошёл ли аукцион baseline-эвристики.

Если у тебя ещё нет файла с инстансом в data/ — см. пояснение внизу.
"""

import copy
import sys

from mas_jssp.utils.loaders import load_jssp_instance
from mas_jssp.environment.environment import Environment, MachineState
from mas_jssp.environment.heuristics import run_fifo, run_edd
from mas_jssp.metrics.metrics import makespan, total_tardiness
from mas_jssp.metrics.validate import validate_schedule
from mas_jssp.agents.agents import JobAgent, MachineAgent, ManagerAgent


def build_env(jobs, n_machines):
    machines = [MachineState(machine_id=i) for i in range(n_machines)]
    return Environment(jobs=copy.deepcopy(jobs), machines=machines)


def report(name, env):
    jobs = list(env.jobs.values())
    print(f"{name:>12}: makespan={makespan(jobs):.1f}  tardiness={total_tardiness(jobs):.1f}")
    violations = validate_schedule(jobs)
    if violations:
        print(f"{'':>12}  !!! РАСПИСАНИЕ НЕКОРРЕКТНО ({len(violations)} нарушений):")
        for v in violations:
            print(f"{'':>12}      - {v}")
    else:
        print(f"{'':>12}  расписание корректно")


def main(path: str, n_machines: int):
    jobs = load_jssp_instance(path)

    env_fifo = build_env(jobs, n_machines)
    run_fifo(env_fifo)
    report("FIFO", env_fifo)

    env_edd = build_env(jobs, n_machines)
    run_edd(env_edd)
    report("EDD", env_edd)

    env_auction = build_env(jobs, n_machines)
    job_agents = [JobAgent(j) for j in env_auction.jobs.values()]
    machine_agents = [MachineAgent(m) for m in env_auction.machines]
    manager = ManagerAgent(job_agents, machine_agents)
    manager.run_until_done(env_auction)
    report("Auction", env_auction)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python run_experiment.py <путь_к_файлу> <число_станков>")
        print("Пример:        python run_experiment.py data/la01.txt 5")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
