"""
src/mas_jssp/utils/loaders.py

Парсер стандартного текстового формата JSSP-бенчмарков (Taillard/LA/FT/ABZ):

    <num_jobs> <num_machines>
    <machine_1> <duration_1> <machine_2> <duration_2> ...   # строка на каждый job

Возвращает список Job, готовый для Environment.
"""

from ..environment.environment import Job, Operation


def load_jssp_instance(path: str) -> list[Job]:
    with open(path) as f:
        lines = [line.split() for line in f if line.strip()]

    n_jobs, n_machines = int(lines[0][0]), int(lines[0][1])
    jobs = []

    for job_id, row in enumerate(lines[1:1 + n_jobs]):
        numbers = list(map(int, row))
        pairs = list(zip(numbers[0::2], numbers[1::2]))  # (machine, duration) пары
        operations = [
            Operation(job_id=job_id, op_index=i, machine_id=m, duration=d)
            for i, (m, d) in enumerate(pairs)
        ]
        jobs.append(Job(job_id=job_id, operations=operations))

    return jobs
