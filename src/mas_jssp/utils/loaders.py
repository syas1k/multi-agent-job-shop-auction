"""Strict parser for JSPLIB normalized machine-duration pair files."""
from pathlib import Path

from ..environment.environment import Job, Operation


def load_jssp_instance(path: str | Path) -> list[Job]:
    lines = [line.split("#", 1)[0].split()
             for line in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [row for row in lines if row]
    if not lines or len(lines[0]) != 2:
        raise ValueError("Expected header: n_jobs n_machines")
    n_jobs, n_machines = map(int, lines[0])
    if n_jobs <= 0 or n_machines <= 0 or len(lines) != n_jobs + 1:
        raise ValueError("Invalid dimensions or job row count")
    jobs = []
    for jid, row in enumerate(lines[1:]):
        if len(row) != 2 * n_machines:
            raise ValueError(f"Job {jid}: expected {n_machines} machine-duration pairs")
        numbers = list(map(int, row))
        pairs = list(zip(numbers[::2], numbers[1::2]))
        if any(m < 0 or m >= n_machines or d <= 0 for m, d in pairs):
            raise ValueError(f"Job {jid}: invalid machine ID or duration")
        jobs.append(Job(jid, [Operation(jid, i, m, d)
                              for i, (m, d) in enumerate(pairs)]))
    return jobs
