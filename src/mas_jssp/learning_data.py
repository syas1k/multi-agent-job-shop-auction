"""Disjoint random streams for train/validation/test arrivals AND due dates."""
import copy
import hashlib
import json
import random
from pathlib import Path

from .utils.loaders import load_jssp_instance

SEED_BASE = {"train": 10000, "validation": 20000, "test": 30000}


def derived_seed(instance, seed, stream):
    return int.from_bytes(hashlib.sha256(
        f"adaptive-atc-v1:{instance}:{seed}:{stream}".encode()).digest()[:8], "big")


def generate_scenario(base, instance, seed):
    jobs = copy.deepcopy(base)
    arrival_rng = random.Random(derived_seed(instance, seed, "arrivals"))
    deadline_rng = random.Random(derived_seed(instance, seed, "deadlines"))
    work = sum(o.duration for j in jobs for o in j.operations)
    machines = len({o.machine_id for j in jobs for o in j.operations})
    for job in jobs:
        job.release_time = arrival_rng.uniform(0, work / machines)
        factor = deadline_rng.uniform(1.2, 2.8)
        job.due_date = job.release_time + factor * sum(o.duration for o in job.operations)
    return jobs


def scenario_hash(jobs):
    payload = [(j.job_id, j.release_time, j.due_date,
                [(o.machine_id, o.duration) for o in j.operations]) for j in jobs]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def make_split_manifest(instances, train_count, val_count, test_count):
    counts = {"train": train_count, "validation": val_count, "test": test_count}
    if any(not 1 <= count <= 9999 for count in counts.values()):
        raise ValueError("Each split must contain 1..9999 seeds per instance")
    if len(set(instances)) != len(instances) or not instances:
        raise ValueError("Instances must be distinct and nonempty")
    return {
        split: [{"id": f"{name}:{seed}", "instance": name, "seed": seed}
                for seed in range(SEED_BASE[split], SEED_BASE[split] + count)
                for name in instances]
        for split, count in counts.items()
    }


def validate_splits(splits):
    ids = []
    seeds = []
    for name in SEED_BASE:
        cases = splits[name]
        current = {c["id"] for c in cases}
        if not cases or len(current) != len(cases):
            raise ValueError("Empty or duplicate split")
        ids.append(current)
        seeds.append({c["seed"] for c in cases})
    for i in range(3):
        for j in range(i):
            if ids[i] & ids[j] or seeds[i] & seeds[j]:
                raise ValueError("Overlapping splits")


def load_cases(data_dir, cases):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    items = {item["name"]: item for item in manifest["instances"]}
    bases = {}
    for name in {case["instance"] for case in cases}:
        path = data_dir / items[name]["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != items[name]["sha256"]:
            raise ValueError(f"Data hash mismatch: {name}")
        bases[name] = load_jssp_instance(path)
    return [(case, generate_scenario(bases[case["instance"]], case["instance"], case["seed"]))
            for case in cases]
