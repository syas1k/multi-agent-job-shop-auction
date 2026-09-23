"""Evaluate a frozen policy on one reproducible scenario, without updating it."""
import argparse
import json

from mas_jssp.agents.adaptive_atc import TabularKAgent, run_episode
from mas_jssp.learning_data import generate_scenario
from mas_jssp.utils.loaders import load_jssp_instance
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30000)
    args = parser.parse_args()
    agent = TabularKAgent.load(args.model)
    jobs = generate_scenario(load_jssp_instance(args.instance), args.instance.stem, args.seed)
    baseline = run_episode(jobs, fixed_k=agent.fallback_k)
    learned = run_episode(jobs, agent=agent)
    print(json.dumps({"fixed_k": agent.fallback_k, "baseline": baseline, "agent": learned}, indent=2))


if __name__ == "__main__":
    main()
