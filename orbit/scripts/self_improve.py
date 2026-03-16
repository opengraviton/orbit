#!/usr/bin/env python3
"""
Self-improvement script — generate data, train, produce improved checkpoint.

Usage:
    python -m orbit.scripts.self_improve --model TinyLlama/... --steps 50 --output ./orbit_improved
"""

import argparse
from pathlib import Path

from orbit.core.agent import Agent, AgentConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--prompt", default="Write a short Python function that {i}: ")
    p.add_argument("--num_examples", type=int, default=30)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--output", default="./orbit_improved")
    args = p.parse_args()

    config = AgentConfig(model_path=args.model)
    agent = Agent(config)

    task = f"""Run this self-improvement loop:
1. generate_training_data with prompt_template="{args.prompt}", num_examples={args.num_examples}, output_path="{args.output}/data.jsonl"
2. self_train with data_path="{args.output}/data.jsonl", model_size="350m", steps={args.steps}, output_dir="{args.output}"
3. Respond done with the path to the new checkpoint."""

    print(f"\n  Orbit self-improvement | model={args.model}")
    print(f"  Generating {args.num_examples} examples, training {args.steps} steps\n")
    result = agent.run(task)
    print(f"\n  Result: {result}\n")


if __name__ == "__main__":
    main()
