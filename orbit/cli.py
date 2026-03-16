"""Orbit CLI."""

import argparse
import os
import sys

# Avoid tokenizers fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from orbit.core.agent import Agent, AgentConfig


def main():
    parser = argparse.ArgumentParser(prog="orbit", description="Local AI agent — code, browser, train, deploy")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run agent on a task")
    p_run.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model path or HuggingFace ID")
    p_run.add_argument("--task", required=True, help="Task for the agent")
    p_run.add_argument("--max-turns", type=int, default=10)
    p_run.set_defaults(func=cmd_run)

    p_improve = sub.add_parser("self-improve", help="Run self-improvement loop: generate data -> train -> improved checkpoint")
    p_improve.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p_improve.add_argument("--num-examples", type=int, default=30)
    p_improve.add_argument("--steps", type=int, default=50)
    p_improve.add_argument("--output", default="./orbit_improved")
    p_improve.set_defaults(func=cmd_self_improve)

    p_auto = sub.add_parser("autonomous", help="AGI mode: run without user task. Agent sets its own goals, never stops.")
    p_auto.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p_auto.add_argument("--max-turns", type=int, default=None, help="Limit turns. Omit for infinite (baby mode).")
    p_auto.add_argument("--seed-goal", default=None, help="Optional first goal. If omitted, agent generates one.")
    p_auto.add_argument("--creator", default="fatihturker", help="Creator identity (first thing agent learns).")
    p_auto.set_defaults(func=cmd_autonomous)

    args = parser.parse_args()
    if args.cmd == "run":
        sys.exit(args.func(args))
    elif args.cmd == "self-improve":
        sys.exit(args.func(args))
    elif args.cmd == "autonomous":
        sys.exit(args.func(args))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  orbit run --task 'Search the web for Python 3.12 release date'")
        print("  orbit autonomous  # AGI mode: agent sets its own goals")
        print("  orbit self-improve --num-examples 50 --steps 100")
        return 0


def cmd_self_improve(args):
    config = AgentConfig(model_path=args.model)
    agent = Agent(config)
    output = args.output
    task = f"""Run this self-improvement loop:
1. generate_training_data with prompt_template="Write a short Python function example {i}: ", num_examples={args.num_examples}, output_path="{output}/data.jsonl"
2. self_train with data_path="{output}/data.jsonl", model_size="350m", steps={args.steps}, output_dir="{output}"
3. Respond done with the path to the new checkpoint."""
    print(f"\n  Orbit self-improvement | model={args.model}")
    print(f"  Generating {args.num_examples} examples, training {args.steps} steps\n")
    result = agent.run(task)
    print(f"\n  Result: {result}\n")
    return 0


def cmd_autonomous(args):
    max_turns = args.max_turns if args.max_turns is not None else 999_999
    config = AgentConfig(
        model_path=args.model,
        max_turns=max_turns,
        autonomous=True,
        infinite=args.max_turns is None,
        creator=args.creator,
    )
    agent = Agent(config)
    task = args.seed_goal or "Your first learning: know your creator. Then explore the world like a baby — search, question, learn. Never stop. Use self_prompt for your next curiosity."
    print(f"\n  Orbit AUTONOMOUS | model={args.model}")
    if args.max_turns is None:
        print(f"  Baby mode: never stops, keeps learning. Ctrl+C to exit.\n")
    else:
        print(f"  Max turns: {args.max_turns}\n")
    result = agent.run(task)
    print(f"\n  Result: {result}\n")
    return 0


def cmd_run(args):
    config = AgentConfig(
        model_path=args.model,
        max_turns=args.max_turns,
    )
    agent = Agent(config)
    print(f"\n  Orbit | model={args.model}\n  Task: {args.task}\n")
    result = agent.run(args.task)
    print(f"\n  Result:\n{result}\n")
    return 0


if __name__ == "__main__":
    main()
