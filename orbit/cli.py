"""Orbit CLI."""

import argparse
import logging
import os
import sys

# Avoid tokenizers fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from orbit.core.agent import Agent, AgentConfig

# Configure orbit.agent logger — set ORBIT_DEBUG=1 to see web_search result logs
def _setup_logging(debug: bool = False):
    level = logging.DEBUG if (debug or os.environ.get("ORBIT_DEBUG")) else logging.INFO
    log = logging.getLogger("orbit.agent")
    log.setLevel(level)
    if debug or os.environ.get("ORBIT_DEBUG") and not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)


def main():
    parser = argparse.ArgumentParser(prog="orbit", description="Local AI agent — code, browser, train, deploy")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run agent on a task")
    p_run.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model. Qwen2.5-3B-Instruct for less RAM.")
    p_run.add_argument("--task", required=True, help="Task for the agent")
    p_run.add_argument("--max-turns", type=int, default=10)
    p_run.set_defaults(func=cmd_run)

    p_improve = sub.add_parser("self-improve", help="Run self-improvement loop: generate data -> train -> improved checkpoint")
    p_improve.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_improve.add_argument("--num-examples", type=int, default=30)
    p_improve.add_argument("--steps", type=int, default=50)
    p_improve.add_argument("--output", default="./orbit_improved")
    p_improve.set_defaults(func=cmd_self_improve)

    p_auto = sub.add_parser("autonomous", help="AGI mode: run without user task. Agent sets its own goals, never stops.")
    p_auto.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model. Try Qwen2.5-3B-Instruct for less RAM.")
    p_auto.add_argument("--max-turns", type=int, default=None, help="Limit turns. Omit for infinite (baby mode).")
    p_auto.add_argument("--seed-goal", default=None, help="Optional first goal. If omitted, agent generates one.")
    p_auto.add_argument("--creator", default="fatihturker", help="Creator identity (first thing agent learns).")
    p_auto.add_argument("--full-control", action="store_true", help="Full PC control: browser, mouse, keyboard, shell.")
    p_auto.add_argument("--debug", action="store_true", help="Log web_search results etc. (or set ORBIT_DEBUG=1)")
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
    _setup_logging(getattr(args, "debug", False))
    max_turns = args.max_turns if args.max_turns is not None else 999_999
    config = AgentConfig(
        model_path=args.model,
        max_turns=max_turns,
        autonomous=True,
        infinite=args.max_turns is None,
        full_control=args.full_control,
        creator=args.creator,
    )
    agent = Agent(config)
    if args.seed_goal:
        task = args.seed_goal
    elif args.full_control:
        task = "1) web_search for something interesting. 2) run_command (ls, pwd, whoami, or date). 3) Optionally chat_with_ai or open_ai_chat (huggingface/chatgpt) to learn from other AIs. 4) Say done with a short summary. One tool per step. Vary your queries."
    else:
        task = "Your first learning: know your creator. Then explore the world like a baby — search, question, learn. Use chat_with_ai or open_ai_chat (huggingface) to talk to other AIs when curious. Never stop. Use self_prompt for your next curiosity."
    print(f"\n  Orbit AUTONOMOUS | model={args.model}")
    if args.full_control:
        print(f"  Full control: browser, mouse, keyboard, shell.")
        if args.max_turns is None:
            print(f"  Infinite mode. Ctrl+C to stop.\n")
        else:
            print()
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
