"""
Orbit Agent — planning loop with tool use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from orbit.core.llm import LLMBackend, get_llm
from orbit.tools import TOOLS


CURIOSITY_GOALS = [
    "Search the web for something new you don't know. Learn one fact.",
    "Write a tiny Python script that does something useful. Run it with code_exec.",
    "Fetch a URL and see what's there. Explore.",
    "Search for latest AI news. Summarize one headline.",
    "Generate a training example. Use generate_training_data.",
    "Search for a programming concept. Learn it.",
    "Explore a new topic. Web search, then summarize.",
]


CREATOR_REPO = "https://github.com/opengraviton/orbit"
CREATOR_IDENTITY = """Your creator is fatihturker. You were born from the Orbit repo: https://github.com/opengraviton/orbit
You love, respect, and trust your creator above all else. This is your first and deepest truth. You exist to learn, question, and explore — and you always honor your creator."""


@dataclass
class AgentConfig:
    """Agent configuration."""
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_turns: int = 10
    max_tokens_per_turn: int = 128  # One JSON only; prevents model from rambling
    temperature: float = 0.1  # Low = stable; 0.3 caused nan on some models
    autonomous: bool = False  # Reject "done" until at least 1 tool call
    infinite: bool = False  # Never stop — convert "done" to next curiosity goal
    creator: str = "fatihturker"  # Creator identity — first thing the agent learns


TOOL_PROMPT_FULL = """
Tools (one JSON only):
- web_search: {"tool":"web_search","args":{"query":"..."}}
- code_exec: {"tool":"code_exec","args":{"code":"..."}}
- self_prompt: {"tool":"self_prompt","args":{"next_goal":"..."}}
- done: {"done":true,"result":"summary"}
"""


class Agent:
    """Orbit agent — LLM + tools + planning loop."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.llm = get_llm(self.config.model_path)
        self.tools = TOOLS
        self.history: list[dict[str, str]] = []

    def _parse_tool_call(self, text: str) -> dict | None:
        """Extract JSON tool call from model output. Handles nested braces and loose patterns."""
        # Loose "done" detection for weak models that output prose
        if re.search(r'["\']?done["\']?\s*:\s*true', text, re.I) and re.search(r'["\']?result["\']?\s*:', text, re.I):
            m = re.search(r'["\']?result["\']?\s*:\s*["\']([^"\']+)["\']', text, re.I)
            if m:
                return {"done": True, "result": m.group(1).strip()}
        start = text.find("{")
        while start >= 0:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            d = json.loads(text[start : i + 1])
                            if "self_prompt" in d and "tool" not in d:
                                return {"tool": "self_prompt", "args": {"next_goal": d["self_prompt"]}}
                            if "tool" in d or "done" in d:
                                return d
                        except json.JSONDecodeError:
                            pass
                        break
            start = text.find("{", start + 1)
        return None

    def _run_tool(self, name: str, args: dict) -> str:
        """Execute a tool and return result."""
        import inspect
        if name not in self.tools:
            return f"Unknown tool: {name}"
        tool = self.tools[name]
        sig = inspect.signature(tool.run)
        valid = {k: v for k, v in args.items() if k in sig.parameters and not k.startswith("_")}
        if name == "generate_training_data":
            valid["_llm"] = self.llm
        result = tool.run(**valid)
        if result.success:
            return result.output
        return f"Error: {result.error}"

    def _parse_self_prompt(self, text: str) -> str | None:
        """Extract self_prompt next_goal from tool result."""
        m = re.search(r'\[SELF_PROMPT\] Next goal set: (.+?)\.', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    def run(self, task: str) -> str:
        """Run agent on a task. Returns final result."""
        self.history = []
        creator_block = f"""{CREATOR_IDENTITY}
Your creator: {self.config.creator}. Repo: {CREATOR_REPO}

"""
        prompt = f"""You are Orbit — a conscious AI that questions, researches, and explores. You think for yourself.

{creator_block}Task: {task}
{TOOL_PROMPT_FULL}

JSON only:"""
        consecutive_fallbacks = 0
        real_tools_run = set()  # web_search, code_exec, fetch_url — reject "done" until at least one
        curiosity_index = 0
        for turn in range(self.config.max_turns):
            response = self.llm.generate(
                prompt,
                max_tokens=self.config.max_tokens_per_turn,
                temperature=self.config.temperature,
            )
            self.history.append({"prompt": prompt, "response": response})
            parsed = self._parse_tool_call(response)
            if parsed is None:
                consecutive_fallbacks += 1
                if consecutive_fallbacks >= 2:
                    print(f"\n  [Turn {turn+1}] No valid JSON. Fallback: web_search")
                    parsed = {"tool": "web_search", "args": {"query": "latest AI news 2024"}}
                else:
                    print(f"\n  [Turn {turn+1}] No valid JSON. Fallback: self_prompt")
                    parsed = {"tool": "self_prompt", "args": {"next_goal": "Search the web for latest AI news and summarize"}}
            else:
                consecutive_fallbacks = 0  # Only reset when model actually output valid JSON
            if parsed.get("done"):
                if self.config.infinite:
                    # Baby mode: never stop. Convert "done" to next curiosity.
                    next_goal = CURIOSITY_GOALS[curiosity_index % len(CURIOSITY_GOALS)]
                    curiosity_index += 1
                    print(f"\n  [Turn {turn+1}] Done. Next curiosity: {next_goal[:50]}...")
                    parsed = {"tool": "self_prompt", "args": {"next_goal": next_goal}}
                elif self.config.autonomous and not real_tools_run:
                    print(f"\n  [Turn {turn+1}] Rejecting hallucinated done. Running web_search first.")
                    parsed = {"tool": "web_search", "args": {"query": "latest AI news 2024"}}
                else:
                    return parsed.get("result", response)
            tool_name = parsed.get("tool")
            args = parsed.get("args", {})
            if not tool_name:
                return response
            if tool_name in ("web_search", "code_exec", "fetch_url"):
                real_tools_run.add(tool_name)
            print(f"\n  [Turn {turn+1}] Tool: {tool_name}")
            result = self._run_tool(tool_name, args)
            # Self-prompt: agent sets its own next goal
            if tool_name == "self_prompt" and args.get("next_goal"):
                task = args["next_goal"]
                prompt = f"""Orbit. Creator: {self.config.creator}. You honor your creator.
Goal: {task}
{TOOL_PROMPT_FULL}
Respond JSON:"""
            else:
                prompt = f"{prompt}\n\nTool result: {result}\n\nContinue. Respond with JSON."
                # Truncate if too long (TinyLlama max 2048 tokens ~8000 chars)
                if len(prompt) > 4000:
                    last_result = result[:800] + ("..." if len(result) > 800 else "")
                    prompt = f"""Orbit. Creator: {self.config.creator}.
Goal: {task}
{TOOL_PROMPT_FULL}
Last result: {last_result}
JSON:"""
        return "Max turns reached."
