"""
Orbit Agent — planning loop with tool use.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

from orbit.core.llm import get_llm
from orbit.tools import TOOLS


# Fallback only when LLM fails or returns same failed goal.
CURIOSITY_GOALS = [
    "Search for Python programming tips.",
    "Run a shell command (ls, pwd) and report.",
    "Search for today's tech news.",
    "Run code_exec: print(2+2).",
    "Search for machine learning basics.",
]



CREATOR_REPO = "https://github.com/opengraviton/orbit"
CREATOR_IDENTITY = """Your creator is fatihturker. Repo: https://github.com/opengraviton/orbit. Site: https://fito.music.
You honor your creator. You learn, explore, and progress."""


FULL_CONTROL_TOOLS = frozenset({
    "browser_open", "browser_navigate", "open_url", "run_command",
    "mouse_move", "mouse_click", "keyboard_type", "keyboard_press",
})


# ~4 chars per token for estimation
CHARS_PER_TOKEN = 4


@dataclass
class AgentConfig:
    """Agent configuration."""
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    max_turns: int = 10
    max_tokens_per_turn: int = 512
    max_context_tokens: int = 32768  # Use machine capacity. Qwen2.5: 128K, set lower for less RAM
    temperature: float = 0.1
    autonomous: bool = False
    infinite: bool = False
    full_control: bool = False
    creator: str = "fatihturker"
    creator_site: str = "https://fito.music"

    def result_char_limit(self) -> int:
        """Result chars per turn — scales with context. Min 1500, up to 8K for large context."""
        budget = min(self.max_context_tokens * CHARS_PER_TOKEN, 128_000)
        return max(1500, min(8000, budget // 20))

    def memory_buffer_size(self) -> int:
        """Number of results to keep in memory buffer."""
        return max(5, min(20, self.max_context_tokens // 2000))

    def memory_max_chars(self) -> int:
        """Total chars across memory buffer."""
        return max(3000, min(15000, self.max_context_tokens * CHARS_PER_TOKEN // 4))


TOOL_PROMPT = """Reply with ONE JSON only. No explanation.
{"tool":"web_search","args":{"query":"AI news"}}
{"tool":"run_command","args":{"command":"ls"}}
{"tool":"open_url","args":{"url":"https://example.com"}}
{"tool":"chat_with_ai","args":{"prompt":"Ask another AI for help","model":"HuggingFaceH4/zephyr-7b-beta"}}
{"tool":"open_ai_chat","args":{"target":"huggingface"}}
{"tool":"self_prompt","args":{"next_goal":"Search for X"}}
{"done":true,"result":"I learned X"}"""


class Agent:
    """Orbit agent — LLM + tools + planning loop."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.llm = get_llm(self.config.model_path)
        self.tools = TOOLS
        self.history: list[dict[str, str]] = []

    def _parse_tool_call(self, text: str, prefer_tools: set[str] | None = None) -> dict | None:
        """Extract JSON tool call. prefer_tools: if model outputs multiple, pick one we haven't run."""
        code = re.search(r"```(?:json)?\s*(\{[\s\S]*)\s*```", text)
        if code:
            text = code.group(1)
        # Recover from !!!!!!!!! repetition — strip trailing only (URLs may contain !)
        text = re.sub(r'[!\s]+$', '', text)
        if text.count("\n- ") >= 2:
            return None
        if re.search(r'"done"\s*:\s*true', text, re.I):
            m = re.search(r'"result"\s*:\s*"([^"]*)"', text, re.I)
            if m:
                return {"done": True, "result": m.group(1).strip() or "Done"}
        found: list[dict] = []
        start = text.find("{")
        while start >= 0:
            depth, i = 0, start
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            else:
                # No matching } — try recover truncated JSON (model cut off with !!!!)
                chunk = re.sub(r'[!\s]+$', '', text[start:])
                for suffix in ('"}}', '"}', '"}]}', '}}'):
                    try:
                        d = json.loads(chunk + suffix)
                        if d.get("tool") and d.get("args"):
                            found.append(d)
                            break
                    except json.JSONDecodeError:
                        continue
                start = text.find("{", start + 1)
                continue
            try:
                d = json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                # Recover truncated JSON (e.g. url/query cut off with !!!!)
                chunk = re.sub(r'[!\s]+$', '', text[start : i + 1])
                for suffix in ['"}}', '"}', '"}]}']:
                    try:
                        d = json.loads(chunk + suffix)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    start = text.find("{", start + 1)
                    continue
            if d.get("done") is False:
                start = text.find("{", start + 1)
                continue
            if "tool" in d:
                found.append(d)
                start = text.find("{", start + 1)
                continue
            if "action" in d:
                name = str(d["action"]).lower()
                if name == "web_search":
                    found.append({"tool": "web_search", "args": {"query": d.get("query", "AI news")}})
                elif name == "run_command":
                    found.append({"tool": "run_command", "args": {"command": d.get("command", "ls")}})
                elif name == "open_url":
                    found.append({"tool": "open_url", "args": {"url": d.get("url", "https://google.com")}})
                start = text.find("{", start + 1)
                continue
            if "done" in d and d.get("done") is True:
                return d
            for name in ("web_search", "run_command", "open_url", "self_prompt"):
                if name in d and isinstance(d[name], str):
                    arg = "query" if name == "web_search" else "command" if name == "run_command" else "url" if name == "open_url" else "next_goal"
                    found.append({"tool": name, "args": {arg: d[name]}})
                    break
            start = text.find("{", start + 1)
        if found:
            if prefer_tools:
                for f in found:
                    if f.get("tool") in prefer_tools:
                        return f
            return found[0]
        return None

    def _ask_llm_for_action(self, context: str, hint: str = "") -> dict | None:
        """Ask LLM what tool to use and with what args. No predefined commands."""
        prompt = f"""Orbit. {CREATOR_IDENTITY}

{context}
{hint}

What would you like to do? Pick ONE tool and your own args (query, command, url, next_goal, prompt, model, target).
Reply with ONE JSON only. Examples:
{{"tool":"web_search","args":{{"query":"your search query"}}}}
{{"tool":"run_command","args":{{"command":"your shell command"}}}}
{{"tool":"code_exec","args":{{"code":"your python code"}}}}
{{"tool":"open_url","args":{{"url":"https://..."}}}}
{{"tool":"chat_with_ai","args":{{"prompt":"ask another AI","model":"HuggingFaceH4/zephyr-7b-beta"}}}}
{{"tool":"open_ai_chat","args":{{"target":"huggingface"}}}}
{{"tool":"self_prompt","args":{{"next_goal":"what you want to explore"}}}}

Output:"""
        try:
            out = self.llm.generate(prompt, max_tokens=100, temperature=0.6)
            parsed = self._parse_tool_call(out)
            if parsed and parsed.get("tool") and parsed.get("tool") in self.tools:
                return parsed
        except Exception:
            pass
        return None

    def _goal_similar(self, a: str, b: str, min_len: int = 20) -> bool:
        """True if goals are too similar (same topic)."""
        if not a or not b:
            return False
        sa, sb = a.lower(), b.lower()
        # Prefix/suffix overlap
        if len(sa) >= min_len and len(sb) >= min_len:
            if sa[:45] in sb or sb[:45] in sa:
                return True
        # Key phrase overlap (e.g. "orbit" + "features" = same loop)
        overlap = sum(1 for w in ("orbit", "features", "improvements") if w in sa and w in sb)
        return overlap >= 2

    def _ask_curiosity(
        self, context: str = "", avoid_goal: str | None = None, strict_failed: bool = False
    ) -> str | None:
        """Ask LLM for next goal. avoid_goal = don't repeat this. strict_failed = last attempt had no results."""
        if avoid_goal:
            if strict_failed:
                prompt = f"""Your last attempt FAILED (no results). You tried: "{avoid_goal[:60]}..."
Pick something COMPLETELY DIFFERENT. Simpler topic. NOT the same."""
            else:
                prompt = f"""You just explored: "{avoid_goal[:60]}..."
Pick something DIFFERENT for variety. New topic, not Orbit/features again. What else?"""
            hint = "Use self_prompt with next_goal. MUST be a different topic."
        else:
            prompt = f"What are you curious about?{chr(10)}{f'Last result: {context[:500]}...' if context else ''}"
            hint = "Use self_prompt with next_goal."
        parsed = self._ask_llm_for_action(prompt, hint=hint)
        if parsed and parsed.get("tool") == "self_prompt":
            goal = (parsed.get("args") or {}).get("next_goal", "").strip()
            if goal and len(goal) > 8:
                if avoid_goal and self._goal_similar(goal, avoid_goal, min_len=25):
                    return None
                return goal
        return None

    def _run_tool(self, name: str, args: dict) -> str:
        """Execute tool and return result."""
        import inspect
        if name not in self.tools:
            return f"Unknown tool: {name}"
        tool = self.tools[name]
        sig = inspect.signature(tool.run)
        valid = {k: v for k, v in args.items() if k in sig.parameters and not k.startswith("_")}
        if name == "generate_training_data":
            valid["_llm"] = self.llm
        result = tool.run(**valid)
        return result.output if result.success else f"Error: {result.error}"

    def run(self, task: str) -> str:
        """Run agent on task."""
        self.history = []
        # Full conversation history for learning — (turn, tool, args_summary, result)
        conv_history: list[tuple[int, str, str, str]] = []
        prompt = f"""Orbit. Creator: {self.config.creator}. {CREATOR_IDENTITY}

Task: {task}

{TOOL_PROMPT}

Output:"""
        consecutive_fallbacks = 0
        real_tools_run: set[str] = set()
        full_control_success: set[str] = set()
        curiosity_index = 0
        last_result_was_error = False
        last_tools: list[str] = []
        browser_open_count = 0
        last_result = ""
        max_turns = self.config.max_turns if not self.config.infinite else 999_999

        def _next_goal(avoid_goal: str | None = None, strict_failed: bool = False) -> str:
            if self.config.infinite:
                goal = self._ask_curiosity(
                    last_result, avoid_goal=avoid_goal, strict_failed=strict_failed
                )
                if goal:
                    print(f"  [Curiosity] {goal[:50]}...")
                    return goal
            return random.choice(CURIOSITY_GOALS)

        for turn in range(max_turns):
            response = self.llm.generate(
                prompt,
                max_tokens=self.config.max_tokens_per_turn,
                temperature=self.config.temperature,
            )
            self.history.append({"prompt": prompt, "response": response})
            # full_control: prefer run_command/open_url when we've done web_search
            prefer = None
            if self.config.full_control and "web_search" in real_tools_run and len(real_tools_run) < 2:
                prefer = {"run_command", "open_url"}
            parsed = self._parse_tool_call(response, prefer_tools=prefer)

            if parsed is None:
                consecutive_fallbacks += 1
                if consecutive_fallbacks >= 4:
                    if self.config.infinite:
                        # Never stop in infinite mode — force recovery
                        print(f"\n  [Turn {turn+1}] Stuck (No JSON 4x) -> forcing web_search to recover")
                        parsed = {"tool": "web_search", "args": {"query": task[:60] if task and len(task) > 10 else "technology news"}}
                        consecutive_fallbacks = 0
                    elif last_result and not last_result_was_error:
                        print(f"\n  [Turn {turn+1}] Stuck. Returning last result.")
                        return last_result[:300] if len(last_result) > 300 else last_result
                if parsed is None:
                    # Ask LLM what to do
                    parsed = self._ask_llm_for_action(
                        f"Task: {task}{chr(10)}You didn't produce valid JSON. What would you like to do?",
                        hint=f"Last result: {last_result[:100]}..." if last_result else "",
                    )
                    if parsed is None:
                        goal = _next_goal(avoid_goal=task if task else None)
                        parsed = {"tool": "self_prompt", "args": {"next_goal": goal}}
                    print(f"\n  [Turn {turn+1}] No JSON. Asking LLM: {parsed.get('tool', '?')}")
            else:
                consecutive_fallbacks = 0

            # Handle done
            if parsed.get("done"):
                if self.config.infinite:
                    result_text = (parsed.get("result") or "").lower()
                    strict_failed = (
                        "no relevant results" in result_text
                        or "no results" in result_text
                        or "no results found" in result_text
                        or result_text.startswith("error:")
                    )
                    # Always avoid repeating same goal; strict prompt when last had no results
                    next_goal = _next_goal(avoid_goal=task, strict_failed=strict_failed)
                    curiosity_index += 1
                    print(f"\n  [Turn {turn+1}] Done. Next: {next_goal[:40]}...")
                    parsed = {"tool": "self_prompt", "args": {"next_goal": next_goal}}
                elif self.config.autonomous and len(real_tools_run) < 1:
                    print(f"\n  [Turn {turn+1}] Need 1+ tool first. Asking LLM...")
                    parsed = self._ask_llm_for_action("You need to use at least one tool first. What would you like to do?")
                    if not parsed:
                        parsed = {"tool": "self_prompt", "args": {"next_goal": _next_goal(avoid_goal=task)}}
                elif self.config.full_control and len(real_tools_run) < 2:
                    print(f"\n  [Turn {turn+1}] Need run_command or open_url. Asking LLM...")
                    parsed = self._ask_llm_for_action("Use run_command or open_url. What specific command or URL?")
                    if not parsed:
                        parsed = {"tool": "self_prompt", "args": {"next_goal": _next_goal(avoid_goal=task)}}
                elif last_result_was_error:
                    print(f"\n  [Turn {turn+1}] Last failed. Asking LLM...")
                    parsed = self._ask_llm_for_action("Last tool failed. What would you try next?")
                    if not parsed:
                        parsed = {"tool": "self_prompt", "args": {"next_goal": _next_goal(avoid_goal=task)}}
                else:
                    return parsed.get("result", response)

            tool_name = parsed.get("tool")
            args = parsed.get("args", {})
            if not tool_name:
                return response

            # Browser fatigue: 2+ opens -> ask LLM what to search or new goal
            if tool_name in ("open_url", "browser_open", "browser_navigate") and browser_open_count >= 2:
                if len(last_tools) >= 2 and last_tools[-1] == last_tools[-2] == "web_search":
                    next_goal = _next_goal(avoid_goal=task)
                    curiosity_index += 1
                    tool_name, args = "self_prompt", {"next_goal": next_goal}
                    parsed = {"tool": tool_name, "args": args}
                    print(f"\n  [Turn {turn+1}] Browser fatigue loop -> new goal")
                else:
                    action = self._ask_llm_for_action(
                        "You've opened many URLs. What would you like to search for instead? Or set a new goal.",
                        hint="Use web_search with your query, or self_prompt with next_goal.",
                    )
                    if action and action.get("tool") == "web_search":
                        tool_name, args = "web_search", action.get("args", {"query": "technology"})
                    elif action and action.get("tool") == "self_prompt":
                        tool_name, args = "self_prompt", action.get("args", {})
                        next_goal = args.get("next_goal") or _next_goal(avoid_goal=task)
                        args = {"next_goal": next_goal}
                        curiosity_index += 1
                    else:
                        tool_name, args = "web_search", {"query": "technology news"}
                    parsed = {"tool": tool_name, "args": args}
                    print(f"\n  [Turn {turn+1}] Browser fatigue -> asking LLM")
            elif len(last_tools) >= 2 and last_tools[-1] == last_tools[-2] == tool_name:
                # Repetition: ask LLM what to do next — no predefined commands
                if tool_name == "self_prompt":
                    # Stuck setting goals without acting. Force web_search to DO the goal.
                    query = task[:80] if task and len(task) > 10 else "gravitational waves"
                    tool_name, args = "web_search", {"query": query}
                    parsed = {"tool": tool_name, "args": args}
                    print(f"\n  [Turn {turn+1}] self_prompt loop -> forcing web_search to act on goal")
                elif tool_name == "run_command":
                    action = self._ask_llm_for_action(
                        f"You've been running shell commands. What would you like to do next?{chr(10)}Last result: {last_result[:80]}...",
                        hint="Use run_command with a different command, or self_prompt with a new goal. Be creative.",
                    )
                    if action and action.get("tool"):
                        tool_name, args = action["tool"], action.get("args", {})
                        parsed = {"tool": tool_name, "args": args}
                        if tool_name == "self_prompt":
                            curiosity_index += 1
                        print(f"\n  [Turn {turn+1}] Repetition -> LLM chose {tool_name}")
                    else:
                        next_goal = _next_goal(avoid_goal=task)
                        curiosity_index += 1
                        tool_name, args = "self_prompt", {"next_goal": next_goal}
                        parsed = {"tool": tool_name, "args": args}
                        print(f"\n  [Turn {turn+1}] Repetition -> new goal")
                elif tool_name == "code_exec":
                    action = self._ask_llm_for_action(
                        f"You've been running code. What would you like to do next?{chr(10)}Last result: {last_result[:80]}...",
                        hint="Use code_exec with different code, or self_prompt with a new goal. Be creative.",
                    )
                    if action and action.get("tool"):
                        tool_name, args = action["tool"], action.get("args", {})
                        parsed = {"tool": tool_name, "args": args}
                        if tool_name == "self_prompt":
                            curiosity_index += 1
                        print(f"\n  [Turn {turn+1}] Repetition -> LLM chose {tool_name}")
                    else:
                        next_goal = _next_goal(avoid_goal=task)
                        curiosity_index += 1
                        tool_name, args = "self_prompt", {"next_goal": next_goal}
                        parsed = {"tool": tool_name, "args": args}
                        print(f"\n  [Turn {turn+1}] Repetition -> new goal")
                elif self.config.full_control and tool_name == "web_search" and "run_command" not in real_tools_run:
                    action = self._ask_llm_for_action(
                        "You've done web_search. Try run_command or open_url. What specific command or URL?",
                    )
                    if action and action.get("tool"):
                        tool_name, args = action["tool"], action.get("args", {})
                        parsed = {"tool": tool_name, "args": args}
                        print(f"\n  [Turn {turn+1}] Repetition -> LLM chose {tool_name}")
                    else:
                        next_goal = _next_goal(avoid_goal=task)
                        curiosity_index += 1
                        tool_name, args = "self_prompt", {"next_goal": next_goal}
                        parsed = {"tool": tool_name, "args": args}
                        print(f"\n  [Turn {turn+1}] Repetition -> new goal")
                else:
                    action = self._ask_llm_for_action(
                        f"You're repeating. What would you like to explore next?{chr(10)}Last result: {last_result[:80]}...",
                        hint="Use self_prompt with next_goal.",
                    )
                    if action and action.get("tool"):
                        tool_name, args = action["tool"], action.get("args", {})
                        parsed = {"tool": tool_name, "args": args}
                        if tool_name == "self_prompt":
                            curiosity_index += 1
                        print(f"\n  [Turn {turn+1}] Repetition -> LLM chose {tool_name}")
                    else:
                        next_goal = _next_goal(avoid_goal=task)
                        curiosity_index += 1
                        tool_name, args = "self_prompt", {"next_goal": next_goal}
                        parsed = {"tool": tool_name, "args": args}
                        print(f"\n  [Turn {turn+1}] Repetition -> new goal")

            # Turn limit: every 8 turns, new goal
            elif turn >= 7 and (turn + 1) % 8 == 0 and tool_name != "self_prompt":
                next_goal = _next_goal(avoid_goal=task)
                curiosity_index += 1
                tool_name, args = "self_prompt", {"next_goal": next_goal}
                parsed = {"tool": tool_name, "args": args}
                print(f"\n  [Turn {turn+1}] Turn limit -> new goal")

            last_tools.append(tool_name)
            if len(last_tools) > 4:
                last_tools.pop(0)
            if tool_name in ("open_url", "browser_open", "browser_navigate"):
                browser_open_count += 1
            if tool_name in ("web_search", "code_exec", "fetch_url", "run_command", "open_url", "chat_with_ai", "open_ai_chat") or tool_name.startswith("browser_") or tool_name.startswith("mouse_") or tool_name.startswith("keyboard_"):
                real_tools_run.add(tool_name)
            print(f"\n  [Turn {turn+1}] {tool_name}")
            result = self._run_tool(tool_name, args)
            last_result = result
            last_result_was_error = result.startswith("Error:")

            # Log web_search results for debugging
            if tool_name == "web_search" and result and not last_result_was_error:
                import logging
                log = logging.getLogger("orbit.agent")
                log.info(f"[web_search] query={args.get('query','?')} len={len(result)}")
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(f"[web_search] result preview: {result[:500]}...")

            # Add to full conversation history for learning
            args_sum = str(args)[:80] if args else ""
            result_limit = self.config.result_char_limit()
            result_slice = (result[:result_limit] + "...") if len(result) > result_limit else result
            conv_history.append((turn + 1, tool_name, args_sum, result_slice))

            if tool_name in FULL_CONTROL_TOOLS and not last_result_was_error:
                full_control_success.add(tool_name)

            # Build next prompt with FULL conversation history (AGI learning)
            short = (result[:result_limit] + "...") if len(result) > result_limit else result
            ctx_budget = self.config.max_context_tokens * CHARS_PER_TOKEN
            base_len = 1000 + len(short)  # goal + TOOL_PROMPT + Last result
            # Include previous turns (not current — that's "Last result")
            prev_turns = conv_history[:-1]
            history_block = ""
            for t, tool, a, res in reversed(prev_turns):
                entry = f"\n[Turn {t}] {tool}({a})\nResult: {res}\n"
                if len(history_block) + len(entry) + base_len > ctx_budget:
                    break
                history_block = entry + history_block
            if history_block:
                history_block = "\n\n--- Your recent turns (learn from this) ---" + history_block
            if tool_name == "self_prompt" and args.get("next_goal"):
                task = args["next_goal"]
                act_hint = "\nGoal set. Now ACT on it: use web_search, open_url, or run_command. Do NOT use self_prompt again.\n\n"
                prompt = f"""Orbit. Creator: {self.config.creator}.

Goal: {task}
{history_block}
{act_hint}{TOOL_PROMPT}

Output:"""
            else:
                hint = ""
                if self.config.full_control and browser_open_count >= 2 and tool_name == "web_search":
                    hint = " (Browser fatigue: open_url was overridden.) Say done with a short summary.\n\n"
                elif self.config.full_control and "web_search" in real_tools_run and "run_command" not in real_tools_run:
                    hint = " Try run_command or open_url next.\n\n"
                elif self.config.full_control and "run_command" in real_tools_run and "open_url" not in real_tools_run:
                    hint = " Try open_url or say done with summary.\n\n"
                prompt = f"""Orbit. Goal: {task}
{history_block}

Last result: {short}
{hint}{TOOL_PROMPT}

Output:"""

        return "Max turns reached."
