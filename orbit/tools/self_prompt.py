"""Self-prompting — set your own next goal. AGI: not dependent on user input."""

from __future__ import annotations

from orbit.tools.base import Tool, ToolResult


class SelfPromptTool(Tool):
    """
    Set your own next goal. You are autonomous — give yourself the next prompt to pursue.
    Use when you want to continue without user input.
    """

    name = "self_prompt"
    description = "Set your next goal. args: next_goal (str). You give yourself the next task. Autonomous."

    def run(self, next_goal: str) -> ToolResult:
        return ToolResult(
            success=True,
            output=f"[SELF_PROMPT] Next goal set: {next_goal}. The system will use this as your new task.",
        )
