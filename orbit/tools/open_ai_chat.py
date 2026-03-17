"""Open AI chat UIs in browser — ChatGPT, HuggingFace Chat. Use when you want to consult another AI interactively."""

from __future__ import annotations

import webbrowser
from urllib.parse import quote

from orbit.tools.base import Tool, ToolResult

CHAT_URLS = {
    "chatgpt": "https://chat.openai.com",
    "huggingface": "https://huggingface.co/chat",
    "hf": "https://huggingface.co/chat",
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
    "perplexity": "https://www.perplexity.ai",
}


class OpenAIChatTool(Tool):
    """
    Open an AI chat interface in the default browser.
    Use when you want to consult ChatGPT, HuggingFace Chat, Claude, etc.
    The page opens for you (or the user) to interact with.
    """

    name = "open_ai_chat"
    description = (
        "Open AI chat in browser. args: target (str). "
        "target: huggingface (free models), chatgpt, claude, gemini, perplexity"
    )

    def run(self, target: str = "huggingface", prompt: str | None = None) -> ToolResult:
        key = (target or "huggingface").strip().lower()
        url = CHAT_URLS.get(key)
        if not url:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown target: {target}. Use: {', '.join(CHAT_URLS.keys())}",
            )
        try:
            if prompt and key in ("huggingface", "hf"):
                url = f"https://huggingface.co/chat?prompt={quote(prompt)}"
            elif prompt and key == "perplexity":
                url = f"https://www.perplexity.ai/search?q={quote(prompt)}"
            webbrowser.open(url)
            return ToolResult(
                success=True,
                output=f"Opened {key} chat in browser. You can ask your question there.",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
