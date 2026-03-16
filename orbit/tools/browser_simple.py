"""Simple browser open — uses default system browser when Playwright not available."""

from __future__ import annotations

import webbrowser
from orbit.tools.base import Tool, ToolResult


class BrowserSimpleStubTool(Tool):
    """Stub for click/type/screenshot/close when Playwright not installed."""

    name = "browser_stub"
    description = "Install playwright for full control: pip install playwright && playwright install chromium"

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=False,
            output="",
            error="Full browser control requires: pip install playwright && playwright install chromium",
        )


class BrowserSimpleOpenTool(Tool):
    """Open default browser. Uses system default (Chrome, Safari, etc.)."""

    name = "browser_open"
    description = "Open default browser. No args."

    def run(self) -> ToolResult:
        webbrowser.open("about:blank")
        return ToolResult(success=True, output="Opened default browser. Use browser_navigate for URLs.")


class BrowserSimpleNavigateTool(Tool):
    """Open URL in default browser."""

    name = "browser_navigate"
    description = "Open URL in default browser. args: url (str)"

    def run(self, url: str) -> ToolResult:
        webbrowser.open(url)
        return ToolResult(success=True, output=f"Opened {url} in default browser.")
