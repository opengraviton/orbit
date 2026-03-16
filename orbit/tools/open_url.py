"""Open URL in default browser — uses system 'open' (macOS), always shows on screen."""

from __future__ import annotations

import platform
import subprocess

from orbit.tools.base import Tool, ToolResult


class OpenUrlTool(Tool):
    """Open URL in default browser. Uses system command — always visible on screen."""

    name = "open_url"
    description = "Open URL in default browser (visible). args: url (str)"

    def run(self, url: str) -> ToolResult:
        try:
            if platform.system() == "Darwin":
                subprocess.run(["open", url], check=True, timeout=10)
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", url], check=True, timeout=10)
            elif platform.system() == "Windows":
                subprocess.run(["start", "", url], check=True, timeout=10, shell=True)
            else:
                import webbrowser
                webbrowser.open(url)
            return ToolResult(success=True, output=f"Opened {url} in default browser.")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
