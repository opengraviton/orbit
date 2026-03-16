"""Fetch data from any URL — HTML, JSON, text, files."""

from __future__ import annotations

from pathlib import Path

import httpx

from orbit.tools.base import Tool, ToolResult


class FetchUrlTool(Tool):
    """Fetch content from any URL. Download files, get HTML/JSON/text."""

    name = "fetch_url"
    description = "Fetch data from URL. args: url (str), save_to (str, optional — save to file), as_json (bool, optional)"

    def run(
        self,
        url: str,
        save_to: str | None = None,
        as_json: bool = False,
    ) -> ToolResult:
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            if save_to:
                Path(save_to).parent.mkdir(parents=True, exist_ok=True)
                Path(save_to).write_bytes(resp.content)
                return ToolResult(success=True, output=f"Saved {len(resp.content)} bytes to {save_to}")
            if as_json:
                return ToolResult(success=True, output=str(resp.json()))
            text = resp.text
            if len(text) > 8000:
                text = text[:8000] + "\n... (truncated)"
            return ToolResult(success=True, output=text)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
