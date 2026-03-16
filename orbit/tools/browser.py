"""Browser control via Playwright."""

from __future__ import annotations

from orbit.tools.base import Tool, ToolResult


class BrowserTool(Tool):
    """Control Chromium browser. Requires: pip install playwright && playwright install chromium."""

    name = "browser"
    description = "Control browser: navigate, click, extract text. args: action (navigate|screenshot|extract), url (optional)"

    def run(self, action: str = "navigate", url: str = "about:blank") -> ToolResult:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="playwright required: pip install playwright && playwright install chromium",
            )
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                if action == "navigate" and url:
                    page.goto(url, timeout=15000)
                    content = page.content()[:2000]
                    browser.close()
                    return ToolResult(success=True, output=f"Loaded. HTML preview:\n{content[:500]}...")
                elif action == "screenshot":
                    page.goto(url, timeout=15000)
                    page.screenshot(path="/tmp/orbit_screenshot.png")
                    browser.close()
                    return ToolResult(success=True, output="Screenshot saved to /tmp/orbit_screenshot.png")
                elif action == "extract":
                    page.goto(url, timeout=15000)
                    text = page.inner_text("body")[:3000]
                    browser.close()
                    return ToolResult(success=True, output=text)
                browser.close()
            return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
