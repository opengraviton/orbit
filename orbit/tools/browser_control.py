"""Full browser control — open visible browser, navigate, click, type, screenshot."""

from __future__ import annotations

import platform
import subprocess

from orbit.tools.base import Tool, ToolResult


def _open_url_system(url: str) -> ToolResult:
    """Open URL in default browser via system command. Works when webbrowser fails."""
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", url], check=True, timeout=5)
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", url], check=True, timeout=5)
        elif platform.system() == "Windows":
            subprocess.run(["start", "", url], check=True, timeout=5, shell=True)
        else:
            import webbrowser
            webbrowser.open(url)
        return ToolResult(success=True, output=f"Opened {url} in default browser.")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))

# Persistent browser instance — stays open across tool calls
_playwright = None
_browser = None
_page = None


def _get_browser(headless: bool = False):
    """Get or create persistent browser. headless=False opens visible window."""
    global _playwright, _browser, _page
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, "playwright required: pip install playwright && playwright install chromium"
    if _browser is None:
        try:
            _playwright = sync_playwright().start()
            _browser = _playwright.chromium.launch(headless=headless)
            _page = _browser.new_page()
            return _page, None
        except Exception as e:
            return None, str(e)
    return _page, None


def _close_browser():
    """Close persistent browser."""
    global _playwright, _browser, _page
    if _browser:
        try:
            _browser.close()
        except Exception:
            pass
        _browser = None
        _page = None
    if _playwright:
        try:
            _playwright.stop()
        except Exception:
            pass
        _playwright = None


class BrowserOpenTool(Tool):
    """Open visible browser window. Stays open for subsequent actions."""

    name = "browser_open"
    description = "Open browser window (visible). No args. Use before navigate/click/type."

    def run(self) -> ToolResult:
        page, err = _get_browser(headless=False)
        if err:
            # Fallback: use system 'open' (macOS) or xdg-open (Linux) — more reliable than webbrowser
            return _open_url_system("https://google.com")
        return ToolResult(success=True, output="Browser opened (visible). Use browser_navigate, browser_click, browser_type.")


class BrowserNavigateTool(Tool):
    """Navigate to URL."""

    name = "browser_navigate"
    description = "Navigate to URL. args: url (str)"

    def run(self, url: str) -> ToolResult:
        page, err = _get_browser(headless=False)
        if err:
            # Fallback: use system 'open' (macOS) — shows browser on screen
            return _open_url_system(url)
        try:
            page.goto(url, timeout=30000)
            return ToolResult(success=True, output=f"Navigated to {url}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class BrowserClickTool(Tool):
    """Click element by selector or text."""

    name = "browser_click"
    description = "Click element. args: selector (str, e.g. 'button', 'a', '#id') or text (str)"

    def run(self, selector: str | None = None, text: str | None = None) -> ToolResult:
        page, err = _get_browser(headless=False)
        if err:
            return ToolResult(success=False, output="", error=err)
        try:
            if text:
                page.click(f"text={text}", timeout=5000)
            elif selector:
                page.click(selector, timeout=5000)
            else:
                return ToolResult(success=False, output="", error="Provide selector or text")
            return ToolResult(success=True, output=f"Clicked: {text or selector}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class BrowserTypeTool(Tool):
    """Type text into focused element or selector."""

    name = "browser_type"
    description = "Type text. args: text (str), selector (str, optional — default focused element)"

    def run(self, text: str, selector: str | None = None) -> ToolResult:
        page, err = _get_browser(headless=False)
        if err:
            return ToolResult(success=False, output="", error=err)
        try:
            if selector:
                page.fill(selector, text, timeout=5000)
            else:
                page.keyboard.type(text, delay=50)
            return ToolResult(success=True, output=f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class BrowserScreenshotTool(Tool):
    """Take screenshot."""

    name = "browser_screenshot"
    description = "Take screenshot. args: path (str, default /tmp/orbit_screenshot.png)"

    def run(self, path: str = "/tmp/orbit_screenshot.png") -> ToolResult:
        page, err = _get_browser(headless=False)
        if err:
            return ToolResult(success=False, output="", error=err)
        try:
            page.screenshot(path=path)
            return ToolResult(success=True, output=f"Screenshot: {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class BrowserCloseTool(Tool):
    """Close browser."""

    name = "browser_close"
    description = "Close browser. No args."

    def run(self) -> ToolResult:
        _close_browser()
        return ToolResult(success=True, output="Browser closed.")
