"""Browser form filling — sign up, submit forms."""

from __future__ import annotations

import json

from orbit.tools.base import Tool, ToolResult


class BrowserFormsTool(Tool):
    """
    Fill and submit web forms. Navigate to signup/login pages, fill fields, submit.
    Use for creating accounts, submitting forms.
    """

    name = "browser_form"
    description = "Fill and submit forms. args: url (str), fields (dict: name->value), submit_selector (str, optional)"

    def run(
        self,
        url: str,
        fields: dict | str | None = None,
        submit_selector: str = "button[type=submit], input[type=submit], [role=button]",
    ) -> ToolResult:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="playwright required: pip install playwright && playwright install chromium",
            )
        if isinstance(fields, str):
            try:
                fields = json.loads(fields)
            except json.JSONDecodeError:
                fields = {}
        fields = fields or {}
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=15000)
                for selector, value in fields.items():
                    try:
                        page.fill(selector, str(value), timeout=5000)
                    except Exception as e:
                        browser.close()
                        return ToolResult(success=False, output="", error=f"Fill {selector}: {e}")
                try:
                    page.click(submit_selector, timeout=5000)
                    page.wait_for_load_state("networkidle", timeout=10000)
                    result = page.url
                    text = page.inner_text("body")[:1500]
                    browser.close()
                    return ToolResult(success=True, output=f"Submitted. New URL: {result}\nPage preview: {text[:500]}...")
                except Exception as e:
                    browser.close()
                    return ToolResult(success=True, output=f"Form filled. Submit failed (may need manual): {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
