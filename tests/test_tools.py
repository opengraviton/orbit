"""Orbit tools tests."""

import tempfile
from pathlib import Path

import pytest

from orbit.tools.base import ToolResult
from orbit.tools.self_prompt import SelfPromptTool
from orbit.tools.code_exec import CodeExecTool
from orbit.tools.fetch_url import FetchUrlTool


def test_self_prompt_tool():
    tool = SelfPromptTool()
    result = tool.run(next_goal="Generate 50 training examples")
    assert result.success
    assert "Next goal set" in result.output
    assert "50 training examples" in result.output


def test_code_exec_tool():
    tool = CodeExecTool()
    result = tool.run(code="print(2 + 2)")
    assert result.success
    assert "4" in result.output


def test_code_exec_tool_error():
    tool = CodeExecTool()
    result = tool.run(code="1/0")
    assert not result.success
    assert "ZeroDivision" in (result.error or "") or "division" in (result.error or "").lower()


@pytest.mark.slow
def test_fetch_url_tool_invalid_url():
    """Fetch invalid URL should fail (network call)."""
    tool = FetchUrlTool()
    result = tool.run(url="http://invalid-nonexistent-domain-xyz123.local/")
    assert not result.success
