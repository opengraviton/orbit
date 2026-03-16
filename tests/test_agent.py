"""Orbit agent tests."""

import pytest

from orbit.core.agent import Agent, AgentConfig
from orbit.tools import TOOLS


def test_agent_config_defaults():
    config = AgentConfig()
    assert config.model_path == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.max_turns == 10


def test_parse_tool_call_tool():
    agent = Agent(AgentConfig())
    text = 'Here is my call: {"tool": "web_search", "args": {"query": "test"}}'
    parsed = agent._parse_tool_call(text)
    assert parsed is not None
    assert parsed.get("tool") == "web_search"
    assert parsed.get("args", {}).get("query") == "test"


def test_parse_tool_call_done():
    agent = Agent(AgentConfig())
    text = 'Finished: {"done": true, "result": "summary"}'
    parsed = agent._parse_tool_call(text)
    assert parsed is not None
    assert parsed.get("done") is True
    assert parsed.get("result") == "summary"


def test_parse_tool_call_self_prompt():
    agent = Agent(AgentConfig())
    text = 'Next: {"self_prompt": "Generate 50 examples"}'
    parsed = agent._parse_tool_call(text)
    assert parsed is not None
    assert parsed.get("tool") == "self_prompt"
    assert parsed.get("args", {}).get("next_goal") == "Generate 50 examples"


def test_run_tool_unknown():
    agent = Agent(AgentConfig())
    result = agent._run_tool("unknown_tool", {})
    assert "Unknown tool" in result


def test_tools_registered():
    assert "code_exec" in TOOLS
    assert "web_search" in TOOLS
    assert "fetch_url" in TOOLS
    assert "self_prompt" in TOOLS
    assert "call_model" in TOOLS
