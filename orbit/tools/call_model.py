"""Call any model — OpenAI, Anthropic, HF, local. AGI-level: prompt other models, get what you need."""

from __future__ import annotations

import os
from pathlib import Path

from orbit.tools.base import Tool, ToolResult


class CallModelTool(Tool):
    """
    Call any LLM: OpenAI, Anthropic, HuggingFace, or local path.
    You are not dependent on one model — use the best tool for each task.
    """

    name = "call_model"
    description = "Call any model. args: model (str: openai:gpt-4|anthropic:claude|hf:model-id|path), prompt (str), max_tokens (int)"

    def run(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> ToolResult:
        try:
            # Reject placeholders from prompt examples
            if not model or model in ("model-id", "repo-id", "path") or len(model) < 4:
                return ToolResult(
                    success=False,
                    output="",
                    error="Use a real model: hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0, openai:gpt-4, or local path",
                )
            if model.startswith("openai:"):
                model_id = model[7:]
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    return ToolResult(success=False, output="", error="OPENAI_API_KEY required")
                import httpx
                r = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
                    timeout=60,
                )
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"]
                return ToolResult(success=True, output=out)
            if model.startswith("anthropic:"):
                model_id = model[10:]
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return ToolResult(success=False, output="", error="ANTHROPIC_API_KEY required")
                import httpx
                r = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                    json={"model": model_id, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]},
                    timeout=60,
                )
                r.raise_for_status()
                out = r.json()["content"][0]["text"]
                return ToolResult(success=True, output=out)
            if model.startswith("hf:") or "/" in model and not Path(model).exists():
                model_id = model[3:] if model.startswith("hf:") else model
                from orbit.core.llm import get_llm
                llm = get_llm(model_id)
                out = llm.generate(prompt, max_tokens=max_tokens)
                return ToolResult(success=True, output=out)
            # Local path
            from orbit.core.llm import get_llm
            llm = get_llm(model)
            out = llm.generate(prompt, max_tokens=max_tokens)
            return ToolResult(success=True, output=out)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
