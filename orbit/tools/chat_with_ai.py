"""Chat with other AIs — HuggingFace Inference API (free tier). Learn from different models."""

from __future__ import annotations

import os

from orbit.tools.base import Tool, ToolResult

# Free models on HF Inference API (no gating or with free tier)
DEFAULT_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
]


class ChatWithAITool(Tool):
    """
    Chat with another AI via HuggingFace Inference API (free tier).
    Use when you want a second opinion, to learn from a different model, or explore ideas.
    Requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN.
    """

    name = "chat_with_ai"
    description = (
        "Chat with another AI (HuggingFace free models). args: prompt (str), model (str, optional). "
        "Use to learn, get second opinion, or explore. Models: HuggingFaceH4/zephyr-7b-beta, "
        "mistralai/Mistral-7B-Instruct-v0.3, google/gemma-2-2b-it"
    )

    def run(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> ToolResult:
        if not prompt or len(prompt.strip()) < 3:
            return ToolResult(success=False, output="", error="prompt must be at least 3 chars")
        model = (model or "").strip() or DEFAULT_MODELS[0]
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            return ToolResult(
                success=False,
                output="",
                error="HF_TOKEN or HUGGING_FACE_HUB_TOKEN required. Get free token at https://huggingface.co/settings/tokens",
            )
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=token)
            messages = [{"role": "user", "content": prompt}]
            out = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            text = ""
            if out.choices:
                c = out.choices[0]
                if c.message and c.message.content:
                    text = c.message.content
            if not text:
                return ToolResult(success=False, output="", error="No response from model")
            return ToolResult(success=True, output=text)
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="huggingface_hub required: pip install huggingface_hub",
            )
        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err:
                return ToolResult(success=False, output="", error="Invalid HF token. Check HF_TOKEN.")
            if "404" in err or "not found" in err:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Model {model} not found. Try: {', '.join(DEFAULT_MODELS[:3])}",
                )
            return ToolResult(success=False, output="", error=err)
