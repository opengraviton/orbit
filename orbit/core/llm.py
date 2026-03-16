"""
LLM backend — Graviton or HuggingFace transformers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class LLMBackend(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        pass


class GravitonBackend(LLMBackend):
    """Use Graviton engine for local inference."""

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self._engine = None

    def _ensure_loaded(self):
        if self._engine is None:
            from graviton.core.engine import GravitonEngine
            from graviton.core.config import GravitonConfig
            cfg = GravitonConfig(model_path=str(self.model_path))
            self._engine = GravitonEngine(config=cfg)
            self._engine.load_model()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self._ensure_loaded()
        return self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)


class TransformersBackend(LLMBackend):
    """Use HuggingFace transformers (fallback)."""

    def __init__(self, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=getattr(torch, "float16", torch.float16),
                device_map="auto",
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        import sys
        from transformers import TextIteratorStreamer

        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("  Generating... ", end="", flush=True)
        import threading
        # do_sample=False when temp very low to avoid nan in multinomial
        do_sample = temperature > 0.2
        gen_kw = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-7),
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )
        result = []
        def run():
            try:
                out = self._model.generate(**gen_kw)
                result.append(self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            except RuntimeError as e:
                if "inf" in str(e) or "nan" in str(e):
                    # Retry with greedy (temperature=0)
                    gen_kw["temperature"] = 1e-7
                    gen_kw["do_sample"] = False
                    out = self._model.generate(**gen_kw)
                    result.append(self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
                else:
                    raise
        t = threading.Thread(target=run)
        t.start()
        for tok in streamer:
            print(tok, end="", flush=True)
        t.join()
        print()
        return result[0] if result else ""


def get_llm(model_path: str, use_graviton: bool = True) -> LLMBackend:
    """Create LLM backend. Prefer Graviton for local checkpoints."""
    p = Path(model_path)
    if use_graviton and (p.exists() or "/" not in model_path and ":" not in model_path):
        try:
            return GravitonBackend(model_path)
        except Exception:
            pass
    return TransformersBackend(model_path)
