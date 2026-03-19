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
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

    def _format_prompt(self, prompt: str) -> str:
        """Use chat template for Instruct models, else raw prompt."""
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template is not None:
            try:
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        import sys
        from transformers import TextIteratorStreamer, StoppingCriteria

        self._ensure_loaded()
        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("  Generating... ", end="", flush=True)
        import threading
        # do_sample=False when temp very low to avoid nan in multinomial
        do_sample = temperature > 0.2
        # Stop when model outputs 5+ consecutive "!" — cuts repetition loop short
        excl_ids = self._tokenizer.encode("!", add_special_tokens=False)
        excl_id = excl_ids[0] if excl_ids else None

        class StopAtExclamationRepetition(StoppingCriteria):
            def __init__(self, tid, thresh=5):
                self.tid = tid
                self.thresh = thresh

            def __call__(self, input_ids, scores, **kwargs):
                if self.tid is None or input_ids.shape[1] < self.thresh:
                    return False
                last = input_ids[0, -self.thresh:]
                return all(t == self.tid for t in last.tolist())

        stop_excl = StopAtExclamationRepetition(excl_id, 5) if excl_id is not None else None
        gen_kw = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-7),
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            repetition_penalty=1.25,  # Prevent !!!!!!!!! repetition loops
            streamer=streamer,
        )
        if stop_excl is not None:
            from transformers import StoppingCriteriaList
            gen_kw["stopping_criteria"] = StoppingCriteriaList([stop_excl])
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
