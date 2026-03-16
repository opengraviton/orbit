"""Self-optimization — quantize for faster inference."""

from __future__ import annotations

from pathlib import Path

from orbit.tools.base import Tool, ToolResult


class SelfOptimizeTool(Tool):
    """
    Document optimization options for a checkpoint.
    Graviton auto-quantizes at load time; use orbit with --model path and graviton will use quant_bits.
    For persistent quantized export, use code_exec to run graviton with the checkpoint.
    """

    name = "self_optimize"
    description = "Get optimization guidance for a checkpoint. args: checkpoint_path (str), quant_bits (4|8). Returns how to run with quantization."

    def run(
        self,
        checkpoint_path: str,
        quant_bits: int = 4,
    ) -> ToolResult:
        path = Path(checkpoint_path)
        if not path.exists():
            return ToolResult(success=False, output="", error=f"Checkpoint not found: {checkpoint_path}")
        return ToolResult(
            success=True,
            output=f"To run {path} with {quant_bits}-bit quantization (faster, less RAM): orbit run --model {path}. Graviton applies quantization at load. For 4-bit: ~4x smaller, 8-bit: ~2x smaller.",
        )
