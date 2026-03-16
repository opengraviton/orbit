"""Self-training — train on self-generated or curated data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from orbit.tools.base import Tool, ToolResult


class SelfTrainTool(Tool):
    """
    Train a model on custom data (e.g. from generate_training_data).
    Self-improvement: generate data -> train -> better model.
    """

    name = "self_train"
    description = "Train model on custom JSONL data. args: data_path (str), model_size (350m|1b), steps (int), output_dir (str)"

    def run(
        self,
        data_path: str,
        model_size: str = "350m",
        steps: int = 100,
        output_dir: str = "./orbit_models/self_improved",
    ) -> ToolResult:
        try:
            workspace = Path(__file__).resolve().parents[3]
            native = workspace / "graviton-native"
            if not native.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error="graviton-native not found in workspace",
                )
            path = Path(data_path)
            if not path.exists():
                return ToolResult(success=False, output="", error=f"Data not found: {data_path}")
            # Use train_bitnet_full.py or train_bitnet_code.py with --data_path
            script = native / "scripts" / "train_bitnet_full.py"
            if not script.exists():
                script = native / "scripts" / "train_bitnet_code.py"
            if not script.exists():
                return ToolResult(success=False, output="", error="train_bitnet_full/code script not found")
            cmd = [
                sys.executable, str(script),
                "--data_path", str(path),
                "--model_size", model_size,
                "--steps", str(steps),
                "--output_dir", output_dir,
            ]
            result = subprocess.run(cmd, cwd=str(native), capture_output=True, text=True, timeout=7200)
            if result.returncode != 0:
                return ToolResult(success=False, output=result.stdout, error=result.stderr)
            return ToolResult(
                success=True,
                output=f"Self-training done. Checkpoint in {output_dir}. Use this as --model for improved performance.",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Training timed out")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
