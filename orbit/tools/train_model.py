"""Train small models via graviton-native."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from orbit.tools.base import Tool, ToolResult


class TrainModelTool(Tool):
    """Train a small BitNet/Omega model. Uses graviton-native."""

    name = "train_model"
    description = "Train a small model (350m, 1b, omega-micro). Requires graviton-native. Returns checkpoint path."

    def run(
        self,
        model_size: str = "350m",
        steps: int = 100,
        output_dir: str = "./orbit_models",
    ) -> ToolResult:
        try:
            # orbit/orbit/tools/ -> go up to workspace (ai/)
            workspace = Path(__file__).resolve().parents[3]
            native = workspace / "graviton-native"
            if not native.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error="graviton-native not found in workspace",
                )
            cmd = [
                sys.executable, "-m", "graviton_native.cli", "run",
                "--num_gpu_cores", "32",
                "--model_size", model_size,
                "--steps", str(steps),
                "--output_dir", output_dir,
            ]
            result = subprocess.run(cmd, cwd=str(native), capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                return ToolResult(success=False, output=result.stdout, error=result.stderr)
            return ToolResult(success=True, output=f"Training done. Checkpoints in {output_dir}")
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Training timed out")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
