"""Create and use small specialized 'slave' models — train, deploy, call."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from orbit.tools.base import Tool, ToolResult


class TrainDeployTool(Tool):
    """
    Train a small specialized model and deploy to HuggingFace.
    Returns repo_id for use with call_deployed_model.
    """

    name = "train_deploy"
    description = "Train small model + deploy to HF. args: data_path (str), model_size (350m|1b), steps (int), repo_id (str). Returns repo_id."

    def run(
        self,
        data_path: str,
        model_size: str = "350m",
        steps: int = 50,
        repo_id: str = "orbit-slave",
        output_dir: str = "./orbit_slaves",
    ) -> ToolResult:
        try:
            workspace = Path(__file__).resolve().parents[3]
            native = workspace / "graviton-native"
            if not native.exists():
                return ToolResult(success=False, output="", error="graviton-native not found")
            path = Path(data_path)
            if not path.exists():
                return ToolResult(success=False, output="", error=f"Data not found: {data_path}")
            script = native / "scripts" / "train_bitnet_full.py"
            if not script.exists():
                script = native / "scripts" / "train_bitnet_code.py"
            if not script.exists():
                return ToolResult(success=False, output="", error="train script not found")
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(script),
                "--data_path", str(path),
                "--model_size", model_size,
                "--steps", str(steps),
                "--output_dir", str(out_dir),
            ]
            result = subprocess.run(cmd, cwd=str(native), capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                return ToolResult(success=False, output=result.stdout, error=result.stderr)
            ckpt = out_dir / f"bitnet-{model_size}"
            if not ckpt.exists():
                return ToolResult(success=True, output=f"Trained. Checkpoint expected at {ckpt}. Deploy with deploy tool.")
            import os
            if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
                from orbit.tools.deploy import DeployTool
                deploy = DeployTool()
                d = deploy.run(checkpoint_path=str(ckpt), repo_id=repo_id)
                if d.success:
                    return ToolResult(success=True, output=f"Deployed to https://huggingface.co/{repo_id}. Use call_deployed_model with repo_id={repo_id}")
            return ToolResult(success=True, output=f"Trained. Checkpoint: {ckpt}. Deploy manually or use deploy tool.")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class CallDeployedModelTool(Tool):
    """Call a deployed model (HF inference API or local path) with a prompt."""

    name = "call_deployed_model"
    description = "Call a deployed/slave model. args: repo_id_or_path (str), prompt (str). Uses HF inference or local."

    def run(
        self,
        repo_id_or_path: str,
        prompt: str,
        max_tokens: int = 256,
    ) -> ToolResult:
        try:
            path = Path(repo_id_or_path)
            if path.exists():
                from orbit.core.llm import get_llm
                llm = get_llm(str(path))
                out = llm.generate(prompt, max_tokens=max_tokens)
                return ToolResult(success=True, output=out)
            import os
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                from huggingface_hub import InferenceClient
                client = InferenceClient(token=token)
                out = client.text_generation(prompt, model=repo_id_or_path, max_new_tokens=max_tokens)
                return ToolResult(success=True, output=out)
            return ToolResult(success=False, output="", error="HF_TOKEN required for remote models, or use local path")
        except ImportError:
            return ToolResult(success=False, output="", error="huggingface_hub required")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
