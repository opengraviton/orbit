"""Deploy model to HuggingFace or similar."""

from __future__ import annotations

from pathlib import Path

from orbit.tools.base import Tool, ToolResult


class DeployTool(Tool):
    """Deploy a model to HuggingFace Hub."""

    name = "deploy"
    description = "Upload model checkpoint to HuggingFace Hub. Needs HF_TOKEN env."

    def run(
        self,
        checkpoint_path: str,
        repo_id: str,
        hf_token: str | None = None,
    ) -> ToolResult:
        try:
            import os
            token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                return ToolResult(success=False, output="", error="HF_TOKEN or HUGGING_FACE_HUB_TOKEN required")
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            path = Path(checkpoint_path)
            if not path.exists():
                return ToolResult(success=False, output="", error=f"Checkpoint not found: {checkpoint_path}")
            api.create_repo(repo_id, exist_ok=True)
            api.upload_folder(folder_path=str(path), repo_id=repo_id, repo_type="model")
            return ToolResult(success=True, output=f"Uploaded to https://huggingface.co/{repo_id}")
        except ImportError:
            return ToolResult(success=False, output="", error="huggingface_hub required: pip install huggingface_hub")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
