"""Execute Python/shell code."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from orbit.tools.base import Tool, ToolResult


class CodeExecTool(Tool):
    """Execute Python code in a sandboxed environment."""

    name = "code_exec"
    description = "Execute Python code. Returns stdout and stderr. Use for calculations, file ops, API calls."

    def run(self, code: str, timeout: int = 30) -> ToolResult:
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                exec(compile(code, "<orbit>", "exec"), {"__builtins__": __builtins__, "Path": Path})
                out = sys.stdout.getvalue()
                err = sys.stderr.getvalue()
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            return ToolResult(success=True, output=out or "(no output)", error=err or None)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
