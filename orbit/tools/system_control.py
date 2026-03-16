"""System control — mouse, keyboard, full PC control."""

from __future__ import annotations

import subprocess
from orbit.tools.base import Tool, ToolResult


class MouseMoveTool(Tool):
    """Move mouse to screen position."""

    name = "mouse_move"
    description = "Move mouse to x,y. args: x (int), y (int)"

    def run(self, x: int, y: int) -> ToolResult:
        try:
            import pyautogui
            pyautogui.moveTo(x, y, duration=0.2)
            return ToolResult(success=True, output=f"Moved to ({x}, {y})")
        except ImportError:
            return ToolResult(success=False, output="", error="pyautogui required: pip install pyautogui")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class MouseClickTool(Tool):
    """Click at position or current location."""

    name = "mouse_click"
    description = "Click. args: x (int, optional), y (int, optional), button (left|right|middle, default left)"

    def run(self, x: int | None = None, y: int | None = None, button: str = "left") -> ToolResult:
        try:
            import pyautogui
            btn = {"left": "left", "right": "right", "middle": "middle"}.get(button, "left")
            if x is not None and y is not None:
                pyautogui.click(x, y, button=btn)
                return ToolResult(success=True, output=f"Clicked {btn} at ({x}, {y})")
            pyautogui.click(button=btn)
            return ToolResult(success=True, output=f"Clicked {btn} at current position")
        except ImportError:
            return ToolResult(success=False, output="", error="pyautogui required: pip install pyautogui")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class KeyboardTypeTool(Tool):
    """Type text via keyboard."""

    name = "keyboard_type"
    description = "Type text. args: text (str)"

    def run(self, text: str) -> ToolResult:
        try:
            import pyautogui
            pyautogui.write(text, interval=0.05)
            return ToolResult(success=True, output=f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}")
        except ImportError:
            return ToolResult(success=False, output="", error="pyautogui required: pip install pyautogui")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class KeyboardPressTool(Tool):
    """Press key or hotkey."""

    name = "keyboard_press"
    description = "Press key. args: key (str, e.g. enter, tab, ctrl+c)"

    def run(self, key: str) -> ToolResult:
        try:
            import pyautogui
            pyautogui.press(key.lower())
            return ToolResult(success=True, output=f"Pressed: {key}")
        except ImportError:
            return ToolResult(success=False, output="", error="pyautogui required: pip install pyautogui")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class RunCommandTool(Tool):
    """Run shell command — full system access."""

    name = "run_command"
    description = "Run shell command. args: command (str), timeout (int, default 60)"

    def run(self, command: str, timeout: int = 60) -> ToolResult:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = result.stdout or "(no stdout)"
            err = result.stderr or ""
            if result.returncode != 0:
                return ToolResult(success=False, output=out, error=f"exit {result.returncode}: {err}")
            return ToolResult(success=True, output=out, error=err or None)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Command timed out")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ScreenSizeTool(Tool):
    """Get screen size for coordinate reference."""

    name = "screen_size"
    description = "Get screen width and height. No args."

    def run(self) -> ToolResult:
        try:
            import pyautogui
            w, h = pyautogui.size()
            return ToolResult(success=True, output=f"Screen: {w}x{h}")
        except ImportError:
            return ToolResult(success=False, output="", error="pyautogui required: pip install pyautogui")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
