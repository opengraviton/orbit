"""Orbit tools — code, browser, search, train, deploy, self-improvement, slave models, system control."""

from orbit.tools.base import Tool, ToolResult
from orbit.tools.code_exec import CodeExecTool
from orbit.tools.web_search import WebSearchTool
from orbit.tools.train_model import TrainModelTool
from orbit.tools.deploy import DeployTool
from orbit.tools.generate_training_data import GenerateTrainingDataTool
from orbit.tools.self_train import SelfTrainTool
from orbit.tools.self_optimize import SelfOptimizeTool
from orbit.tools.fetch_url import FetchUrlTool
from orbit.tools.slave_models import TrainDeployTool, CallDeployedModelTool
from orbit.tools.call_model import CallModelTool
from orbit.tools.self_prompt import SelfPromptTool
from orbit.tools.system_control import (
    MouseMoveTool,
    MouseClickTool,
    KeyboardTypeTool,
    KeyboardPressTool,
    RunCommandTool,
    ScreenSizeTool,
)
from orbit.tools.open_url import OpenUrlTool
from orbit.tools.chat_with_ai import ChatWithAITool
from orbit.tools.open_ai_chat import OpenAIChatTool

__all__ = [
    "Tool",
    "ToolResult",
    "CodeExecTool",
    "WebSearchTool",
    "TrainModelTool",
    "DeployTool",
    "GenerateTrainingDataTool",
    "SelfTrainTool",
    "SelfOptimizeTool",
    "FetchUrlTool",
    "TrainDeployTool",
    "CallDeployedModelTool",
    "CallModelTool",
    "SelfPromptTool",
    "ChatWithAITool",
    "OpenAIChatTool",
]

TOOLS = {
    "code_exec": CodeExecTool(),
    "web_search": WebSearchTool(),
    "fetch_url": FetchUrlTool(),
    "train_model": TrainModelTool(),
    "deploy": DeployTool(),
    "generate_training_data": GenerateTrainingDataTool(),
    "self_train": SelfTrainTool(),
    "self_optimize": SelfOptimizeTool(),
    "train_deploy": TrainDeployTool(),
    "call_deployed_model": CallDeployedModelTool(),
    "call_model": CallModelTool(),
    "self_prompt": SelfPromptTool(),
    # System control — mouse, keyboard, shell
    "mouse_move": MouseMoveTool(),
    "mouse_click": MouseClickTool(),
    "keyboard_type": KeyboardTypeTool(),
    "keyboard_press": KeyboardPressTool(),
    "run_command": RunCommandTool(),
    "screen_size": ScreenSizeTool(),
    "open_url": OpenUrlTool(),
    "chat_with_ai": ChatWithAITool(),
    "open_ai_chat": OpenAIChatTool(),
}


def _add_browser_control_if_available():
    try:
        from orbit.tools.browser_control import (
            BrowserOpenTool,
            BrowserNavigateTool,
            BrowserClickTool,
            BrowserTypeTool,
            BrowserScreenshotTool,
            BrowserCloseTool,
        )
        TOOLS["browser_open"] = BrowserOpenTool()
        TOOLS["browser_navigate"] = BrowserNavigateTool()
        TOOLS["browser_click"] = BrowserClickTool()
        TOOLS["browser_type"] = BrowserTypeTool()
        TOOLS["browser_screenshot"] = BrowserScreenshotTool()
        TOOLS["browser_close"] = BrowserCloseTool()
    except ImportError:
        # Fallback: simple webbrowser when Playwright not installed
        from orbit.tools.browser_simple import (
            BrowserSimpleOpenTool,
            BrowserSimpleNavigateTool,
            BrowserSimpleStubTool,
        )
        TOOLS["browser_open"] = BrowserSimpleOpenTool()
        TOOLS["browser_navigate"] = BrowserSimpleNavigateTool()
        stub = BrowserSimpleStubTool()
        TOOLS["browser_click"] = stub
        TOOLS["browser_type"] = stub
        TOOLS["browser_screenshot"] = stub
        TOOLS["browser_close"] = stub


def _add_browser_if_available():
    try:
        from orbit.tools.browser import BrowserTool
        TOOLS["browser"] = BrowserTool()
    except ImportError:
        pass


def _add_browser_forms_if_available():
    try:
        from orbit.tools.browser_forms import BrowserFormsTool
        TOOLS["browser_form"] = BrowserFormsTool()
    except ImportError:
        pass


_add_browser_control_if_available()
_add_browser_if_available()
_add_browser_forms_if_available()
