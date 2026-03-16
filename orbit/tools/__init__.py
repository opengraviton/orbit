"""Orbit tools — code, browser, search, train, deploy, self-improvement, slave models."""

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
}


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


_add_browser_if_available()
_add_browser_forms_if_available()
