"""Base tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: str | None = None


class Tool(ABC):
    """Base class for Orbit tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool."""
        pass
