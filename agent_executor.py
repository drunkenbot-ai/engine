"""Autonomous Agent Tool Executor for DrunkenBot LLM-IDE.

Note: Core implementation has moved to `inference.core.agent_executor`.
This module re-exports all symbols for backwards compatibility.
"""

from __future__ import annotations

from inference.core.agent_executor import (
    execute_agent_tool,
    execute_python_code,
    execute_web_search,
    parse_tool_calls,
)

__all__ = [
    "parse_tool_calls",
    "execute_python_code",
    "execute_web_search",
    "execute_agent_tool",
]
