"""The LLM agent layer — judge/reconciler, local LLM client, and MCP tool bridge.

Nothing here imports a third-party library at module load: ``openai`` and the
``mcp`` SDK are pulled in lazily inside the methods that need them, so the whole
package still imports with only the standard library present and the agent loop
can be exercised offline with a fake ``chat`` object.
"""

from __future__ import annotations

from .llm import LocalLLM
from .loop import AgentJudge
from .mcp_client import McpToolbox

__all__ = ["AgentJudge", "LocalLLM", "McpToolbox"]
