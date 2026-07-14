"""MCP tool bridge — exposes configured MCP servers as OpenAI function tools.

The judge can reach for external tools (a SearXNG/DuckDuckGo web-search MCP, a
Firecrawl fetch MCP) when the structured sources conflict or lack roles.
:class:`McpToolbox` connects to the servers named in ``cfg.mcp.servers``, lists
their tools, converts each tool's JSON-Schema into the OpenAI function-calling
shape, and dispatches ``call(name, args)`` back to the owning server.

Everything degrades gracefully: the ``mcp`` SDK is imported lazily, and *any*
connection or protocol error collapses the toolbox to zero tools rather than
raising — a judge with no tools simply reasons from the structured candidates.

The MCP SDK is async; we bridge to the harness's synchronous world by running
each operation on a private event loop. Connections are opened per-operation
(list, then per call) which keeps the async session lifetime inside one
coroutine and avoids leaking a loop across the sync boundary.
"""

from __future__ import annotations

_MAX_RESULT_CHARS = 4000


def _truncate(text: str, limit: int = _MAX_RESULT_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _mcp_tool_to_openai(tool) -> dict:
    """Convert one MCP tool descriptor into an OpenAI function-tool schema."""
    schema = getattr(tool, "inputSchema", None)
    if not isinstance(schema, dict) or not schema:
        schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", "") or "",
            "parameters": schema,
        },
    }


def _result_to_text(result) -> str:
    """Flatten an MCP ``CallToolResult`` into plain text."""
    parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
        else:
            data = getattr(item, "data", None)
            if data:
                parts.append("[binary content omitted]")
    if not parts:
        # Some servers put a structured payload on .structuredContent.
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            import json
            try:
                return json.dumps(structured)
            except Exception:
                return str(structured)
    return "\n".join(parts)


class McpToolbox:
    """Connects configured MCP servers and surfaces them as OpenAI tools."""

    def __init__(self, mcp_cfg=None):
        servers = getattr(mcp_cfg, "servers", None)
        if servers is None and isinstance(mcp_cfg, dict):
            servers = mcp_cfg.get("servers")
        self._servers: dict = dict(servers or {})
        self._tools: list[dict] = []
        self._tool_server: dict[str, str] = {}
        self._connected = False

    # -- context-manager sugar (no persistent state to release) -------------
    def __enter__(self) -> "McpToolbox":
        self.connect()
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self._connected = False

    # -- sync/async bridge --------------------------------------------------
    @staticmethod
    def _run(coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            asyncio.set_event_loop(None)
            loop.close()

    def _open(self, sconf):
        """Async context manager yielding an initialized MCP ``ClientSession``."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _cm():
            from mcp import ClientSession
            transport = (sconf.get("transport") or "sse").lower()
            if transport in ("sse", "http", "https"):
                from mcp.client.sse import sse_client
                url = sconf.get("url")
                async with sse_client(url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
            else:  # stdio
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client
                params = StdioServerParameters(
                    command=sconf.get("command"),
                    args=list(sconf.get("args") or []),
                    env=sconf.get("env"),
                )
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session

        return _cm()

    # -- connect / discover -------------------------------------------------
    def connect(self) -> None:
        """List tools from every configured server. Never raises."""
        if self._connected:
            return
        self._connected = True
        self._tools = []
        self._tool_server = {}
        try:
            import mcp  # noqa: F401 — presence check; real imports happen in _open
        except Exception:
            return
        for name, sconf in self._servers.items():
            if not isinstance(sconf, dict):
                continue
            try:
                tools = self._run(self._list_tools(sconf))
            except Exception:
                continue
            for tool in tools or []:
                tname = getattr(tool, "name", "")
                if not tname:
                    continue
                self._tools.append(_mcp_tool_to_openai(tool))
                self._tool_server[tname] = name

    async def _list_tools(self, sconf):
        async with self._open(sconf) as session:
            resp = await session.list_tools()
            return list(getattr(resp, "tools", None) or [])

    async def _call_tool(self, sconf, name, args):
        async with self._open(sconf) as session:
            result = await session.call_tool(name, args or {})
            return _result_to_text(result)

    # -- public API ---------------------------------------------------------
    def openai_tools(self) -> list[dict]:
        """The discovered tools as OpenAI function schemas (possibly empty)."""
        self.connect()
        return list(self._tools)

    def call(self, name: str, args: dict) -> str:
        """Invoke a tool by name; returns its text result (truncated ~4k).

        Unknown tools and any failure return ``""`` so the loop can keep going.
        """
        try:
            self.connect()
            server = self._tool_server.get(name)
            if server is None:
                return ""
            sconf = self._servers.get(server)
            if not isinstance(sconf, dict):
                return ""
            return _truncate(self._run(self._call_tool(sconf, name, args or {})))
        except Exception:
            return ""
