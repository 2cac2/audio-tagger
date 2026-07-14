"""Thin OpenAI-compatible client for a locally-served LLM.

Points the ``openai`` SDK at a configurable ``base_url`` (vLLM, llama.cpp,
Ollama, LM Studio — anything speaking the Chat Completions API) and exposes the
two calls the harness needs: ``chat`` (with optional tool-calling) and ``ping``
(a cheap ``/models`` probe used to decide whether the agent is even reachable).

The ``openai`` package is imported lazily *inside* the methods, so the package
still imports with only the standard library present, and every offline test can
drive :class:`~audio_tagger.agent.loop.AgentJudge` with a fake object that merely
implements ``.chat(messages, tools=None) -> dict``.
"""

from __future__ import annotations

import base64


def audio_content_part(clip_bytes: bytes, fmt: str = "wav") -> dict:
    """Build an OpenAI ``input_audio`` content part from raw clip bytes.

    Omni models (Qwen-Omni etc.) accept audio the same way images are passed —
    a base64 blob inside a content-list part. Used by the audio-verify layer::

        messages=[{"role": "user", "content": [
            {"type": "text", "text": "What language is this?"},
            audio_content_part(clip, "wav"),
        ]}]
    """
    b64 = base64.b64encode(clip_bytes or b"").decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}


def _message_to_dict(msg) -> dict:
    """Normalize an SDK message object (or dict) into a plain dict.

    Guarantees ``role``/``content`` keys and, when present, a ``tool_calls``
    list of ``{id, type, function:{name, arguments}}`` dicts so the agent loop
    can dispatch tools uniformly whether it is talking to the real SDK or a fake.
    """
    if isinstance(msg, dict):
        return msg
    # openai>=1.0 returns pydantic models with model_dump().
    dump = getattr(msg, "model_dump", None)
    if callable(dump):
        try:
            d = dump()
            if isinstance(d, dict):
                d.setdefault("role", "assistant")
                return d
        except Exception:
            pass
    # Manual fallback for anything else duck-typed like a message.
    out: dict = {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", None),
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        norm = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            norm.append({
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(fn, "name", "") if fn else "",
                    "arguments": getattr(fn, "arguments", "") if fn else "",
                },
            })
        out["tool_calls"] = norm
    return out


class LocalLLM:
    """OpenAI-compatible chat client bound to one endpoint from the config."""

    def __init__(self, llm_cfg):
        self.cfg = llm_cfg
        self.base_url = getattr(llm_cfg, "base_url", "http://localhost:8000/v1")
        self.model = getattr(llm_cfg, "model", "local-model")
        self.api_key = getattr(llm_cfg, "api_key", None) or "not-needed"
        self.timeout = getattr(llm_cfg, "timeout_sec", 60)
        self.max_agent_turns = getattr(llm_cfg, "max_agent_turns", 6)
        self.supports_audio_input = getattr(llm_cfg, "supports_audio_input", True)
        self._client_obj = None

    # -- lazy client --------------------------------------------------------
    def _client(self):
        if self._client_obj is None:
            from openai import OpenAI  # lazy: package imports without openai
            self._client_obj = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client_obj

    # -- chat ---------------------------------------------------------------
    def chat(self, messages, tools=None) -> dict:
        """One chat-completion turn; returns the assistant message as a dict.

        When ``tools`` is passed the model may answer with ``tool_calls`` instead
        of content — the returned dict carries them through for the caller to
        dispatch. Temperature is pinned to 0 for reproducible judging.
        """
        client = self._client()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        return _message_to_dict(resp.choices[0].message)

    # -- audio helper (instance-level convenience) --------------------------
    @staticmethod
    def audio_content_part(clip_bytes: bytes, fmt: str = "wav") -> dict:
        return audio_content_part(clip_bytes, fmt)

    # -- health probe -------------------------------------------------------
    def ping(self) -> bool:
        """Cheap reachability check via ``/models``. ``True`` iff it responds."""
        try:
            self._client().models.list()
            return True
        except Exception:
            return False
