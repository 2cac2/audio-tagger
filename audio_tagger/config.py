"""Harness configuration.

A single ``HarnessConfig`` object carries every knob the pipeline needs: the
local LLM endpoint, MCP tool servers, per-source enable/URL/key settings, the
confidence gate thresholds, audio-verification and lyrics options. It loads from
a YAML file (``pyyaml`` is imported lazily *inside* ``load`` so the package
still imports with nothing installed), falling back to ``./config.yaml`` and
then to the built-in defaults, and layers environment variables on top for
secrets.

Consumers should read defensively (``getattr(cfg.llm, "model", ...)``,
``cfg.sources.get("jiosaavn")``) so a partial config never crashes a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3.5-Omni-7B"
    api_key: str | None = None
    max_agent_turns: int = 6
    timeout_sec: int = 60
    supports_audio_input: bool = True


@dataclass
class McpConfig:
    # name -> {transport, url|command, args}
    servers: dict = field(default_factory=dict)


@dataclass
class SourceConfig:
    enabled: bool = True
    base_url: str | None = None
    api_key: str | None = None
    extra: dict = field(default_factory=dict)   # e.g. spotify client_secret


@dataclass
class ThresholdConfig:
    auto: float = 0.80
    agreement_floor: float = 0.5
    min_independent_sources: int = 2
    agent_confidence_cap: float = 0.60


@dataclass
class VerifyConfig:
    offset: str = "middle"
    clip_seconds: int = 25
    sample_rate: int = 16000


@dataclass
class LyricsConfig:
    enabled: bool = True
    write_plain_to_tag: bool = True


class SourceRegistry(dict):
    """A ``dict[str, SourceConfig]`` that yields an enabled default for unknown
    source names, so ``cfg.sources["whatever"]`` / ``.get(...)`` never KeyErrors.
    """

    def __missing__(self, key: str) -> SourceConfig:
        sc = SourceConfig()
        self[key] = sc
        return sc

    def get(self, key, default=None):  # type: ignore[override]
        if key in self:
            return dict.__getitem__(self, key)
        return default if default is not None else self[key]


# Sources known to the pipeline and their non-default baselines.
_DEFAULT_SOURCES: dict[str, SourceConfig] = {
    "musicbrainz": SourceConfig(enabled=True),
    "acoustid": SourceConfig(enabled=True),
    "jiosaavn": SourceConfig(enabled=True, base_url="http://localhost:3500"),
    "deezer": SourceConfig(enabled=True),
    "youtube": SourceConfig(enabled=True),   # label/Topic descriptions -> LLM parse
    "itunes": SourceConfig(enabled=True),
    "spotify": SourceConfig(enabled=False),   # 2024-26 API lockdown — off by default
    "discogs": SourceConfig(enabled=False),   # needs a token — off by default
    "lrclib": SourceConfig(enabled=True),
    "genius": SourceConfig(enabled=False),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(obj, data: dict | None):
    """Overlay dict values onto a dataclass instance, ignoring unknown keys."""
    if not isinstance(data, dict):
        return obj
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key in known and value is not None:
            setattr(obj, key, value)
    return obj


def _default_sources() -> SourceRegistry:
    reg = SourceRegistry()
    for name, sc in _DEFAULT_SOURCES.items():
        reg[name] = SourceConfig(
            enabled=sc.enabled, base_url=sc.base_url, api_key=sc.api_key,
            extra=dict(sc.extra),
        )
    return reg


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    sources: SourceRegistry = field(default_factory=_default_sources)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    lyrics: LyricsConfig = field(default_factory=LyricsConfig)
    path: str | None = None

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: str | None = None) -> "HarnessConfig":
        """Build a config from YAML + env, falling back to built-in defaults.

        Search order: the explicit ``path`` if given, else ``./config.yaml`` if
        it exists, else nothing (pure defaults). ``pyyaml`` is imported here,
        lazily; if it is missing or the file is unreadable we degrade to
        defaults rather than raise.
        """
        data: dict = {}
        used_path: str | None = None

        candidate = path
        if candidate is None and os.path.exists("config.yaml"):
            candidate = "config.yaml"

        if candidate and os.path.exists(candidate):
            try:
                import yaml  # lazy — keep stdlib-only import of the package
                with open(candidate, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict):
                    data = loaded
                    used_path = candidate
            except Exception:
                data = {}

        cfg = cls.from_dict(data)
        cfg.path = used_path
        cfg._apply_env(os.environ)
        return cfg

    @classmethod
    def from_dict(cls, data: dict | None) -> "HarnessConfig":
        data = data or {}
        cfg = cls()
        _apply(cfg.llm, data.get("llm"))
        _apply(cfg.thresholds, data.get("thresholds"))
        _apply(cfg.verify, data.get("verify"))
        _apply(cfg.lyrics, data.get("lyrics"))

        mcp = data.get("mcp") or {}
        servers = mcp.get("servers") if isinstance(mcp, dict) else None
        if isinstance(servers, dict):
            cfg.mcp.servers = dict(servers)

        src_data = data.get("sources")
        if isinstance(src_data, dict):
            for name, sconf in src_data.items():
                existing = cfg.sources.get(name)
                if isinstance(sconf, dict):
                    _apply(existing, sconf)
                    # stash any unknown keys (client_secret, etc.) into extra
                    known = {f.name for f in fields(existing)}
                    for k, v in sconf.items():
                        if k not in known:
                            existing.extra[k] = v
                cfg.sources[name] = existing
        return cfg

    # -- environment overrides ---------------------------------------------
    def _apply_env(self, env) -> None:
        """Layer secrets from the environment over file/default values."""
        llm_key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
        if llm_key:
            self.llm.api_key = llm_key

        if env.get("ACOUSTID_KEY"):
            self.sources["acoustid"].api_key = env["ACOUSTID_KEY"]
        if env.get("DISCOGS_TOKEN"):
            self.sources["discogs"].api_key = env["DISCOGS_TOKEN"]
        if env.get("GENIUS_TOKEN"):
            self.sources["genius"].api_key = env["GENIUS_TOKEN"]

        spotify_id = env.get("SPOTIFY_CLIENT_ID")
        spotify_secret = env.get("SPOTIFY_CLIENT_SECRET")
        if spotify_id:
            self.sources["spotify"].api_key = spotify_id
        if spotify_secret:
            self.sources["spotify"].extra["client_secret"] = spotify_secret
