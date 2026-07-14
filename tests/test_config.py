"""HarnessConfig tests — YAML load, built-in defaults, env overrides.

The YAML-parsing test skips cleanly when ``pyyaml`` is not installed (the package
imports and runs without it; yaml is lazy-loaded only inside ``load``).
"""

import pytest

from audio_tagger.config import HarnessConfig


# --------------------------------------------------------------------------- #
# Defaults when no file is present.
# --------------------------------------------------------------------------- #
def test_defaults_when_file_absent(tmp_path):
    cfg = HarnessConfig.load(str(tmp_path / "does-not-exist.yaml"))
    assert cfg.thresholds.auto == 0.80
    assert cfg.thresholds.agreement_floor == 0.5
    assert cfg.thresholds.min_independent_sources == 2
    assert cfg.thresholds.agent_confidence_cap == 0.60
    assert cfg.verify.clip_seconds == 25
    assert cfg.verify.sample_rate == 16000
    assert cfg.lyrics.enabled is True
    # Known sources carry their documented default enable state.
    assert cfg.sources["musicbrainz"].enabled is True
    assert cfg.sources["spotify"].enabled is False
    assert cfg.sources["discogs"].enabled is False


def test_unknown_source_defaults_enabled():
    cfg = HarnessConfig()
    assert cfg.sources["some-new-source"].enabled is True


# --------------------------------------------------------------------------- #
# YAML load overrides.
# --------------------------------------------------------------------------- #
def test_yaml_overrides(tmp_path):
    pytest.importorskip("yaml")
    yaml_text = (
        "llm:\n"
        "  base_url: http://example:9000/v1\n"
        "  model: my-omni\n"
        "  max_agent_turns: 4\n"
        "thresholds:\n"
        "  auto: 0.5\n"
        "  min_independent_sources: 3\n"
        "verify:\n"
        "  clip_seconds: 30\n"
        "sources:\n"
        "  jiosaavn:\n"
        "    base_url: http://saavn.local:3500\n"
        "  spotify:\n"
        "    enabled: true\n"
        "    client_secret: shh\n"
        "mcp:\n"
        "  servers:\n"
        "    websearch:\n"
        "      transport: sse\n"
        "      url: http://search:8080/sse\n"
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    cfg = HarnessConfig.load(str(path))
    assert cfg.llm.base_url == "http://example:9000/v1"
    assert cfg.llm.model == "my-omni"
    assert cfg.llm.max_agent_turns == 4
    assert cfg.thresholds.auto == 0.5
    assert cfg.thresholds.min_independent_sources == 3
    assert cfg.verify.clip_seconds == 30
    assert cfg.sources["jiosaavn"].base_url == "http://saavn.local:3500"
    assert cfg.sources["spotify"].enabled is True
    # Unknown keys are stashed into `extra`.
    assert cfg.sources["spotify"].extra.get("client_secret") == "shh"
    assert "websearch" in cfg.mcp.servers


# --------------------------------------------------------------------------- #
# Environment overrides for secrets.
# --------------------------------------------------------------------------- #
def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTID_KEY", "acoustid-secret")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "spotify-secret")
    cfg = HarnessConfig.load(str(tmp_path / "missing.yaml"))
    assert cfg.sources["acoustid"].api_key == "acoustid-secret"
    assert cfg.llm.api_key == "llm-secret"
    assert cfg.sources["spotify"].extra.get("client_secret") == "spotify-secret"
