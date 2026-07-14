"""Interactive review TUI for the audio-tagger harness.

The heavy Textual UI lives in :mod:`audio_tagger.tui.app`; the pure-Python
session model and (de)serialization live in :mod:`audio_tagger.tui.state`. Only
``state`` is imported at package load, so ``import audio_tagger.tui`` (and the
whole package) still works with only the standard library present — ``textual``
is pulled in lazily inside :func:`run_tui`.

Entry point::

    from audio_tagger.tui import run_tui
    run_tui("run.json")            # review a --json-out document
    run_tui("~/music", apply=True) # re-resolve a folder and write on accept
"""

from __future__ import annotations

from .state import (
    ReviewItem,
    ReviewSession,
    load_session,
    save_session,
)

__all__ = [
    "run_tui",
    "load_session",
    "save_session",
    "ReviewSession",
    "ReviewItem",
]


def _build_judge(cfg):
    """Construct an :class:`AgentJudge` for steering, or ``None`` if unavailable.

    Wiring failures (missing ``openai``/``mcp``, bad config) degrade to ``None``
    so the TUI still runs for accept/skip/edit — steering simply reports the
    agent as unavailable.
    """
    try:
        from ..agent.llm import LocalLLM
        from ..agent.loop import AgentJudge
    except Exception:
        return None
    llm = None
    try:
        llm = LocalLLM(getattr(cfg, "llm", None))
    except Exception:
        llm = None
    toolbox = None
    try:
        from ..agent.mcp_client import McpToolbox
        toolbox = McpToolbox(getattr(cfg, "mcp", None))
    except Exception:
        toolbox = None
    try:
        return AgentJudge(cfg, llm=llm, toolbox=toolbox)
    except Exception:
        return None


def run_tui(source, *, config=None, apply=False, judge=None):
    """Load a review session from ``source`` and launch the Textual UI.

    ``source``  : a ``--json-out`` resolutions JSON, a saved ``review.json`` to
                  resume, or a library path/audio file to (re)resolve.
    ``config``  : a ``HarnessConfig``, a path to a config file, or ``None`` to
                  load the default config.
    ``apply``   : write tags for real on accept (default is dry-run).
    ``judge``   : inject a pre-built agent judge (mainly for tests); when
                  ``None`` one is constructed from ``config``.

    ``textual`` is imported here, not at module load, so the package imports
    without it installed.
    """
    from ..config import HarnessConfig

    if config is None or isinstance(config, str):
        cfg = HarnessConfig.load(config)
    else:
        cfg = config

    session = load_session(source, cfg)
    session.dry_run = not apply

    if judge is None:
        judge = _build_judge(cfg)

    from .app import ReviewApp  # imports textual — kept lazy
    app = ReviewApp(session=session, judge=judge, dry_run=not apply)
    app.run()
    return session
