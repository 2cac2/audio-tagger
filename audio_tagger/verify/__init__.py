"""Audio-verification layer — a 25s clip as tie-breaking evidence.

:func:`extract_clip` pulls a mono 16 kHz WAV from the middle of a track (ffmpeg,
degrading to ``None`` when unavailable). :func:`verify_clip` sends it to a local
omni LLM and parses a strict-JSON :class:`VerifyReport` of what the audio sounds
like. :func:`consistency` scores how well a structured candidate agrees with
that report in ``[-1, 1]``, which the resolver folds in as a small ``±`` nudge
(and, on agreement, as corroboration for ``audio_verified``). The audio is only
ever evidence — it never picks a candidate on its own.

Nothing here imports a third-party library at load time; ``openai`` is pulled in
lazily and ``ffmpeg`` is a subprocess, so the package imports with only the
standard library present and the layer is fully driveable offline with a fake
``chat`` object.
"""

from __future__ import annotations

from .audio_clip import extract_clip
from .omni import VerifyReport, consistency, verify_clip

__all__ = ["extract_clip", "verify_clip", "consistency", "VerifyReport"]
