"""Extract a short audio clip for the omni verifier, via ffmpeg.

The audio-verification layer needs a small, uniform slice of the track to hand
to the local omni model: ~25 seconds, mono, 16 kHz, taken from the *middle* of
the song (intros/outros are often instrumental and unrepresentative of the
vocal). :func:`extract_clip` shells out to ``ffmpeg`` and streams the WAV back
over a stdout pipe — nothing is written to disk.

Graceful degradation is the whole contract: if ``ffmpeg`` (or ``ffprobe``) is
not installed, or the encode fails for any reason, the function returns ``None``
and **never raises**. Callers treat ``None`` as "no audio evidence available".

Only the standard library is used here, so the package still imports with
nothing installed.
"""

from __future__ import annotations

import shutil
import subprocess

# When we cannot probe the duration, seek here (past a typical intro) and hope
# for the best — a short track will just yield a shorter clip.
_DEFAULT_FALLBACK_OFFSET = 45.0

# WAV files start with a 44-byte RIFF/fmt header; anything at or below that is a
# header-only (i.e. empty) encode, which we treat as failure.
_MIN_WAV_BYTES = 44


def _probe_duration(path: str) -> float | None:
    """Return the track duration in seconds via ``ffprobe``, or ``None``.

    Missing ``ffprobe``, a non-zero exit, an unparseable value or any OS error
    all degrade to ``None`` — the caller falls back to a fixed offset.
    """
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [
                exe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or b"").decode("utf-8", "ignore").strip()
    try:
        dur = float(text)
    except ValueError:
        return None
    return dur if dur > 0 else None


def _resolve_start(path: str, offset, seconds: float) -> float:
    """Compute the clip start (seconds) from the ``offset`` argument.

    An explicit numeric ``offset`` (int/float, or a numeric string) is used as a
    literal start time. ``"middle"`` (the default) probes the duration and
    centers the clip; if the duration is unknown it falls back to
    :data:`_DEFAULT_FALLBACK_OFFSET`.
    """
    # An explicit numeric offset wins outright (bools are not numbers here).
    if isinstance(offset, bool):
        pass
    elif isinstance(offset, (int, float)):
        return max(0.0, float(offset))
    elif isinstance(offset, str):
        token = offset.strip().lower()
        if token not in ("middle", "mid", "center", "centre", ""):
            try:
                return max(0.0, float(token))
            except ValueError:
                pass  # not a number — treat as "middle"

    # "middle": center the clip on the track when we know how long it is.
    dur = _probe_duration(path)
    if dur and dur > 0:
        return max(0.0, dur / 2.0 - float(seconds) / 2.0)
    return _DEFAULT_FALLBACK_OFFSET


def extract_clip(
    path: str,
    offset="middle",
    seconds: int = 25,
    sr: int = 16000,
) -> bytes | None:
    """Return ~``seconds`` of mono ``sr``-Hz WAV from ``path``, or ``None``.

    ``offset`` is ``"middle"`` (probe + center) by default, or an explicit start
    time in seconds. The clip is produced by ``ffmpeg`` and returned as raw WAV
    bytes over a stdout pipe. Returns ``None`` — never raising — if ``ffmpeg`` is
    absent, the path is empty, the encode fails, or the output is header-only.
    """
    if not path:
        return None
    exe = shutil.which("ffmpeg")
    if not exe:
        return None

    start = _resolve_start(path, offset, seconds)
    cmd = [
        exe, "-nostdin", "-v", "error",
        "-ss", f"{start:.3f}",     # input seek (fast) to the clip start
        "-t", str(int(seconds)),
        "-i", str(path),
        "-ac", "1",                # downmix to mono
        "-ar", str(int(sr)),       # resample
        "-f", "wav",
        "-",                       # stream to stdout
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(30, int(seconds) * 3),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None

    data = proc.stdout or b""
    return data if len(data) > _MIN_WAV_BYTES else None
