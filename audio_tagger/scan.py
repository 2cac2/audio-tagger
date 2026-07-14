"""Read local files into InputTracks, including AcoustID fingerprints when
pyacoustid + fpcalc are available (fingerprint is optional — the harness
degrades to text search without it).
"""

from __future__ import annotations

import os

from .models import InputTrack

AUDIO_EXTS = {".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wav", ".wma"}


def _existing_tags(path: str) -> dict:
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(path, easy=True)
        if audio is None:
            return {}
        length = getattr(getattr(audio, "info", None), "length", None)
        return {
            "title": (audio.get("title") or [None])[0],
            "artist": (audio.get("artist") or [None])[0],
            "album": (audio.get("album") or [None])[0],
            "duration": length,
        }
    except Exception:
        return {}


def _fingerprint(path: str):
    try:
        import acoustid
        duration, fp = acoustid.fingerprint_file(path)
        return duration, fp.decode() if isinstance(fp, bytes) else fp
    except Exception:
        return None, None


def scan_path(root: str, fingerprint: bool = True) -> list[InputTrack]:
    tracks: list[InputTrack] = []
    paths = [root] if os.path.isfile(root) else _walk(root)
    for p in paths:
        tags = _existing_tags(p)
        dur = tags.get("duration")
        fp = None
        if fingerprint:
            fdur, fp = _fingerprint(p)
            dur = dur or fdur
        tracks.append(InputTrack(
            path=p,
            existing_title=tags.get("title") or _title_from_name(p),
            existing_artist=tags.get("artist"),
            existing_album=tags.get("album"),
            duration_sec=dur,
            acoustid_fingerprint=fp,
        ))
    return tracks


def _walk(root: str):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                yield os.path.join(dirpath, f)


def _title_from_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # Strip common junk: leading track numbers, site tags in brackets.
    import re
    stem = re.sub(r"^\s*\d{1,3}[\s._-]+", "", stem)
    stem = re.sub(r"[\[(](www\.|.*?\.(com|in|net)).*?[\])]", "", stem, flags=re.I)
    return stem.replace("_", " ").strip()
