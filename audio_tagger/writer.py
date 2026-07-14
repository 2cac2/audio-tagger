"""Tag writer.

Writes multi-valued tags correctly (null-separated) via mutagen, and always
supports a dry-run so nothing touches your files until you say so. Also
snapshots the original tags to a JSON sidecar for rollback.

Requires: mutagen  (pip install mutagen)
"""

from __future__ import annotations

import json
import os

# Map our internal field names -> the tag keys mutagen's EasyMP3/EasyMP4 use.
_EASY_KEYS = {
    "title": "title",
    "artist": "artist",
    "albumartist": "albumartist",
    "album": "album",
    "composer": "composer",
    "lyricist": "lyricist",
    "year": "date",
    "grouping": "grouping",
    "musicbrainz_trackid": "musicbrainz_trackid",
    "musicbrainz_albumid": "musicbrainz_albumid",
}


def _backup_path(audio_path: str) -> str:
    return audio_path + ".tags.bak.json"


def write_tags(audio_path: str, tags: dict, dry_run: bool = True) -> dict:
    """Apply tags to a file. Returns a diff summary. Multi-valued lists are
    preserved as real multiple values so players show true collaborations."""
    from mutagen import File as MutagenFile

    audio = MutagenFile(audio_path, easy=True)
    if audio is None:
        return {"path": audio_path, "error": "unreadable/unsupported format"}

    before = {k: list(v) for k, v in audio.items()}
    planned: dict[str, list[str]] = {}
    for field, value in tags.items():
        key = _EASY_KEYS.get(field)
        if not key:
            continue
        planned[key] = [str(x) for x in value] if isinstance(value, (list, tuple)) else [str(value)]

    # The compilation flag ('comp') isn't an Easy key; handled below per-format.
    comp = tags.get("comp")

    if dry_run:
        return {"path": audio_path, "dry_run": True, "before": before, "after": planned,
                "comp": comp}

    if not os.path.exists(_backup_path(audio_path)):
        with open(_backup_path(audio_path), "w", encoding="utf-8") as fh:
            json.dump(before, fh, ensure_ascii=False, indent=2)

    for key, value in planned.items():
        audio[key] = value
    audio.save()
    _write_comp_flag(audio_path, comp)
    return {"path": audio_path, "dry_run": False, "written": planned, "comp": comp}


def _write_comp_flag(audio_path: str, comp) -> None:
    """Set the compilation flag using the format-native frame."""
    if comp is None:
        return
    ext = os.path.splitext(audio_path)[1].lower()
    try:
        if ext in (".mp3",):
            from mutagen.id3 import ID3, TCMP
            id3 = ID3(audio_path)
            id3.add(TCMP(encoding=3, text=[str(int(comp))]))
            id3.save()
        elif ext in (".m4a", ".mp4", ".aac"):
            from mutagen.mp4 import MP4
            mp4 = MP4(audio_path)
            mp4["cpil"] = bool(int(comp))
            mp4.save()
        # FLAC/Vorbis: beets/players read a COMPILATION comment.
        elif ext in (".flac", ".ogg", ".opus"):
            from mutagen import File as MF
            f = MF(audio_path)
            f["compilation"] = [str(int(comp))]
            f.save()
    except Exception:
        pass  # never let the flag break the main write


def rollback(audio_path: str) -> bool:
    """Restore tags from the sidecar backup if present."""
    bak = _backup_path(audio_path)
    if not os.path.exists(bak):
        return False
    from mutagen import File as MutagenFile
    with open(bak, encoding="utf-8") as fh:
        before = json.load(fh)
    audio = MutagenFile(audio_path, easy=True)
    if audio is None:
        return False
    audio.delete()
    for k, v in before.items():
        audio[k] = v
    audio.save()
    return True
