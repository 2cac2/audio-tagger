"""Tag writer.

Writes multi-valued tags correctly (as true multiple values) via mutagen, and
always supports a dry-run so nothing touches your files until you say so. Also
snapshots the original tags to a JSON sidecar for rollback.

Design notes:
  * mutagen's Easy interfaces (EasyID3 / EasyMP4) raise ``KeyError`` on any key
    they don't know. Before every write we register the credit keys we depend on
    (composer, lyricist, the two MusicBrainz ids) plus a plural ``artists``
    companion (``TXXX:ARTISTS`` on MP3, iTunes freeform on MP4) so players show
    real separate credits instead of an "A & B" mashed string.
  * List tag values are written as real multiple values, and MP3 is saved as
    ID3v2.4 so those multiple values are stored natively.

Requires: mutagen  (pip install mutagen) — imported lazily so the package still
imports with only the standard library present.
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
    # Plural companion holding the individual playback singers as separate
    # values (Picard/MusicBee convention). Populated from "artist" at write time.
    "artists": "artists",
}

# Extensions we know how to attach lyrics to.
_LYRICS_MP3 = (".mp3",)
_LYRICS_MP4 = (".m4a", ".mp4", ".aac", ".m4b")
_LYRICS_VORBIS = (".flac", ".ogg", ".opus")

_EASY_KEYS_REGISTERED = False


def _register_easy_keys() -> None:
    """Teach mutagen's Easy interfaces the keys we write.

    Idempotent and defensive: stock mutagen already registers most of these, so
    we only add the ones that may be missing, and swallow any failure — a
    registration hiccup must never break a write. Run lazily (mutagen imported
    inside) so the package imports with the standard library alone.
    """
    global _EASY_KEYS_REGISTERED
    if _EASY_KEYS_REGISTERED:
        return
    try:
        from mutagen.easyid3 import EasyID3

        valid = EasyID3.valid_keys
        if "composer" not in valid:
            EasyID3.RegisterTextKey("composer", "TCOM")
        if "lyricist" not in valid:
            EasyID3.RegisterTextKey("lyricist", "TEXT")
        if "musicbrainz_trackid" not in valid:
            EasyID3.RegisterTXXXKey("musicbrainz_trackid", "MusicBrainz Release Track Id")
        if "musicbrainz_albumid" not in valid:
            EasyID3.RegisterTXXXKey("musicbrainz_albumid", "MusicBrainz Album Id")
        if "artists" not in valid:
            EasyID3.RegisterTXXXKey("artists", "ARTISTS")
    except Exception:
        pass

    try:
        from mutagen.easymp4 import EasyMP4Tags

        # MP4 has no native lyricist/artists atoms; store as iTunes freeform.
        EasyMP4Tags.RegisterFreeformKey("lyricist", "LYRICIST")
        EasyMP4Tags.RegisterFreeformKey("artists", "ARTISTS")
    except Exception:
        pass

    _EASY_KEYS_REGISTERED = True


def _backup_path(audio_path: str) -> str:
    return audio_path + ".tags.bak.json"


def write_tags(audio_path: str, tags: dict, dry_run: bool = True) -> dict:
    """Apply tags to a file. Returns a diff summary. Multi-valued lists are
    preserved as real multiple values so players show true collaborations."""
    from mutagen import File as MutagenFile

    _register_easy_keys()

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

    # Mirror the singers into the plural ARTISTS companion so players that read
    # it show each singer separately rather than a joined "A & B" string.
    if "artist" in planned:
        planned.setdefault("artists", list(planned["artist"]))

    # The compilation flag ('comp') isn't an Easy key; handled below per-format.
    comp = tags.get("comp")

    if dry_run:
        return {"path": audio_path, "dry_run": True, "before": before, "after": planned,
                "comp": comp}

    if not os.path.exists(_backup_path(audio_path)):
        with open(_backup_path(audio_path), "w", encoding="utf-8") as fh:
            json.dump(before, fh, ensure_ascii=False, indent=2)

    for key, value in planned.items():
        try:
            audio[key] = value  # a list writes as true multiple values
        except Exception:
            # An unsupported key for this particular format is skipped rather
            # than aborting the whole write; the registered keys won't hit this.
            pass
    _save_audio(audio, audio_path)
    _write_comp_flag(audio_path, comp)
    return {"path": audio_path, "dry_run": False, "written": planned, "comp": comp}


def _save_audio(audio, audio_path: str) -> None:
    """Save, forcing ID3v2.4 for MP3 so multi-value frames store natively."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in _LYRICS_MP3:
        try:
            audio.save(v2_version=4)
            return
        except TypeError:
            pass  # non-ID3 saver without the kwarg; fall through
    audio.save()


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


def write_lyrics_tag(audio_path: str, text: str, dry_run: bool = False) -> bool:
    """Embed plain lyrics into the file's native lyrics frame.

    Format mapping: MP3 -> ``USLT`` (ID3, saved v2.4), MP4/M4A -> ``\\xa9lyr``,
    FLAC/OGG/Opus -> Vorbis ``LYRICS`` + ``UNSYNCEDLYRICS``. Returns True when
    lyrics were written (or, in dry-run, when the format is supported and there
    is text to write); False on empty text, unsupported format, or any error.
    Never raises.
    """
    if not text or not str(text).strip():
        return False
    text = str(text)
    ext = os.path.splitext(audio_path)[1].lower()

    supported = ext in _LYRICS_MP3 or ext in _LYRICS_MP4 or ext in _LYRICS_VORBIS
    if dry_run:
        return supported
    if not supported:
        return False

    try:
        if ext in _LYRICS_MP3:
            from mutagen.id3 import ID3, USLT
            from mutagen.id3 import ID3NoHeaderError
            try:
                id3 = ID3(audio_path)
            except ID3NoHeaderError:
                id3 = ID3()
            id3.delall("USLT")
            id3.add(USLT(encoding=3, lang="eng", desc="", text=text))
            id3.save(audio_path, v2_version=4)
            return True
        if ext in _LYRICS_MP4:
            from mutagen.mp4 import MP4
            mp4 = MP4(audio_path)
            mp4["\xa9lyr"] = [text]
            mp4.save()
            return True
        if ext in _LYRICS_VORBIS:
            from mutagen import File as MF
            f = MF(audio_path)
            if f is None:
                return False
            f["LYRICS"] = [text]
            f["UNSYNCEDLYRICS"] = [text]
            f.save()
            return True
    except Exception:
        return False
    return False


def rollback(audio_path: str) -> bool:
    """Restore tags from the sidecar backup if present."""
    bak = _backup_path(audio_path)
    if not os.path.exists(bak):
        return False
    from mutagen import File as MutagenFile

    _register_easy_keys()
    with open(bak, encoding="utf-8") as fh:
        before = json.load(fh)
    audio = MutagenFile(audio_path, easy=True)
    if audio is None:
        return False
    audio.delete()
    for k, v in before.items():
        try:
            audio[k] = v
        except Exception:
            pass
    _save_audio(audio, audio_path)
    return True
