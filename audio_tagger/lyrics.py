"""Lyrics fetching: LRCLIB (synced, free, no auth) -> Genius (fallback).

LRCLIB returns time-synced .lrc when available, which is what you want for
karaoke-style display; Genius covers Hindi/Punjabi lyrics (often in Latin
transliteration and/or Devanagari) but plain-text only and needs a token.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request

from .models import Candidate
from .translit import sim

_LRCLIB = "https://lrclib.net/api/get"
_LRCLIB_SEARCH = "https://lrclib.net/api/search"

# LRC line timing tags, e.g. "[00:12.34]" or "[1:02:33.500]"; and whole-line
# metadata tags like "[ar:...]" / "[ti:...]" / "[length:...]" that shouldn't
# appear in a plain-lyrics tag.
_LRC_TIMESTAMP = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")
_LRC_METADATA = re.compile(r"^\[[a-zA-Z]+:.*\]$")


def fetch_lrclib(title: str, artist: str, album: str | None = None,
                 duration_sec: float | None = None, timeout: float = 8.0) -> dict | None:
    """Return {'synced': str|None, 'plain': str|None} or None."""
    params = {"track_name": title, "artist_name": artist}
    if album:
        params["album_name"] = album
    if duration_sec:
        params["duration"] = int(duration_sec)
    url = f"{_LRCLIB}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "audio-tagger/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except Exception:
        return None
    if not data or (not data.get("syncedLyrics") and not data.get("plainLyrics")):
        return None
    return {"synced": data.get("syncedLyrics"), "plain": data.get("plainLyrics")}


def fetch_lrclib_search(title: str, artist: str | None = None,
                        duration_sec: float | None = None,
                        timeout: float = 8.0) -> dict | None:
    """Fuzzy LRCLIB lookup via ``/api/search`` — the fallback for ``/api/get``.

    LRCLIB's exact ``/api/get`` needs the track name, artist, album and duration
    to line up, which frequently misses Hindi songs (romanization variance,
    unknown album, wrong duration). ``/api/search`` is forgiving: it returns
    ranked hits for a free-text query. We pick the best hit by title similarity
    (transliteration-tolerant) with a tie-break toward duration proximity, and
    prefer hits that actually carry lyrics. Returns ``{'synced','plain'}`` or
    ``None``.
    """
    q = " ".join(p for p in (title, artist) if p).strip()
    if not q:
        return None
    url = f"{_LRCLIB_SEARCH}?{urllib.parse.urlencode({'q': q})}"
    req = urllib.request.Request(url, headers={"User-Agent": "audio-tagger/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None

    def score(hit: dict) -> tuple:
        has_lyrics = bool(hit.get("syncedLyrics") or hit.get("plainLyrics"))
        title_sim = sim(title, hit.get("trackName"))
        dur_pen = 0.0
        if duration_sec and hit.get("duration"):
            try:
                dur_pen = -min(abs(float(hit["duration"]) - float(duration_sec)) / 30.0, 1.0)
            except (TypeError, ValueError):
                dur_pen = 0.0
        # Lyrics-bearing hits first, then closest title, then closest duration.
        return (has_lyrics, round(title_sim, 3), dur_pen)

    best = max(data, key=score)
    if not (best.get("syncedLyrics") or best.get("plainLyrics")):
        return None
    if sim(title, best.get("trackName")) < 0.5:
        return None
    return {"synced": best.get("syncedLyrics"), "plain": best.get("plainLyrics")}


def fetch_youtube_lyrics(candidate: Candidate) -> dict | None:
    """Plain lyrics carried in a YouTube candidate's description.

    Official label uploads often paste the full lyrics into the video
    description; the YouTube source stashes them in ``candidate.raw['lyrics']``.
    Returns ``{'plain': str}`` (no synced timing) or ``None``.
    """
    text = (candidate.raw or {}).get("lyrics")
    if isinstance(text, str) and text.strip():
        return {"plain": text.strip(), "synced": None}
    return None


def fetch_genius(title: str, artist: str, token: str | None = None,
                 timeout: float = 8.0) -> dict | None:
    """Plain-text lyrics via Genius search. Needs GENIUS_TOKEN. Returns
    {'plain': str} pointing at the match; scraping the page body is left to
    the caller's policy (Genius ToS), so we return the URL + snippet only."""
    token = token or os.getenv("GENIUS_TOKEN")
    if not token:
        return None
    import requests
    resp = requests.get(
        "https://api.genius.com/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": f"{title} {artist}"}, timeout=timeout,
    )
    if resp.status_code != 200:
        return None
    hits = resp.json().get("response", {}).get("hits", [])
    if not hits:
        return None
    top = hits[0]["result"]
    return {"plain": None, "url": top.get("url"), "title": top.get("full_title")}


def fetch_lyrics(candidate: Candidate, duration_sec: float | None = None) -> dict | None:
    """Resolve lyrics for a candidate, best source first.

    Order: LRCLIB exact (``/api/get``, synced) -> LRCLIB fuzzy (``/api/search``,
    synced; recovers Hindi songs the exact endpoint misses) -> YouTube
    description lyrics (plain, when the chosen candidate came from a label
    upload) -> Genius (pointer only). Returns the first hit with a ``source`` tag.
    """
    artist = (candidate.singers or [""])[0]
    album = candidate.film or candidate.release_title

    got = fetch_lrclib(candidate.title, artist, album, duration_sec)
    if got:
        got["source"] = "lrclib"
        return got

    got = fetch_lrclib_search(candidate.title, artist, duration_sec)
    if got:
        got["source"] = "lrclib-search"
        return got

    got = fetch_youtube_lyrics(candidate)
    if got:
        got["source"] = "youtube"
        return got

    got = fetch_genius(candidate.title, artist)
    if got:
        got["source"] = "genius"
    return got


def _plain_from_synced(synced: str) -> str:
    """Strip LRC timing/metadata tags to recover plain lyric text."""
    lines: list[str] = []
    for raw in synced.splitlines():
        if _LRC_METADATA.match(raw.strip()):
            continue
        lines.append(_LRC_TIMESTAMP.sub("", raw).strip())
    return "\n".join(lines).strip("\n")


def choose_writable(got: dict | None) -> tuple[str | None, str | None]:
    """Split a fetch result into (synced, plain) for the two write targets.

    ``synced`` (LRC) is meant for a sidecar ``.lrc``; ``plain`` is meant for the
    file's lyrics tag. Either may be ``None``. When only synced lyrics exist we
    derive plain text by stripping the LRC timing tags, so a tag can still be
    written from an LRCLIB-only result. Empty/whitespace values become ``None``.
    """
    if not got:
        return None, None
    synced = got.get("synced")
    synced = synced if isinstance(synced, str) and synced.strip() else None
    plain = got.get("plain")
    plain = plain if isinstance(plain, str) and plain.strip() else None
    if plain is None and synced is not None:
        plain = _plain_from_synced(synced) or None
    return synced, plain
