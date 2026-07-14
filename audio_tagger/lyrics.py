"""Lyrics fetching: LRCLIB (synced, free, no auth) -> Genius (fallback).

LRCLIB returns time-synced .lrc when available, which is what you want for
karaoke-style display; Genius covers Hindi/Punjabi lyrics (often in Latin
transliteration and/or Devanagari) but plain-text only and needs a token.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from .models import Candidate

_LRCLIB = "https://lrclib.net/api/get"


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
    """Try synced first (LRCLIB), then Genius metadata as a pointer."""
    artist = (candidate.singers or [""])[0]
    album = candidate.film or candidate.release_title
    got = fetch_lrclib(candidate.title, artist, album, duration_sec)
    if got:
        got["source"] = "lrclib"
        return got
    got = fetch_genius(candidate.title, artist)
    if got:
        got["source"] = "genius"
    return got
