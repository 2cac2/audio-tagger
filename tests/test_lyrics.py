"""Lyrics resolution — offline, urllib monkeypatched to canned LRCLIB payloads.

Covers the live-run finding that LRCLIB's exact ``/api/get`` misses Hindi songs
and the fuzzy ``/api/search`` recovers them, plus YouTube-description lyrics.
"""

import io
import json

from audio_tagger.models import ArtistCredit, Candidate, ReleaseType
import audio_tagger.lyrics as lyrics


class _FakeHTTP:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return self._body


def _cand(title="Kesariya", singer="Arijit Singh"):
    return Candidate(source="t", title=title, release_title="Brahmastra",
                     release_type=ReleaseType.SOUNDTRACK, film="Brahmastra",
                     credits=[ArtistCredit(singer, "singer")])


def test_fetch_lyrics_falls_back_to_search(monkeypatch):
    """/api/get returns nothing (as it does live for Kesariya); /api/search
    returns a synced hit, and fetch_lyrics recovers it."""
    search_payload = [
        {"trackName": "Kesariya", "artistName": "Pritam feat. Arijit Singh",
         "duration": 268, "syncedLyrics": "[00:12.00]kesariya tera ishq hai piya",
         "plainLyrics": "kesariya tera ishq hai piya"},
        {"trackName": "Totally Different Song", "artistName": "Someone",
         "duration": 200, "syncedLyrics": "[00:01.00]nope"},
    ]

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if "/api/get" in url:
            return _FakeHTTP({})                 # exact endpoint misses
        if "/api/search" in url:
            return _FakeHTTP(search_payload)     # fuzzy endpoint hits
        raise AssertionError(url)

    monkeypatch.setattr(lyrics.urllib.request, "urlopen", fake_urlopen)

    got = lyrics.fetch_lyrics(_cand(), duration_sec=268.0)
    assert got is not None
    assert got["source"] == "lrclib-search"
    assert "kesariya" in (got.get("synced") or "").lower()
    # the best hit (right title + duration) is chosen over the decoy
    assert "nope" not in (got.get("synced") or "")


def test_fetch_lyrics_uses_youtube_description(monkeypatch):
    """When LRCLIB has nothing but the chosen candidate is a YouTube label
    upload carrying lyrics in raw, those are used as a plain-lyrics fallback."""
    def fake_urlopen(req, timeout=None):
        return _FakeHTTP({}) if "/api/get" in req.full_url else _FakeHTTP([])

    monkeypatch.setattr(lyrics.urllib.request, "urlopen", fake_urlopen)

    c = _cand()
    c.raw = {"lyrics": "kesariya tera ishq hai piya\ndil ye mera..."}
    got = lyrics.fetch_lyrics(c)
    assert got is not None and got["source"] == "youtube"
    assert "kesariya" in got["plain"].lower()
    synced, plain = lyrics.choose_writable(got)
    assert synced is None and plain and "kesariya" in plain.lower()
