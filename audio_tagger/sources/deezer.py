"""Deezer source — keyless fallback and ISRC bridge.

Deezer's public API needs no key and covers modern Bollywood / Punjabi / indipop
well. Its two contributions here: a ``record_type`` (album / single / EP /
compilation) that helps demote "greatest hits" rips, and an **ISRC** per track —
a stable recording identifier the resolver uses to decide that two sources are
describing the very same recording, independent of transliteration. The ISRC is
stashed in ``raw`` for that cross-source identity check. Since Deezer replaced
Spotify as the demoted commercial API, keyless reliability matters more than
role depth (it exposes no composer/lyricist).

Requires: requests (imported lazily).
"""

from __future__ import annotations

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from .base import query_string, score_match

_SEARCH = "https://api.deezer.com/search"
_ALBUM = "https://api.deezer.com/album"
_TRACK = "https://api.deezer.com/track"

_RECORD_TYPE = {
    "album": ReleaseType.ALBUM,
    "single": ReleaseType.SINGLE,
    "ep": ReleaseType.EP,
    "compile": ReleaseType.COMPILATION,
    "compilation": ReleaseType.COMPILATION,
}


class DeezerSource:
    name = "deezer"

    def __init__(self, limit: int = 5, timeout: float = 8.0):
        self.limit = limit
        self.timeout = timeout
        self.enabled = True   # keyless

    def _get(self, requests, url: str) -> dict | None:
        try:
            resp = requests.get(url, timeout=self.timeout,
                                headers={"User-Agent": "audio-tagger/0.1"})
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def search(self, track: InputTrack) -> list[Candidate]:
        try:
            import requests
        except Exception:
            return []
        q = query_string(track)
        if not q:
            return []
        try:
            resp = requests.get(_SEARCH, params={"q": q, "limit": self.limit},
                                timeout=self.timeout,
                                headers={"User-Agent": "audio-tagger/0.1"})
            resp.raise_for_status()
            hits = resp.json().get("data") or []
        except Exception:
            return []

        out: list[Candidate] = []
        for item in hits[: self.limit]:
            if not isinstance(item, dict):
                continue
            cand = self._to_candidate(requests, track, item)
            if cand:
                out.append(cand)
        return out

    def _to_candidate(self, requests, track: InputTrack, item: dict) -> Candidate | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None
        artist_name = ((item.get("artist") or {}).get("name") or "").strip()
        album = item.get("album") or {}
        album_title = (album.get("title") or "").strip()

        # Album detail: record_type + release date (best-effort).
        rtype = ReleaseType.UNKNOWN
        release_date = None
        album_id = album.get("id")
        if album_id:
            detail = self._get(requests, f"{_ALBUM}/{album_id}")
            if detail:
                rtype = _RECORD_TYPE.get((detail.get("record_type") or "").lower(),
                                         ReleaseType.ALBUM)
                release_date = detail.get("release_date") or None

        # Track detail: ISRC (the cross-source recording bridge).
        isrc = None
        track_id = item.get("id")
        if track_id:
            tdetail = self._get(requests, f"{_TRACK}/{track_id}")
            if tdetail:
                isrc = tdetail.get("isrc") or None
                release_date = release_date or (tdetail.get("release_date") or None)

        credits = [ArtistCredit(artist_name, "singer")] if artist_name else []
        return Candidate(
            source=self.name,
            title=title,
            release_title=album_title,
            release_type=rtype,
            credits=credits,
            year=_year(release_date),
            release_date=release_date,
            match_score=score_match(track, title, [artist_name]),
            raw={"isrc": isrc, "album_id": album_id, "track_id": track_id, "hit": item},
        )


def _year(iso: str | None) -> int | None:
    if iso and len(iso) >= 4 and iso[:4].isdigit():
        return int(iso[:4])
    return None
