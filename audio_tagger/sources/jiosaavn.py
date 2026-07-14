"""JioSaavn source — the ground truth for Hindi/Punjabi film music.

Talks to a self-hosted, saavn.dev-style unofficial API (e.g.
``sumitkolhe/jiosaavn-api`` on port 3500). JioSaavn models film music the way
this library needs it: the album *is* the film, the music director is named,
playback singers are listed in order, and each song carries its language — which
the audio-verify step can cross-check. Because the API is unofficial and its
shape drifts, this adapter is deliberately thin and defensive: every field
access tolerates the older ``primaryArtists`` string shape and the newer
``artists.primary`` list shape, and every HTTP failure degrades to ``[]``.

Requires: requests (imported lazily).
"""

from __future__ import annotations

import os

from ..heuristics import looks_like_compilation
from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from .base import query_string, score_match


class JioSaavnSource:
    name = "jiosaavn"

    def __init__(self, base_url: str | None = None, limit: int = 5, timeout: float = 8.0):
        self.base_url = (base_url or os.getenv("JIOSAAVN_BASE_URL")
                         or "http://localhost:3500").rstrip("/")
        self.limit = limit
        self.timeout = timeout
        self.enabled = bool(self.base_url)

    # -- HTTP helpers (all failures -> None/[]) -----------------------------
    def _get(self, requests, path: str, params: dict | None = None):
        try:
            resp = requests.get(f"{self.base_url}{path}", params=params or {},
                                timeout=self.timeout,
                                headers={"User-Agent": "audio-tagger/0.1"})
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def _search_songs(self, requests, q: str) -> list[dict]:
        data = self._get(requests, "/api/search/songs", {"query": q, "limit": self.limit})
        results = ((data or {}).get("data") or {}).get("results")
        return results if isinstance(results, list) else []

    def _song_detail(self, requests, song_id) -> dict | None:
        if not song_id:
            return None
        data = self._get(requests, f"/api/songs/{song_id}")
        payload = (data or {}).get("data")
        if isinstance(payload, list) and payload:
            return payload[0]
        if isinstance(payload, dict):
            return payload
        return None

    def search(self, track: InputTrack) -> list[Candidate]:
        if not self.enabled:
            return []
        try:
            import requests
        except Exception:
            return []
        q = query_string(track)
        if not q:
            return []
        out: list[Candidate] = []
        for song in self._search_songs(requests, q)[: self.limit]:
            if not isinstance(song, dict):
                continue
            detailed = self._song_detail(requests, song.get("id")) or song
            cand = self._to_candidate(track, detailed)
            if cand:
                out.append(cand)
        return out

    # -- mapping ------------------------------------------------------------
    def _to_candidate(self, track: InputTrack, song: dict) -> Candidate | None:
        title = _text(song.get("name") or song.get("title") or song.get("song"))
        if not title:
            return None
        album = _album_name(song)
        singers = [ArtistCredit(n, "singer") for n in _primary_artists(song)]
        composers = [ArtistCredit(n, "composer") for n in _music_directors(song)]
        year = _year(song.get("year"))
        language = _text(song.get("language"))

        cand = Candidate(
            source=self.name,
            title=title,
            release_title=album or title,
            release_type=ReleaseType.SOUNDTRACK,
            credits=singers + composers,
            film=album or None,
            year=year,
            release_date=_release_date(song),
            match_score=score_match(track, title, [c.name for c in singers]),
            raw={
                "language": language,
                "label": _text(song.get("label")),
                "id": song.get("id"),
                "url": song.get("url"),
                "song": song,
            },
        )
        # A JioSaavn "album" that is really a compilation (Superhits, Vol. 3...)
        # should not masquerade as a film soundtrack.
        if looks_like_compilation(cand):
            cand.release_type = ReleaseType.COMPILATION
            cand.film = None
        return cand


# ---------------------------------------------------------------------------
# Field extraction — tolerant of both the old and new API shapes
# ---------------------------------------------------------------------------

def _text(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def _album_name(song: dict) -> str:
    album = song.get("album")
    if isinstance(album, dict):
        return _text(album.get("name") or album.get("title"))
    return _text(album)


def _names_from(value) -> list[str]:
    """Pull display names out of a list-of-dicts or a comma/&-joined string."""
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                n = _text(item.get("name"))
            else:
                n = _text(item)
            if n:
                out.append(n)
    elif isinstance(value, str):
        for chunk in value.replace(" & ", ",").split(","):
            n = chunk.strip()
            if n:
                out.append(n)
    return out


def _primary_artists(song: dict) -> list[str]:
    artists = song.get("artists")
    if isinstance(artists, dict):
        primary = _names_from(artists.get("primary"))
        if primary:
            return primary
    return _names_from(song.get("primaryArtists"))


def _music_directors(song: dict) -> list[str]:
    # Newer payloads nest the music director; older ones expose "music".
    for key in ("music", "music_director", "musicDirector"):
        names = _names_from(song.get(key))
        if names:
            return names
    return []


def _year(v) -> int | None:
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v[:4].isdigit():
        return int(v[:4])
    return None


def _release_date(song: dict) -> str | None:
    d = song.get("releaseDate") or song.get("release_date")
    return d if isinstance(d, str) and d else None
