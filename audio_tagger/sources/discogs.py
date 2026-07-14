"""Discogs source.

Best for older / physical Indian releases (HMV, Saregama, T-Series
pressings) that predate the streaming era and are thin in MusicBrainz.
Discogs models credits richly (it can distinguish vocals / music / lyrics),
so it is a useful second opinion on roles for vintage film music.

Requires a personal access token:
  export DISCOGS_TOKEN=...
If unset, the adapter disables itself (returns []).

Requires: requests
"""

from __future__ import annotations

import os

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from .base import query_string

_FORMAT_HINT_COMPILATION = {"compilation", "comp"}


def _role_of(role_text: str) -> str | None:
    r = (role_text or "").lower()
    if any(k in r for k in ("vocal", "singer", "voice")):
        return "singer"
    if any(k in r for k in ("music", "composed", "composer")):
        return "composer"
    if any(k in r for k in ("lyric", "written", "words")):
        return "lyricist"
    if "produce" in r:
        return "producer"
    if any(k in r for k in ("mix", "remix")):
        return "mixer"
    return None


class DiscogsSource:
    name = "discogs"

    def __init__(self, token: str | None = None, limit: int = 5):
        self.token = token or os.getenv("DISCOGS_TOKEN")
        self.limit = limit
        self.enabled = bool(self.token)

    def search(self, track: InputTrack) -> list[Candidate]:
        if not self.enabled:
            return []
        import requests
        q = query_string(track)
        if not q:
            return []
        headers = {"User-Agent": "audio-tagger/0.1", "Authorization": f"Discogs token={self.token}"}
        resp = requests.get(
            "https://api.discogs.com/database/search",
            headers=headers, params={"q": q, "type": "release", "per_page": self.limit},
            timeout=8,
        )
        resp.raise_for_status()
        out: list[Candidate] = []
        for r in resp.json().get("results", []):
            formats = [f.lower() for f in (r.get("format") or [])]
            is_comp = bool(set(formats) & _FORMAT_HINT_COMPILATION)
            genres = [g.lower() for g in (r.get("genre") or []) + (r.get("style") or [])]
            is_soundtrack = "soundtrack" in genres or "stage & screen" in genres
            rtype = (ReleaseType.COMPILATION if is_comp
                     else ReleaseType.SOUNDTRACK if is_soundtrack
                     else ReleaseType.ALBUM)
            out.append(Candidate(
                source=self.name,
                title=track.existing_title or r.get("title", ""),
                release_title=r.get("title", ""),
                release_type=rtype,
                film=r.get("title") if is_soundtrack else None,
                year=int(r["year"]) if str(r.get("year", "")).isdigit() else None,
                match_score=0.55,
                raw=r,
            ))
        return out
