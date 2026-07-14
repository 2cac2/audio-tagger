"""iTunes Search API source.

Free, no auth, and — importantly for this library — strong coverage of
Bollywood / Punjabi / indipop soundtracks and singles, with the track's
film album named directly. Weaker on role breakdown (no separate lyricist),
so it mainly contributes the *release* identity and a sanity check on the
artist, while MusicBrainz fills roles.

https://performance-partners.apple.com/search-api
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from .base import query_string

_ENDPOINT = "https://itunes.apple.com/search"


def _split_artists(name: str) -> list[str]:
    # iTunes collapses collaborators into one string: "A & B feat. C".
    seps = [" & ", ", ", " feat. ", " ft. ", " featuring ", " with ", " x "]
    parts = [name]
    for sep in seps:
        parts = [p for chunk in parts for p in chunk.split(sep)]
    return [p.strip() for p in parts if p.strip()]


class ITunesSource:
    name = "itunes"

    def __init__(self, country: str = "IN", limit: int = 5, timeout: float = 8.0):
        # country=IN biases results toward the correct regional catalog.
        self.country = country
        self.limit = limit
        self.timeout = timeout

    def search(self, track: InputTrack) -> list[Candidate]:
        q = query_string(track)
        if not q:
            return []
        params = urllib.parse.urlencode({
            "term": q, "media": "music", "entity": "song",
            "country": self.country, "limit": self.limit,
        })
        req = urllib.request.Request(f"{_ENDPOINT}?{params}",
                                     headers={"User-Agent": "audio-tagger/0.1"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)

        out: list[Candidate] = []
        for r in data.get("results", []):
            coll = r.get("collectionName", "") or ""
            is_soundtrack = r.get("collectionType", "").lower() == "soundtrack" \
                or "soundtrack" in coll.lower() \
                or r.get("primaryGenreName", "").lower() == "soundtrack"
            rtype = ReleaseType.SOUNDTRACK if is_soundtrack else ReleaseType.ALBUM
            credits = [ArtistCredit(n, "singer") for n in _split_artists(r.get("artistName", ""))]
            out.append(Candidate(
                source=self.name,
                title=r.get("trackName", ""),
                release_title=coll,
                release_type=rtype,
                credits=credits,
                film=coll if is_soundtrack else None,
                release_date=r.get("releaseDate"),
                year=_year(r.get("releaseDate")),
                match_score=0.7,   # iTunes gives no score; treat as moderate prior
                raw=r,
            ))
        return out


def _year(iso: str | None) -> int | None:
    if iso and len(iso) >= 4 and iso[:4].isdigit():
        return int(iso[:4])
    return None
