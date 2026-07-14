"""Spotify source.

Strong coverage of modern Bollywood / Punjabi / indipop, and it exposes
multiple track artists distinctly (good for collaborations). No composer or
lyricist though, so it contributes release identity + the singer list.

Requires a client id/secret (Client-Credentials flow, no user login):
  export SPOTIFY_CLIENT_ID=...  SPOTIFY_CLIENT_SECRET=...
If unset, the adapter disables itself gracefully (returns []).

Requires: requests  (pip install requests)
"""

from __future__ import annotations

import os
import time

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from .base import query_string

_ALBUM_TYPE = {
    "album": ReleaseType.ALBUM,
    "single": ReleaseType.SINGLE,
    "compilation": ReleaseType.COMPILATION,
}


class SpotifySource:
    name = "spotify"

    def __init__(self, client_id: str | None = None, client_secret: str | None = None,
                 market: str = "IN", limit: int = 5):
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self.market = market
        self.limit = limit
        self._token = None
        self._token_exp = 0.0
        self.enabled = bool(self.client_id and self.client_secret)

    def _bearer(self) -> str | None:
        if not self.enabled:
            return None
        if self._token and time.monotonic() < self._token_exp - 30:
            return self._token
        import requests
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret), timeout=8,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_exp = time.monotonic() + payload.get("expires_in", 3600)
        return self._token

    def search(self, track: InputTrack) -> list[Candidate]:
        token = self._bearer()
        if not token:
            return []
        import requests
        q = query_string(track)
        if not q:
            return []
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": q, "type": "track", "market": self.market, "limit": self.limit},
            timeout=8,
        )
        resp.raise_for_status()
        out: list[Candidate] = []
        for item in resp.json().get("tracks", {}).get("items", []):
            album = item.get("album", {})
            rtype = _ALBUM_TYPE.get(album.get("album_type", ""), ReleaseType.ALBUM)
            credits = [ArtistCredit(a["name"], "singer") for a in item.get("artists", [])]
            out.append(Candidate(
                source=self.name,
                title=item.get("name", ""),
                release_title=album.get("name", ""),
                release_type=rtype,
                credits=credits,
                release_date=album.get("release_date"),
                year=_year(album.get("release_date")),
                is_various_artists=any(
                    a.get("name", "").lower() == "various artists"
                    for a in album.get("artists", [])
                ),
                match_score=0.72,
                raw=item,
            ))
        return out


def _year(iso: str | None) -> int | None:
    if iso and len(iso) >= 4 and iso[:4].isdigit():
        return int(iso[:4])
    return None
