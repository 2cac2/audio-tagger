"""MusicBrainz source.

The only source with a structured role model (singer / composer / lyricist
via recording-artist and work relationships) and a real soundtrack
release-group type. Best authority when coverage exists; coverage for older
and regional Indian film music is patchy, which is why it is one of several.

Requires: musicbrainzngs  (pip install musicbrainzngs)
No API key; MusicBrainz asks for a descriptive User-Agent and rate-limits
to ~1 req/s, which musicbrainzngs enforces for us.
"""

from __future__ import annotations

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from .base import query_string

_PRIMARY_TYPE = {
    "album": ReleaseType.ALBUM,
    "single": ReleaseType.SINGLE,
    "ep": ReleaseType.EP,
}
# Secondary types dominate when present (a soundtrack album, a compilation...).
_SECONDARY_TYPE = {
    "soundtrack": ReleaseType.SOUNDTRACK,
    "compilation": ReleaseType.COMPILATION,
}


def _release_type(rg: dict) -> ReleaseType:
    secondaries = [s.lower() for s in rg.get("secondary-type-list", [])]
    for s in secondaries:
        if s in _SECONDARY_TYPE:
            return _SECONDARY_TYPE[s]
    primary = (rg.get("primary-type") or "").lower()
    return _PRIMARY_TYPE.get(primary, ReleaseType.UNKNOWN)


class MusicBrainzSource:
    name = "musicbrainz"

    def __init__(self, app: str = "audio-tagger", version: str = "0.1",
                 contact: str = "https://github.com/2cac2/audio-tagger", limit: int = 5):
        import musicbrainzngs
        musicbrainzngs.set_useragent(app, version, contact)
        self._mb = musicbrainzngs
        self.limit = limit

    def search(self, track: InputTrack) -> list[Candidate]:
        q = query_string(track)
        if not q:
            return []
        res = self._mb.search_recordings(query=q, limit=self.limit)
        out: list[Candidate] = []
        for rec in res.get("recording-list", []):
            out.extend(self._recording_to_candidates(rec))
        return out

    def _recording_to_candidates(self, rec: dict) -> list[Candidate]:
        title = rec.get("title", "")
        score = float(rec.get("ext:score", 0)) / 100.0

        singers = [
            ArtistCredit(name=ac["artist"]["name"], role="singer",
                         mbid=ac["artist"].get("id"))
            for ac in rec.get("artist-credit", []) if isinstance(ac, dict) and "artist" in ac
        ]
        # Work relationships carry composer/lyricist for film songs.
        composers, lyricists = [], []
        for work_rel in rec.get("work-relation-list", []):
            for rel in work_rel.get("work", {}).get("artist-relation-list", []):
                who = rel.get("artist", {}).get("name")
                rtype = rel.get("type", "").lower()
                if not who:
                    continue
                if "composer" in rtype:
                    composers.append(ArtistCredit(who, "composer"))
                elif "lyric" in rtype or "librett" in rtype:
                    lyricists.append(ArtistCredit(who, "lyricist"))

        candidates: list[Candidate] = []
        for rel in rec.get("release-list", []) or [{}]:
            rg = rel.get("release-group", {}) or {}
            rtype = _release_type(rg)
            film = None
            if rtype == ReleaseType.SOUNDTRACK:
                film = rg.get("title") or rel.get("title")
            candidates.append(Candidate(
                source=self.name,
                title=title,
                release_title=rel.get("title") or rg.get("title", ""),
                release_type=rtype,
                credits=singers + composers + lyricists,
                film=film,
                release_date=rel.get("date"),
                is_various_artists=rg.get("artist-credit-phrase", "").lower() == "various artists",
                recording_mbid=rec.get("id"),
                release_mbid=rel.get("id"),
                match_score=score,
                raw=rec,
            ))
        return candidates
