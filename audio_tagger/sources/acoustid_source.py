"""AcoustID fingerprint source.

The one source that identifies *which recording the audio actually is*, rather
than what the filename claims — so it catches covers, remasters and mislabelled
"greatest hits" rips that string matching cannot. The scan step already computes
a Chromaprint fingerprint (``track.acoustid_fingerprint``); this adapter finally
uses it, POSTing it to the AcoustID lookup service and turning the returned
MusicBrainz recording ids into candidates. Those recording MBIDs are the seed
the resolver hands to MusicBrainz for a full role lookup.

Requires a free AcoustID application API key (https://acoustid.org/new-application),
supplied via config or ``ACOUSTID_KEY``. Without a key — or without a
fingerprint+duration on the track — the adapter self-disables and returns [].

Requires: requests (imported lazily).
"""

from __future__ import annotations

import os

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType

_ENDPOINT = "https://api.acoustid.org/v2/lookup"

_PRIMARY_TYPE = {
    "album": ReleaseType.ALBUM,
    "single": ReleaseType.SINGLE,
    "ep": ReleaseType.EP,
}


def _release_type(rg: dict) -> ReleaseType:
    secondaries = [s.lower() for s in (rg.get("secondarytypes") or [])]
    if "soundtrack" in secondaries:
        return ReleaseType.SOUNDTRACK
    if "compilation" in secondaries:
        return ReleaseType.COMPILATION
    primary = (rg.get("type") or "").lower()
    return _PRIMARY_TYPE.get(primary, ReleaseType.UNKNOWN)


class AcoustIDSource:
    name = "acoustid"

    def __init__(self, api_key: str | None = None, limit: int = 5, timeout: float = 8.0):
        self.api_key = api_key or os.getenv("ACOUSTID_KEY")
        self.limit = limit
        self.timeout = timeout
        self.enabled = bool(self.api_key)

    def search(self, track: InputTrack) -> list[Candidate]:
        # Needs both a fingerprint and a duration to look anything up, and a key.
        if not self.enabled or not self.api_key:
            return []
        if not track.acoustid_fingerprint or not track.duration_sec:
            return []
        try:
            import requests
        except Exception:
            return []
        try:
            resp = requests.post(
                _ENDPOINT,
                data={
                    "client": self.api_key,
                    "duration": int(track.duration_sec),
                    "fingerprint": track.acoustid_fingerprint,
                    "meta": "recordings+releasegroups",
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        if not isinstance(data, dict) or data.get("status") != "ok":
            return []

        out: list[Candidate] = []
        for result in (data.get("results") or [])[: self.limit]:
            # The AcoustID score is a *real* audio-identity confidence, not a
            # string-match prior — carry it straight onto the candidate.
            score = float(result.get("score", 0.0) or 0.0)
            for rec in result.get("recordings") or []:
                out.extend(self._recording_to_candidates(rec, score, result.get("id")))
        return out

    def _recording_to_candidates(self, rec: dict, score: float, acoustid_id) -> list[Candidate]:
        rec_id = rec.get("id")
        if not rec_id:
            return []
        title = rec.get("title", "") or ""
        singers = [
            ArtistCredit(name=a.get("name", ""), role="singer", mbid=a.get("id"))
            for a in rec.get("artists") or [] if a.get("name")
        ]
        raw = {"via": "fingerprint", "acoustid": acoustid_id, "score": score, "recording": rec}

        groups = rec.get("releasegroups") or [{}]
        out: list[Candidate] = []
        for rg in groups:
            rtype = _release_type(rg)
            film = rg.get("title") if rtype == ReleaseType.SOUNDTRACK else None
            out.append(Candidate(
                source=self.name,
                title=title,
                release_title=rg.get("title", "") or "",
                release_type=rtype,
                credits=singers,
                film=film,
                recording_mbid=rec_id,
                release_mbid=None,
                match_score=score,
                raw=raw,
            ))
        return out
