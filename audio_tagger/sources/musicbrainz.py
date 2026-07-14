"""MusicBrainz source.

The only source with a structured role model — singer (recording artist-credit
and vocal/performer relations), composer and lyricist (via the recording's
*work* relationships) — and a real soundtrack release-group type. Best authority
when coverage exists; coverage for older and regional Indian film music is
patchy, which is why it is one of several.

The original adapter searched but never fetched relationships, so composer and
lyricist always came back empty. This version does a two-step lookup: a text
search (and/or AcoustID-seeded recording MBIDs) to find candidate recordings,
then ``get_recording_by_id`` with the relationship includes for the top few, so
roles actually populate. Results are cached in memory and on disk under
``~/.cache/audio-tagger/`` to respect MusicBrainz's ~1 req/s etiquette across
runs.

Requires: musicbrainzngs (imported lazily, in __init__).
No API key; MusicBrainz asks for a descriptive User-Agent and rate-limits to
~1 req/s, which musicbrainzngs enforces for us.
"""

from __future__ import annotations

import json
import os

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from ..translit import sim
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

_DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "audio-tagger")

# Relationships we ask for so composer / lyricist / vocal credits populate.
_INCLUDES = ["artists", "artist-rels", "releases", "release-groups",
             "work-rels", "work-level-rels", "artist-credits"]
# A trimmed set to retry with if a server/version rejects one of the above.
_FALLBACK_INCLUDES = ["artists", "releases", "artist-credits", "work-level-rels"]

# Recordings we reach only via an AcoustID fingerprint seed carry a high prior:
# the audio already matched, so MusicBrainz's own (absent) search score should
# not drag them down.
_SEED_MB_SCORE = 0.85


def _release_type(rg: dict) -> ReleaseType:
    secondaries = [s.lower() for s in (rg.get("secondary-type-list") or [])]
    for s in secondaries:
        if s in _SECONDARY_TYPE:
            return _SECONDARY_TYPE[s]
    primary = (rg.get("primary-type") or "").lower()
    return _PRIMARY_TYPE.get(primary, ReleaseType.UNKNOWN)


def _dedupe(credits: list[ArtistCredit]) -> list[ArtistCredit]:
    out: list[ArtistCredit] = []
    seen: set[str] = set()
    for c in credits:
        key = (c.name or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


class MusicBrainzSource:
    name = "musicbrainz"

    def __init__(self, app: str = "audio-tagger", version: str = "0.1",
                 contact: str = "https://github.com/2cac2/audio-tagger", limit: int = 5,
                 cache_dir: str | None = None):
        import musicbrainzngs
        musicbrainzngs.set_useragent(app, version, contact)
        self._mb = musicbrainzngs
        self.limit = limit
        self.cache_dir = cache_dir or _DEFAULT_CACHE
        self._mem_cache: dict[str, dict] = {}

    # -- public -------------------------------------------------------------
    def search(self, track: InputTrack, seed_mbids: list[str] | None = None) -> list[Candidate]:
        """Resolve a track to candidates.

        ``seed_mbids`` (recording MBIDs from AcoustID) are always looked up in
        full; the text search contributes up to 3 more distinct recordings. Each
        selected recording is fetched with relationship includes so roles land.
        """
        ext_scores: dict[str, float] = {}
        ordered_ids: list[str] = []

        for mbid in seed_mbids or []:
            if mbid and mbid not in ordered_ids:
                ordered_ids.append(mbid)
                ext_scores.setdefault(mbid, _SEED_MB_SCORE)

        q = query_string(track)
        if q:
            try:
                res = self._mb.search_recordings(query=q, limit=self.limit)
            except Exception:
                res = {}
            distinct: list[str] = []
            for rec in res.get("recording-list", []) or []:
                rid = rec.get("id")
                if not rid:
                    continue
                ext_scores.setdefault(rid, float(rec.get("ext:score", 0) or 0) / 100.0)
                if rid not in distinct:
                    distinct.append(rid)
            for rid in distinct[:3]:
                if rid not in ordered_ids:
                    ordered_ids.append(rid)

        out: list[Candidate] = []
        for rid in ordered_ids:
            rec = self._get_recording(rid)
            if not rec:
                continue
            out.extend(self._recording_to_candidates(track, rec, ext_scores.get(rid, 0.0)))
        return out

    # -- recording fetch + cache -------------------------------------------
    def _get_recording(self, rec_id: str) -> dict | None:
        if rec_id in self._mem_cache:
            return self._mem_cache[rec_id]
        disk = self._read_cache(rec_id)
        if disk is not None:
            self._mem_cache[rec_id] = disk
            return disk
        data = self._fetch_recording(rec_id)
        if data is not None:
            self._mem_cache[rec_id] = data
            self._write_cache(rec_id, data)
        return data

    def _fetch_recording(self, rec_id: str) -> dict | None:
        for includes in (_INCLUDES, _FALLBACK_INCLUDES):
            try:
                res = self._mb.get_recording_by_id(rec_id, includes=includes)
                rec = res.get("recording") if isinstance(res, dict) else None
                if rec is not None:
                    return rec
            except Exception:
                continue
        return None

    def _cache_path(self, rec_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in rec_id)
        return os.path.join(self.cache_dir, f"mb_rec_{safe}.json")

    def _read_cache(self, rec_id: str) -> dict | None:
        path = self._cache_path(rec_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _write_cache(self, rec_id: str, data: dict) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self._cache_path(rec_id), "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception:
            pass

    # -- parsing ------------------------------------------------------------
    def _recording_to_candidates(self, track: InputTrack, rec: dict,
                                 mb_ext_score: float) -> list[Candidate]:
        title = rec.get("title", "") or ""
        # Blend MusicBrainz's own score with a transliteration-tolerant title
        # match so a strong MB hit on the wrong-spelled title cannot run away.
        score = max(0.0, min(1.0, 0.6 * mb_ext_score + 0.4 * sim(track.existing_title, title)))

        credits = self._parse_credits(rec)

        candidates: list[Candidate] = []
        for rel in rec.get("release-list", []) or [{}]:
            rg = rel.get("release-group", {}) or {}
            rtype = _release_type(rg)
            film = None
            if rtype == ReleaseType.SOUNDTRACK:
                film = rg.get("title") or rel.get("title")
            va_phrase = (rg.get("artist-credit-phrase")
                         or rel.get("artist-credit-phrase") or "").lower()
            candidates.append(Candidate(
                source=self.name,
                title=title,
                release_title=rel.get("title") or rg.get("title", ""),
                release_type=rtype,
                credits=credits,
                film=film,
                release_date=rel.get("date"),
                is_various_artists=va_phrase == "various artists",
                recording_mbid=rec.get("id"),
                release_mbid=rel.get("id"),
                match_score=score,
                raw=rec,
            ))
        return candidates

    @staticmethod
    def _parse_credits(rec: dict) -> list[ArtistCredit]:
        singers: list[ArtistCredit] = []
        composers: list[ArtistCredit] = []
        lyricists: list[ArtistCredit] = []
        seen_singer: set[str] = set()

        def add_singer(name: str | None, mbid: str | None) -> None:
            if not name:
                return
            key = name.strip().lower()
            if key in seen_singer:
                return
            seen_singer.add(key)
            singers.append(ArtistCredit(name, "singer", mbid))

        # Primary recording artist credit — playback singers, in order.
        for ac in rec.get("artist-credit", []) or []:
            if isinstance(ac, dict) and "artist" in ac:
                art = ac["artist"]
                add_singer(art.get("name"), art.get("id"))

        # Recording-level relations (vocal / performer / occasionally composer).
        for rel in rec.get("artist-relation-list", []) or []:
            artist = rel.get("artist", {}) or {}
            who = artist.get("name")
            mbid = artist.get("id")
            rtype = (rel.get("type") or "").lower()
            if not who:
                continue
            if "vocal" in rtype or "performer" in rtype:
                add_singer(who, mbid)
            elif "composer" in rtype or "writer" in rtype:
                composers.append(ArtistCredit(who, "composer", mbid))
            elif "lyric" in rtype or "librett" in rtype:
                lyricists.append(ArtistCredit(who, "lyricist", mbid))

        # Work relationships carry composer/lyricist for film songs.
        for work_rel in rec.get("work-relation-list", []) or []:
            work = work_rel.get("work", {}) or {}
            for rel in work.get("artist-relation-list", []) or []:
                artist = rel.get("artist", {}) or {}
                who = artist.get("name")
                mbid = artist.get("id")
                rtype = (rel.get("type") or "").lower()
                if not who:
                    continue
                if "composer" in rtype:
                    composers.append(ArtistCredit(who, "composer", mbid))
                elif "lyric" in rtype or "librett" in rtype:
                    lyricists.append(ArtistCredit(who, "lyricist", mbid))
                elif "writer" in rtype:
                    composers.append(ArtistCredit(who, "composer", mbid))

        return singers + _dedupe(composers) + _dedupe(lyricists)
