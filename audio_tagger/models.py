"""Core data model for the tagging harness.

Everything the sources return is normalized into these shapes so the
resolver, heuristics and tag-mapper never have to care which source a
candidate came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ReleaseType(str, Enum):
    """Normalized release-group type, used by the preference heuristic."""

    SOUNDTRACK = "soundtrack"      # film OST — the target for Bollywood film songs
    ALBUM = "album"               # a normal studio album (indipop, non-film)
    SINGLE = "single"             # standalone single
    COMPILATION = "compilation"   # "Greatest Hits", "Superhits", "Party Mix"...
    EP = "ep"
    UNKNOWN = "unknown"


@dataclass
class ArtistCredit:
    """A single credited person, with the role that decides which tag it lands in."""

    name: str
    role: str                     # "singer" | "composer" | "lyricist" | "producer" | "mixer" | "performer"
    mbid: Optional[str] = None    # MusicBrainz artist id when known


@dataclass
class Candidate:
    """One possible identification of a track, from one source."""

    source: str                        # "musicbrainz" | "itunes" | "spotify" | "discogs" | "websearch"
    title: str
    release_title: str                 # album/soundtrack/single title as the source names it
    release_type: ReleaseType
    credits: list[ArtistCredit] = field(default_factory=list)
    film: Optional[str] = None         # film name if this is a film song
    year: Optional[int] = None
    release_date: Optional[str] = None  # ISO date if available; used to prefer earliest release
    is_various_artists: bool = False
    recording_mbid: Optional[str] = None
    release_mbid: Optional[str] = None
    match_score: float = 0.0           # source's own confidence 0..1 (e.g. AcoustID/MB score)
    raw: dict = field(default_factory=dict)  # original payload, for debugging/review

    # --- convenience accessors by role -------------------------------------
    def _names(self, role: str) -> list[str]:
        return [c.name for c in self.credits if c.role == role and c.name]

    @property
    def singers(self) -> list[str]:
        return self._names("singer") or self._names("performer")

    @property
    def composers(self) -> list[str]:
        return self._names("composer")

    @property
    def lyricists(self) -> list[str]:
        return self._names("lyricist")

    @property
    def producers(self) -> list[str]:
        return self._names("producer")

    @property
    def mixers(self) -> list[str]:
        return self._names("mixer")


@dataclass
class InputTrack:
    """What we know about a local file before tagging."""

    path: str
    existing_title: Optional[str] = None
    existing_artist: Optional[str] = None
    existing_album: Optional[str] = None
    duration_sec: Optional[float] = None
    acoustid_fingerprint: Optional[str] = None


@dataclass
class Resolution:
    """The harness's final decision for one track."""

    track: InputTrack
    chosen: Optional[Candidate]
    confidence: float                       # 0..1 combined confidence
    agreement: float                        # cross-source agreement 0..1
    tags: dict = field(default_factory=dict)   # final tag field -> value(s)
    lyrics_path: Optional[str] = None
    used_websearch_fallback: bool = False
    needs_review: bool = True
    reasons: list[str] = field(default_factory=list)  # human-readable why
