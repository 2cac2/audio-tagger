"""Source adapter protocol.

Each source turns an InputTrack into a list of normalized Candidates.
Adapters must be independent and side-effect free; the resolver isolates
failures so one dead source never blocks a track.
"""

from __future__ import annotations

from typing import Protocol

from ..models import Candidate, InputTrack
from ..translit import sim  # first-party, stdlib-only — safe to import eagerly


class Source(Protocol):
    name: str

    def search(self, track: InputTrack) -> list[Candidate]:
        ...


def query_string(track: InputTrack) -> str:
    """Best free-text query we can build from what a file already claims."""
    parts = [track.existing_title, track.existing_artist, track.existing_album]
    return " ".join(p for p in parts if p).strip()


def score_match(track: InputTrack, title: str, artists: list[str]) -> float:
    """Transliteration-tolerant match score between a file and a candidate.

    Weighs title similarity 0.7 and artist similarity 0.3, using ``translit.sim``
    so romanization drift ("humein"/"hume") does not tank an otherwise correct
    match. The result is clamped to ``[0, 1]``. Replaces the hardcoded priors
    (0.55 / 0.7 / 0.72) that older adapters used for every hit alike.
    """
    title_sim = sim(track.existing_title, title)
    artist_sim = sim(track.existing_artist, " ".join(a for a in (artists or []) if a))
    score = 0.7 * title_sim + 0.3 * artist_sim
    return max(0.0, min(1.0, score))
