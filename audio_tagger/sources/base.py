"""Source adapter protocol.

Each source turns an InputTrack into a list of normalized Candidates.
Adapters must be independent and side-effect free; the resolver isolates
failures so one dead source never blocks a track.
"""

from __future__ import annotations

from typing import Protocol

from ..models import Candidate, InputTrack


class Source(Protocol):
    name: str

    def search(self, track: InputTrack) -> list[Candidate]:
        ...


def query_string(track: InputTrack) -> str:
    """Best free-text query we can build from what a file already claims."""
    parts = [track.existing_title, track.existing_artist, track.existing_album]
    return " ".join(p for p in parts if p).strip()
