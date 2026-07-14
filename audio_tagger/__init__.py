"""audio_tagger — an AI metadata harness for Indian (Bollywood / Punjabi /
indipop) music libraries.

Design in one line: a proven structured-data cascade does the identification,
a release-preference heuristic beats the "greatest hits" trap, an album
reconciliation pass keeps multi-artist films as one album, and an LLM is used
only as a constrained tie-breaker on disagreement — never to invent metadata.
"""

from .models import ArtistCredit, Candidate, InputTrack, ReleaseType, Resolution

__version__ = "0.1.0"
__all__ = ["ArtistCredit", "Candidate", "InputTrack", "ReleaseType", "Resolution"]
