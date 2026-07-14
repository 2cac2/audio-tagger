"""Source adapters. Import lazily so a missing optional dep or key never
breaks the whole package — the CLI decides which sources to activate.

The concrete source classes are re-exported here for convenience, but each
module only touches its third-party dependency *inside* methods (or, for
MusicBrainz, inside ``__init__``), so importing this package pulls in nothing
beyond the standard library.
"""

from .acoustid_source import AcoustIDSource
from .base import Source, query_string, score_match
from .deezer import DeezerSource
from .discogs import DiscogsSource
from .itunes import ITunesSource
from .jiosaavn import JioSaavnSource
from .musicbrainz import MusicBrainzSource
from .spotify import SpotifySource

__all__ = [
    "Source",
    "query_string",
    "score_match",
    "AcoustIDSource",
    "DeezerSource",
    "DiscogsSource",
    "ITunesSource",
    "JioSaavnSource",
    "MusicBrainzSource",
    "SpotifySource",
]
