"""Map a resolved Candidate onto the user's chosen tag layout.

User's modeling decision:
  ARTIST       = playback singer(s)                       [multi-valued]
  ALBUMARTIST  = music director / composer                [film]
                 -> producer, then mixer                  [Punjabi / indipop / non-film]
  ALBUM        = film soundtrack        if it is a film song
                 else the album title   if the track belongs to an album
                 else the song title    if it is a true standalone single
  COMPOSER     = music director / composer                [always, when known]
  LYRICIST     = lyricist                                 [always, when known]

Multi-valued tags are represented as a list; the beets writer joins them
with a null separator so players show real multi-artist credits rather than
"A & B" mashed into one string.
"""

from __future__ import annotations

from .credits import canonical_acts
from .models import Candidate, ReleaseType


def _album_field(c: Candidate) -> str:
    """Apply the ALBUM branch logic."""
    # Film song -> the soundtrack / film is the album.
    if c.film:
        title = c.film
        # Normalize to the conventional OST naming if the source didn't.
        if "soundtrack" not in title.lower() and "motion picture" not in title.lower():
            return f"{title} (Original Motion Picture Soundtrack)"
        return title

    if c.release_type == ReleaseType.SOUNDTRACK and c.release_title:
        return c.release_title

    # Belongs to a real album/EP -> use that album.
    if c.release_type in (ReleaseType.ALBUM, ReleaseType.EP) and c.release_title:
        return c.release_title

    # True single (or unknown release) -> the song title stands in as album.
    return c.release_title or c.title


def _album_artist(c: Candidate) -> list[str]:
    """ALBUMARTIST cascade: composer -> producer -> mixer -> singers.

    Composer names are folded through ``credits.canonical_acts`` so a known
    duo/trio (e.g. Vishal Dadlani + Shekhar Ravjiani) shows as the single
    credited act ("Vishal-Shekhar") rather than two separate names. Singers,
    producers and mixers are left as-is — only composer acts fold.
    """
    if c.composers:
        return canonical_acts(c.composers)
    if c.producers:
        return c.producers
    if c.mixers:
        return c.mixers
    # Last resort so the field is never empty (keeps players from bucketing
    # everything under a blank "Various Artists").
    return c.singers or ["Various Artists"]


def build_tags(c: Candidate) -> dict:
    """Produce the final tag dict for one chosen candidate."""
    tags: dict[str, object] = {
        "title": c.title,
        "album": _album_field(c),
        "albumartist": _album_artist(c),
    }

    if c.singers:
        tags["artist"] = c.singers
    if c.composers:
        # Fold composer duos/trios to the credited act so COMPOSER and
        # ALBUMARTIST agree (both show "Vishal-Shekhar", not the split names).
        tags["composer"] = canonical_acts(c.composers)
    if c.lyricists:
        tags["lyricist"] = c.lyricists
    if c.year:
        tags["year"] = c.year
    if c.film:
        # Keep the raw film name too; useful for grouping/searching.
        tags["grouping"] = c.film

    # Provenance so a later human can audit where this came from.
    if c.recording_mbid:
        tags["musicbrainz_trackid"] = c.recording_mbid
    if c.release_mbid:
        tags["musicbrainz_albumid"] = c.release_mbid

    return tags
