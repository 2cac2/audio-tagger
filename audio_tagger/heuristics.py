"""Release-preference scoring.

The single most important rule for this library: when the same recording
appears on both an original film soundtrack (or an artist's own album) and
on a "Greatest Hits / Superhits / Party Mix" compilation, we must prefer
the original release. Generic taggers pick whatever ranks first, which is
usually the compilation. This module encodes the preference explicitly.
"""

from __future__ import annotations

import re

from .models import Candidate, ReleaseType

# Words that strongly signal a compilation even when a source mislabels the type.
_COMPILATION_MARKERS = re.compile(
    r"\b("
    r"greatest\s+hits|superhits|super\s+hits|best\s+of|hits?\s+of|"
    r"party\s+mix|dance\s+mix|remix(?:es)?|collection|golden|"
    r"top\s+\d+|vol\.?\s*\d+|volume\s+\d+|chartbusters?|"
    r"all\s+time|evergreen|nonstop|non\s*stop|jukebox|mashup"
    r")\b",
    re.IGNORECASE,
)

# Base desirability by release type (higher = more preferred).
_TYPE_BASE = {
    ReleaseType.SOUNDTRACK: 1.00,
    ReleaseType.ALBUM: 0.90,
    ReleaseType.EP: 0.80,
    ReleaseType.SINGLE: 0.75,
    ReleaseType.UNKNOWN: 0.40,
    ReleaseType.COMPILATION: 0.10,
}


def looks_like_compilation(candidate: Candidate) -> bool:
    """True if the title text screams compilation, regardless of declared type."""
    text = f"{candidate.release_title} {candidate.film or ''}"
    return bool(_COMPILATION_MARKERS.search(text))


def _year_from(candidate: Candidate) -> int | None:
    if candidate.year:
        return candidate.year
    if candidate.release_date and len(candidate.release_date) >= 4:
        try:
            return int(candidate.release_date[:4])
        except ValueError:
            return None
    return None


def release_preference_score(candidate: Candidate) -> float:
    """Score a candidate's *release* (not its match accuracy) in [0, 1].

    Combines: declared type, textual compilation markers, various-artists
    penalty, and a mild earliest-release bonus (originals predate reissues).
    """
    score = _TYPE_BASE.get(candidate.release_type, 0.4)

    # Textual compilation markers override an optimistic declared type.
    if looks_like_compilation(candidate):
        score = min(score, 0.15)

    # A soundtrack is inherently various-artists in the album-artist sense, so
    # only penalize VA when it is *not* a soundtrack (i.e. a comp masquerade).
    if candidate.is_various_artists and candidate.release_type != ReleaseType.SOUNDTRACK:
        score -= 0.20

    # A film song that correctly names its film gets a small nudge — that is
    # the canonical "original album" for Bollywood.
    if candidate.film and candidate.release_type == ReleaseType.SOUNDTRACK:
        score += 0.10

    # Earliest-release bonus: originals come first, reissues/comps later.
    year = _year_from(candidate)
    if year:
        # 1950..2030 mapped to a tiny 0..0.05 bonus for being older.
        span = max(0, min(80, 2030 - year))
        score += (span / 80) * 0.05

    return max(0.0, min(1.0, score))


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Return candidates sorted best-first.

    Ordering key blends the source's own match score with our release
    preference, so an accurate match on a compilation still loses to a
    slightly weaker match on the original soundtrack.
    """
    def key(c: Candidate) -> float:
        return 0.45 * c.match_score + 0.55 * release_preference_score(c)

    return sorted(candidates, key=key, reverse=True)
