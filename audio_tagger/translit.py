"""Transliteration-tolerant text normalization for romanized Hindi/Punjabi.

MusicBrainz, JioSaavn, iTunes and the file tags themselves disagree wildly on
how to romanize Devanagari: "humein" / "hume" / "humey", "pyaar" / "pyar",
"khoobsurat" / "khubsurat", "zindagi" / "jindagi". A plain string ratio treats
those as different songs. This module folds the common romanization variants so
similarity survives the spelling drift, and pulls the film name out of the
`(From "Film")` / `(Film)` suffixes that streaming services bolt onto titles.

Only the standard library is used, so this imports with nothing installed.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Whole-token spelling fixes applied before the character folds below. Keys are
# in their raw romanized spelling (folds would otherwise rewrite them first).
_TOKEN_FIXES = {
    "humein": "hame",
    "humey": "hame",
    "hume": "hame",
    "kabhie": "kabhi",
}

# Ordered substring folds that collapse romanization variants onto one form.
# Order matters: multi-char rules (sh, ph) must run before any single-char rule
# that could consume their first letter.
_FOLDS: list[tuple[str, str]] = [
    ("aa", "a"),
    ("ee", "i"),
    ("oo", "u"),
    ("ph", "f"),
    ("sh", "s"),   # soft-tier: "sh" and "s" treated alike
    ("w", "v"),    # w <-> v collapse onto v
    ("z", "j"),    # soft z ("zindagi" ~ "jindagi")
]

_WS = re.compile(r"\s+")


def _strip_diacritics(s: str) -> str:
    """NFKD-decompose and drop combining marks (é -> e, ā -> a)."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _light_norm(s: str | None) -> str:
    """Lightly normalize: strip diacritics, lowercase, drop punctuation.

    This is the "raw" comparison form — no romanization folding — so that
    ``sim`` can take the better of the light and the fully-folded ratio.
    """
    if not s:
        return ""
    s = _strip_diacritics(s).lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return _WS.sub(" ", s).strip()


def normalize_roman_hindi(s: str | None) -> str:
    """Aggressively fold romanization variants to a single canonical form.

    Applies (in order): diacritic strip, lowercase, punctuation -> space,
    whole-token fixes (humein/humey/hume -> hame, kabhie -> kabhi), then the
    character folds (aa->a, ee->i, oo->u, ph->f, sh->s, w->v, z->j). Whitespace
    is collapsed. Returns "" for empty input.
    """
    base = _light_norm(s)
    if not base:
        return ""

    tokens = [_TOKEN_FIXES.get(tok, tok) for tok in base.split(" ")]
    folded = " ".join(tokens)
    for src, dst in _FOLDS:
        folded = folded.replace(src, dst)
    return _WS.sub(" ", folded).strip()


# ---------------------------------------------------------------------------
# Film extraction
# ---------------------------------------------------------------------------

# `(From "Brahmastra")`, `(From 'Film')`, `(From Film)` — quotes optional.
_FROM_RE = re.compile(r"\(\s*from\s+(?P<film>.+?)\s*\)", re.IGNORECASE)

# A trailing bare parenthetical, e.g. `Song (Ae Dil Hai Mushkil)`.
_TRAILING_PAREN_RE = re.compile(r"\(([^()]*)\)\s*$")

# Parentheticals that name an edit/version, not a film — never a film name.
_VERSION_MARKERS = re.compile(
    r"\b("
    r"remix|reprise|unplugged|acoustic|live|lo-?fi|slowed|reverb|cover|"
    r"version|mix|remaster(?:ed)?|edit|instrumental|karaoke|female|male|"
    r"duet|sped\s*up|extended|radio|club|remake|lyrical|full\s*song|"
    r"video|audio|reloaded|refix|mashup"
    r")\b",
    re.IGNORECASE,
)

_QUOTES = "\"'“”‘’"


def extract_film_from_title(title: str) -> tuple[str, str | None]:
    """Split a title into ``(clean_title, film_or_None)``.

    Recognizes ``(From "Film")`` first (the streaming-service convention), then
    a trailing bare ``(Film)`` — unless that parenthetical is an obvious
    version/edit marker (``(Remix)``, ``(Unplugged)``...), which is left in the
    title and reported as no film. The film is captured *before* stripping.
    """
    if not title:
        return "", None
    title = title.strip()

    m = _FROM_RE.search(title)
    if m:
        film = m.group("film").strip().strip(_QUOTES).strip()
        clean = (title[: m.start()] + title[m.end():]).strip().strip("-").strip()
        return (clean or title, film or None)

    m = _TRAILING_PAREN_RE.search(title)
    if m:
        inner = m.group(1).strip()
        if inner and not _VERSION_MARKERS.search(inner):
            clean = title[: m.start()].strip().strip("-").strip()
            return (clean or title, inner)

    return (title, None)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def sim(a: str | None, b: str | None) -> float:
    """Transliteration-tolerant similarity of two strings in [0, 1].

    Returns the better of two SequenceMatcher ratios: one on the lightly
    normalized (raw) forms and one on the fully romanization-folded forms. This
    way an exact raw match and a fold-only match both score high. Returns 0.0 if
    either input is empty.
    """
    la, lb = _light_norm(a), _light_norm(b)
    if not la or not lb:
        return 0.0

    raw_ratio = SequenceMatcher(None, la, lb).ratio()

    ra, rb = normalize_roman_hindi(a), normalize_roman_hindi(b)
    roman_ratio = SequenceMatcher(None, ra, rb).ratio() if ra and rb else 0.0

    return max(raw_ratio, roman_ratio)
