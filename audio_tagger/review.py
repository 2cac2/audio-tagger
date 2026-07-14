"""Review queue.

Anything below the auto-write confidence threshold is written here instead
of onto your files, so the residual (large for Bollywood given ~85%
fingerprint accuracy) is auditable rather than silently mis-tagged.

Format is a plain CSV you can eyeball in any spreadsheet; approving a row is
just setting `approved` to 1 and (optionally) correcting a field.

The CSV round-trips the full provenance a write needs — not just the visible
credit strings but ``year``, ``grouping`` (film), the compilation flag, and
the two MusicBrainz ids — so ``apply-reviews`` reconstructs the same tag dict
the auto path would have written instead of silently dropping that context.
"""

from __future__ import annotations

import csv
import re

_FIELDS = [
    "approved", "path", "confidence", "agreement", "used_websearch_fallback",
    "title", "artist", "albumartist", "album", "grouping", "year",
    "composer", "lyricist", "comp",
    "musicbrainz_trackid", "musicbrainz_albumid",
    "release_type", "source", "reasons",
]

# Approved-flag truthy tokens (kept from the original parsing).
_TRUTHY = ("1", "yes", "true", "y")

_YEAR_RE = re.compile(r"(\d{4})")


def _join(v) -> str:
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    return "" if v is None else str(v)


def _split(v: str) -> list[str]:
    """Split a '; '-joined multi-value cell back into a clean list."""
    return [s.strip() for s in (v or "").split(";") if s.strip()]


def _parse_year(v) -> int | None:
    """Pull a 4-digit year out of the cell (handles '2011' and '2011-01-01')."""
    if v is None:
        return None
    m = _YEAR_RE.search(str(v))
    return int(m.group(1)) if m else None


def _parse_comp(v) -> int:
    """Normalize the compilation flag to a 0/1 int (default 0)."""
    return 1 if str(v or "").strip().lower() in _TRUTHY else 0


def _parse_mbid(v) -> str | None:
    """Return the MusicBrainz id string, or None when the cell is empty."""
    s = str(v or "").strip()
    return s or None


def write_queue(resolutions, path: str) -> int:
    """Write the review CSV; return the number of rows needing review."""
    rows = [r for r in resolutions if r.needs_review]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            t = r.tags
            w.writerow({
                "approved": 0,
                "path": r.track.path,
                "confidence": r.confidence,
                "agreement": r.agreement,
                "used_websearch_fallback": int(r.used_websearch_fallback),
                "title": _join(t.get("title")),
                "artist": _join(t.get("artist")),
                "albumartist": _join(t.get("albumartist")),
                "album": _join(t.get("album")),
                "grouping": _join(t.get("grouping")),
                "year": _join(t.get("year")),
                "composer": _join(t.get("composer")),
                "lyricist": _join(t.get("lyricist")),
                "comp": _join(t.get("comp")),
                "musicbrainz_trackid": _join(t.get("musicbrainz_trackid")),
                "musicbrainz_albumid": _join(t.get("musicbrainz_albumid")),
                "release_type": r.chosen.release_type.value if r.chosen else "",
                "source": r.chosen.source if r.chosen else "",
                "reasons": " | ".join(r.reasons),
            })
    return len(rows)


def read_approvals(path: str) -> dict[str, dict]:
    """Read back an edited queue; return {path: corrected_fields} for approved rows.

    The returned field dict mirrors a ``tagmap.build_tags`` result so
    ``apply-reviews`` can hand it straight to ``writer.write_tags``: multi-value
    credits come back as lists, ``year`` as ``int | None``, ``comp`` as a 0/1
    int, and the MusicBrainz ids as strings (or ``None`` when blank).
    """
    approved: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("approved", "")).strip().lower() in _TRUTHY:
                approved[row["path"]] = {
                    "title": (row.get("title") or "").strip(),
                    "artist": _split(row.get("artist", "")),
                    "albumartist": _split(row.get("albumartist", "")),
                    "album": (row.get("album") or "").strip(),
                    "grouping": (row.get("grouping") or "").strip(),
                    "year": _parse_year(row.get("year")),
                    "composer": _split(row.get("composer", "")),
                    "lyricist": _split(row.get("lyricist", "")),
                    "comp": _parse_comp(row.get("comp")),
                    "musicbrainz_trackid": _parse_mbid(row.get("musicbrainz_trackid")),
                    "musicbrainz_albumid": _parse_mbid(row.get("musicbrainz_albumid")),
                }
    return approved
