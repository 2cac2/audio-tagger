"""Review queue.

Anything below the auto-write confidence threshold is written here instead
of onto your files, so the residual (large for Bollywood given ~85%
fingerprint accuracy) is auditable rather than silently mis-tagged.

Format is a plain CSV you can eyeball in any spreadsheet; approving a row is
just setting `approved` to 1 and (optionally) correcting a field.
"""

from __future__ import annotations

import csv

_FIELDS = [
    "approved", "path", "confidence", "agreement", "used_websearch_fallback",
    "title", "artist", "albumartist", "album", "composer", "lyricist",
    "release_type", "source", "reasons",
]


def _join(v) -> str:
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    return "" if v is None else str(v)


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
                "composer": _join(t.get("composer")),
                "lyricist": _join(t.get("lyricist")),
                "release_type": r.chosen.release_type.value if r.chosen else "",
                "source": r.chosen.source if r.chosen else "",
                "reasons": " | ".join(r.reasons),
            })
    return len(rows)


def read_approvals(path: str) -> dict[str, dict]:
    """Read back an edited queue; return {path: corrected_fields} for approved rows."""
    approved: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("approved", "")).strip() in ("1", "yes", "true", "y"):
                approved[row["path"]] = {
                    "title": row["title"],
                    "artist": [s.strip() for s in row["artist"].split(";") if s.strip()],
                    "albumartist": [s.strip() for s in row["albumartist"].split(";") if s.strip()],
                    "album": row["album"],
                    "composer": [s.strip() for s in row["composer"].split(";") if s.strip()],
                    "lyricist": [s.strip() for s in row["lyricist"].split(";") if s.strip()],
                }
    return approved
