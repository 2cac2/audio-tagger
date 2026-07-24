"""Copy-only library organization.

Copies tracks into a clean ``<Album>/<Title>`` folder tree **without ever
touching the originals**. Copy-only is a hard guarantee, enforced here, not just
a convention:

* the only filesystem mutation is :func:`shutil.copy2` (which *copies* bytes and
  metadata to a NEW path) — never ``move``, ``rename``, ``remove``, or a write
  back onto a source file;
* a destination that already exists is **never overwritten** — an identical file
  is skipped as a duplicate, a differing file gets a ``" (2)"`` suffix;
* a source is never copied onto itself (``src == dest`` is refused).

Layout is album/film based: ``<dest_root>/<Album>/<NN? Title><ext>``. The album
comes from the resolved/existing ALBUM tag (for Bollywood this is the film OST),
so a film's tracks land together regardless of how many singers they have.

Standard library only — ``mutagen`` is imported lazily inside :func:`items_from_paths`
so the module imports with nothing installed.
"""

from __future__ import annotations

import filecmp
import os
import re
import shutil
from dataclasses import dataclass

# Characters illegal in a path segment on common filesystems (plus control
# chars). Unicode letters (Devanagari, Gurmukhi, …) are preserved.
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAILING = re.compile(r"[ .]+$")


def sanitize_segment(name: str | None, fallback: str = "Unknown", maxlen: int = 150) -> str:
    """Make ``name`` safe to use as a single folder/file path segment."""
    s = _INVALID.sub("_", (name or "").strip())
    s = re.sub(r"\s+", " ", s).strip()
    s = _TRAILING.sub("", s)          # no trailing dots/spaces (Windows-hostile)
    if not s:
        s = fallback
    if len(s) > maxlen:
        s = s[:maxlen].rstrip()
    return s or fallback


@dataclass
class OrgItem:
    """One track to place: its source path and the album/title it belongs under."""

    src: str
    album: str
    title: str
    track_no: int | None = None


@dataclass
class CopyOp:
    """A planned (then executed) copy. ``status`` records the outcome."""

    src: str
    dest: str
    album: str
    status: str = "planned"   # planned|dry_run|copied|renamed|skipped_dup|error
    note: str = ""


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def _filename(item: OrgItem, ext: str) -> str:
    title = sanitize_segment(item.title, fallback="Untitled")
    if item.track_no:
        return f"{item.track_no:02d} {title}{ext}"
    return f"{title}{ext}"


def plan(items: list[OrgItem], dest_root: str) -> list[CopyOp]:
    """Compute the copy operations for ``items`` under ``dest_root``.

    Resolves *within-plan* collisions (two different sources mapping to the same
    destination) by appending ``" (2)"``, ``" (3)"`` …, so a single run never
    plans two copies onto one path.
    """
    root = os.path.abspath(dest_root)
    ops: list[CopyOp] = []
    used: set[str] = set()
    for it in items:
        ext = os.path.splitext(it.src)[1]
        album = sanitize_segment(it.album, fallback="Unknown Album")
        folder = os.path.join(root, album)
        base = _filename(it, ext)
        stem, e = os.path.splitext(base)
        dest = os.path.join(folder, base)
        n = 2
        while dest.lower() in used:
            dest = os.path.join(folder, f"{stem} ({n}){e}")
            n += 1
        used.add(dest.lower())
        ops.append(CopyOp(src=os.path.abspath(it.src), dest=dest, album=album))
    return ops


# ---------------------------------------------------------------------------
# execution (copy-only)
# ---------------------------------------------------------------------------

def _next_free(dest: str) -> str:
    folder = os.path.dirname(dest)
    stem, ext = os.path.splitext(os.path.basename(dest))
    n = 2
    while True:
        cand = os.path.join(folder, f"{stem} ({n}){ext}")
        if not os.path.exists(cand):
            return cand
        n += 1


def execute(ops: list[CopyOp], dry_run: bool = True, on_progress=None) -> dict:
    """Perform the planned copies. ``dry_run`` plans without writing anything.

    COPY-ONLY: the only mutating call is :func:`shutil.copy2`, which writes a new
    file at ``dest``; sources are opened read-only and never modified, moved, or
    deleted. Existing destinations are never overwritten. ``on_progress(op)`` is
    invoked after each op so a UI can update live. Returns a summary dict.
    """
    summary = {"total": len(ops), "copied": 0, "renamed": 0,
               "skipped_dup": 0, "errors": 0, "dry_run": 0, "bytes": 0}
    for op in ops:
        try:
            if os.path.abspath(op.src) == os.path.abspath(op.dest):
                op.status, op.note = "error", "source and destination are the same file"
                summary["errors"] += 1
            elif not os.path.isfile(op.src):
                op.status, op.note = "error", "source file missing"
                summary["errors"] += 1
            else:
                renamed = False
                if os.path.exists(op.dest):
                    if filecmp.cmp(op.src, op.dest, shallow=False):
                        op.status, op.note = "skipped_dup", "identical file already present"
                        summary["skipped_dup"] += 1
                        if on_progress:
                            on_progress(op)
                        continue
                    op.dest = _next_free(op.dest)    # never overwrite a different file
                    op.note, renamed = "renamed to avoid overwrite", True
                if dry_run:
                    op.status = "dry_run"
                    summary["dry_run"] += 1
                else:
                    os.makedirs(os.path.dirname(op.dest), exist_ok=True)
                    shutil.copy2(op.src, op.dest)    # <-- the ONLY mutation: a copy
                    op.status = "renamed" if renamed else "copied"
                    summary["renamed" if renamed else "copied"] += 1
                    try:
                        summary["bytes"] += os.path.getsize(op.dest)
                    except OSError:
                        pass
        except Exception as exc:                     # a bad file must not abort the run
            op.status, op.note = "error", str(exc)
            summary["errors"] += 1
        if on_progress:
            on_progress(op)
    return summary


# ---------------------------------------------------------------------------
# item builders
# ---------------------------------------------------------------------------

def _as_int(value) -> int | None:
    try:
        return int(str(value).split("/")[0])
    except (TypeError, ValueError):
        return None


def items_from_paths(paths: list[str]) -> list[OrgItem]:
    """Build items from files' *current* tags (no network, no re-resolution)."""
    out: list[OrgItem] = []
    for p in paths:
        tags = _read_tags(p)
        out.append(OrgItem(
            src=p,
            album=tags.get("album") or "Unknown Album",
            title=tags.get("title") or os.path.splitext(os.path.basename(p))[0],
            track_no=tags.get("track_no"),
        ))
    return out


def _read_tags(path: str) -> dict:
    try:
        from mutagen import File as MutagenFile   # lazy: stdlib-only import of module
        audio = MutagenFile(path, easy=True)
        if audio is None:
            return {}

        def first(key):
            v = audio.get(key)
            return v[0] if v else None

        return {"album": first("album"), "title": first("title"),
                "track_no": _as_int(first("tracknumber"))}
    except Exception:
        return {}


def items_from_review_items(review_items) -> list[OrgItem]:
    """Build items from resolved :class:`ReviewItem`s (the ``tag --json-out`` shape).

    Uses each item's ``final_tags()`` (post-albumize ALBUM/TITLE), falling back to
    the file's existing tags, so organization follows the *resolved* album.
    """
    out: list[OrgItem] = []
    for it in review_items:
        tags = it.final_tags() if hasattr(it, "final_tags") else {}
        album = tags.get("album") or getattr(it.track, "existing_album", None) or "Unknown Album"
        title = (tags.get("title") or getattr(it.track, "existing_title", None)
                 or os.path.splitext(os.path.basename(it.track.path))[0])
        out.append(OrgItem(
            src=it.track.path, album=str(album), title=str(title),
            track_no=_as_int(tags.get("track") or tags.get("tracknumber")),
        ))
    return out
