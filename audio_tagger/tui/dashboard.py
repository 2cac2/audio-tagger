"""Full-workflow dashboard controller (tag → organize), immich-go styled.

This module holds the *pure* workflow controller — no Textual, standard library
only — so it is fully unit-testable. The immich-go-styled Textual UI lives in
:mod:`audio_tagger.tui.dashboard_app` and is imported lazily by
:func:`run_dashboard`.

The controller reuses the existing pipeline and adds no resolution logic:

1. **Analyze** — ``tui.state.load_session`` scans + resolves the source (or loads
   a ``tag --json-out`` file), yielding reviewable items.
2. **Organize** — ``organize.plan`` / ``organize.execute`` copy each resolved
   track into ``<dest>/<Album>/<Title>``. COPY-ONLY: originals are never moved,
   deleted, or modified (see :mod:`audio_tagger.organize`).
3. Optionally **tag the copies** with the resolved tags (never the originals).

Progress is surfaced through an ``on_event`` callback so any front-end (the
Textual dashboard, or a plain-text runner) can render it live.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkflowStats:
    """Live counters shown as immich-go-style stat tiles."""

    scanned: int = 0
    resolved_auto: int = 0
    needs_review: int = 0
    copied: int = 0
    renamed: int = 0
    duplicates: int = 0
    errors: int = 0
    tagged_copies: int = 0
    bytes: int = 0
    stage: str = "idle"          # idle|analyze|organize|done
    done: bool = False

    def as_tiles(self) -> list[tuple[str, int, str]]:
        """(label, value, kind) triples; ``kind`` drives the tile color."""
        return [
            ("Scanned", self.scanned, "info"),
            ("Auto-tagged", self.resolved_auto, "ok"),
            ("Needs review", self.needs_review, "warn"),
            ("Copied", self.copied + self.renamed, "ok"),
            ("Duplicates", self.duplicates, "muted"),
            ("Errors", self.errors, "err"),
        ]


# One event = (kind, payload). kinds: "stage", "progress", "log", "stats", "done".
def _noop(_kind, _payload):  # default sink
    pass


def run_workflow(source: str, dest: str, *, config=None, apply: bool = False,
                 tag_copies: bool = False, on_event=None) -> WorkflowStats:
    """Run Analyze → Organize, emitting events, and return the final stats.

    ``apply=False`` is a dry-run: nothing is copied and no tags are written.
    ``on_event(kind, payload)`` receives:
      * ("stage", {"name","total"})           — a stage started
      * ("progress", {"stage","done","total","label"})  — one unit advanced
      * ("log", "text")                        — a log line
      * ("stats", WorkflowStats)               — counters changed
      * ("done", WorkflowStats)                — finished
    """
    emit = on_event or _noop
    stats = WorkflowStats(stage="analyze")

    # -- 1. Analyze (scan + resolve, reusing the pipeline) ------------------
    emit("stage", {"name": "analyze", "total": 0})
    emit("log", f"Analyzing {source} …")
    from ..tui.state import load_session
    session = load_session(source, config=config)
    items = list(session.items)
    stats.scanned = len(items)
    stats.needs_review = sum(1 for it in items if getattr(it, "needs_review", True))
    stats.resolved_auto = stats.scanned - stats.needs_review
    emit("stats", stats)
    emit("log", f"Resolved {stats.scanned} track(s): "
                f"{stats.resolved_auto} auto, {stats.needs_review} to review")

    # -- 2. Organize (copy-only) -------------------------------------------
    from .. import organize as org
    org_items = org.items_from_review_items(items)
    ops = org.plan(org_items, dest)
    tags_by_dest = {}
    if tag_copies:
        for op, it in zip(ops, items):
            tags_by_dest[op.dest] = it.final_tags() if hasattr(it, "final_tags") else {}

    stats.stage = "organize"
    total = len(ops)
    emit("stage", {"name": "organize", "total": total})
    emit("log", f"{'Copying' if apply else 'Planning'} {total} file(s) into {dest} "
                f"(copy-only — originals untouched)")

    done = 0

    def _progress(op):
        nonlocal done
        done += 1
        if op.status in ("copied",):
            stats.copied += 1
        elif op.status == "renamed":
            stats.renamed += 1
        elif op.status == "skipped_dup":
            stats.duplicates += 1
        elif op.status == "error":
            stats.errors += 1
            emit("log", f"error: {op.note} ({op.src})")
        emit("progress", {"stage": "organize", "done": done, "total": total,
                          "label": op.album})
        emit("stats", stats)

    summary = org.execute(ops, dry_run=not apply, on_progress=_progress)
    stats.bytes = summary.get("bytes", 0)

    # -- 3. Optionally tag the COPIES (never originals) --------------------
    if tag_copies and apply:
        from ..writer import write_tags
        for op in ops:
            if op.status in ("copied", "renamed") and tags_by_dest.get(op.dest):
                try:
                    write_tags(op.dest, tags_by_dest[op.dest], dry_run=False)
                    stats.tagged_copies += 1
                except Exception as exc:
                    emit("log", f"tag-copy failed: {exc}")
        emit("log", f"tagged {stats.tagged_copies} copied file(s) in the new tree")

    stats.stage = "done"
    stats.done = True
    emit("stats", stats)
    emit("done", stats)
    return stats


def run_dashboard(source: str, dest: str, *, config=None, apply: bool = False,
                  tag_copies: bool = False) -> None:
    """Launch the immich-go-styled Textual dashboard (falls back to text)."""
    try:
        from .dashboard_app import ImmichDashboard
    except Exception as exc:   # textual not installed -> headless text runner
        print(f"[note] TUI unavailable ({exc}); running headless.")
        return _run_headless(source, dest, config=config, apply=apply, tag_copies=tag_copies)
    ImmichDashboard(source, dest, config=config, apply=apply, tag_copies=tag_copies).run()


def _run_headless(source, dest, *, config, apply, tag_copies):
    def on_event(kind, payload):
        if kind == "log":
            print(f"  · {payload}")
        elif kind == "done":
            s = payload
            print(f"\nDone — scanned {s.scanned}, copied {s.copied + s.renamed}, "
                  f"duplicates {s.duplicates}, errors {s.errors}.")
    run_workflow(source, dest, config=config, apply=apply,
                 tag_copies=tag_copies, on_event=on_event)
