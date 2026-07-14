"""The Textual review UI.

A two-pane interactive reviewer over a :class:`~audio_tagger.tui.state.ReviewSession`:

* **Queue pane** (left): every track flagged for review, each with a status
  glyph, its confidence, and a short reason.
* **Detail pane** (right): the current file's tags beside each candidate
  (title, album/film, year, singers, composers, lyricist, type, source, and
  match/agreement scores), the resolver's ``reasons`` log, and the audio-verify
  report if one ran. A candidate list drives which candidate is highlighted.
* **Steering box** (bottom): free-text guidance. Submitting runs
  :meth:`ReviewItem.steer` in a background worker (with a spinner), which calls
  ``AgentJudge.search(track, hints=ranked, steer=<text>)`` and merges the
  returned, confidence-capped candidates in for human approval.

Key bindings: ``a`` accept the highlighted candidate, ``e`` edit its fields,
``s`` skip, ``p`` play a clip (ffplay), ``w`` write all accepted, ``q`` quit
(saving the session to ``review.json`` so it can be resumed).

All pipeline work is delegated: :func:`audio_tagger.writer.write_tags` writes,
:func:`audio_tagger.tagmap.build_tags` (via ``ReviewItem.final_tags``) maps tags,
and :class:`~audio_tagger.agent.loop.AgentJudge` judges. ``textual`` is imported
at module load, but this module is only imported from
:func:`audio_tagger.tui.run_tui`, so the package still imports without it.
"""

from __future__ import annotations

import os
import subprocess

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    Static,
)

from ..writer import write_tags
from .state import (
    STATUS_ACCEPTED,
    STATUS_GLYPHS,
    ReviewItem,
    ReviewSession,
    save_session,
    short_reason,
)

_EDIT_FIELDS = [
    ("title", "Title", False),
    ("album", "Album", False),
    ("albumartist", "Album artist(s)", True),
    ("artist", "Singer(s)", True),
    ("composer", "Composer(s)", True),
    ("lyricist", "Lyricist(s)", True),
    ("year", "Year", False),
    ("grouping", "Film", False),
]


def _join_multi(value) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(x) for x in value)
    return "" if value is None else str(value)


def _split_multi(text: str) -> list[str]:
    return [s.strip() for s in (text or "").split(";") if s.strip()]


def _names(cand, role_list_attr: str) -> str:
    try:
        return ", ".join(getattr(cand, role_list_attr)) or "—"
    except Exception:
        return "—"


class EditScreen(ModalScreen):
    """A modal form to hand-edit the final tag fields before accepting."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, item: ReviewItem):
        super().__init__()
        self._item = item
        self._tags = item.final_tags()
        self._inputs: dict[str, Input] = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="edit-box"):
            yield Label("Edit tags — Enter/Save to apply, Esc to cancel", id="edit-title")
            for key, label, _multi in _EDIT_FIELDS:
                yield Label(label)
                inp = Input(value=_join_multi(self._tags.get(key)), id=f"edit-{key}")
                self._inputs[key] = inp
                yield inp
            with Horizontal(id="edit-buttons"):
                yield Button("Save", variant="success", id="edit-save")
                yield Button("Cancel", variant="default", id="edit-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edit-save":
            self._save()
        else:
            self.action_cancel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._save()

    def _save(self) -> None:
        tags = dict(self._tags)
        for key, _label, multi in _EDIT_FIELDS:
            raw = self._inputs[key].value
            if multi:
                tags[key] = _split_multi(raw)
            elif key == "year":
                tags[key] = raw.strip()
            else:
                tags[key] = raw.strip()
            if not tags[key]:
                tags.pop(key, None)
        self.dismiss(tags)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ReviewApp(App):
    """Textual application driving one :class:`ReviewSession`."""

    CSS = """
    #queue { width: 38%; border: round $primary; }
    #detail { border: round $secondary; padding: 0 1; }
    #candidates { height: auto; max-height: 40%; border: round $accent; }
    #steer { dock: bottom; }
    #spinner { dock: bottom; height: 1; display: none; }
    #edit-box { width: 70%; height: 80%; border: thick $primary; background: $panel; padding: 1 2; }
    #edit-buttons { height: auto; align: center middle; padding-top: 1; }
    Static { padding: 0 1; }
    """

    BINDINGS = [
        ("a", "accept", "Accept"),
        ("e", "edit", "Edit"),
        ("s", "skip", "Skip"),
        ("p", "play", "Play"),
        ("w", "write_all", "Write all"),
        ("q", "quit_save", "Quit"),
    ]

    def __init__(self, session: ReviewSession, judge=None, dry_run: bool = True):
        super().__init__()
        self.session = session
        self.judge = judge
        self.dry_run = dry_run
        self._queue: list[ReviewItem] = []
        self._current: ReviewItem | None = None

    # -- layout -------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            yield ListView(id="queue")
            with VerticalScroll(id="detail"):
                yield Static(id="filetags")
                yield Static(id="table")
                yield ListView(id="candidates")
                yield Static(id="reasons")
                yield Static(id="audio")
        yield Input(placeholder="Steer the agent, e.g. 'film version from Rockstar (2011), singer Mohit Chauhan'", id="steer")
        yield LoadingIndicator(id="spinner")
        yield Footer()

    def on_mount(self) -> None:
        mode = "DRY-RUN" if self.dry_run else "APPLY"
        self.title = "audio-tagger review"
        self.sub_title = f"{mode} · {len(self.session.queue_items())} to review"
        self._refresh_queue(select=0)

    # -- queue --------------------------------------------------------------
    def _refresh_queue(self, select: int | None = None) -> None:
        self._queue = self.session.queue_items()
        lv = self.query_one("#queue", ListView)
        keep = lv.index if select is None else select
        lv.clear()
        for item in self._queue:
            lv.append(ListItem(Label(self._queue_line(item))))
        if self._queue:
            idx = 0 if keep is None else max(0, min(keep, len(self._queue) - 1))
            lv.index = idx
            self._show(self._queue[idx])

    def _queue_line(self, item: ReviewItem) -> str:
        glyph = STATUS_GLYPHS.get(item.status, "○")
        return f"{glyph} {item.confidence:.2f}  {item.title()}  · {short_reason(item)}"

    def _update_queue_row(self, item: ReviewItem) -> None:
        if item not in self._queue:
            return
        lv = self.query_one("#queue", ListView)
        row = self._queue.index(item)
        try:
            lv.children[row].query_one(Label).update(self._queue_line(item))
        except Exception:
            self._refresh_queue()

    # -- detail -------------------------------------------------------------
    def _show(self, item: ReviewItem) -> None:
        self._current = item
        self.query_one("#filetags", Static).update(self._render_filetags(item))
        self._render_table(item)
        self._populate_candidates(item)
        self.query_one("#reasons", Static).update(self._render_reasons(item))
        self.query_one("#audio", Static).update(self._render_audio(item))

    def _render_filetags(self, item: ReviewItem) -> Text:
        t = item.track
        dur = f"{t.duration_sec:.0f}s" if t.duration_sec else "?"
        txt = Text()
        txt.append("FILE  ", style="bold")
        txt.append(os.path.basename(t.path) + "\n")
        txt.append(
            f"  existing: title={t.existing_title or '—'} · "
            f"artist={t.existing_artist or '—'} · album={t.existing_album or '—'} · {dur}\n"
        )
        txt.append(
            f"  status={item.status}  conf={item.confidence:.2f}  "
            f"agreement={item.agreement:.2f}  sources={item.n_agreeing_sources}"
        )
        if item.audio_verified:
            txt.append("  audio-verified", style="green")
        if item.agent_used:
            txt.append("  agent-used", style="magenta")
        return txt

    def _render_table(self, item: ReviewItem) -> None:
        table = Table(expand=True, show_lines=False, pad_edge=False)
        table.add_column("field", style="bold", no_wrap=True)
        for i, c in enumerate(item.candidates):
            head = f"[{i}] {c.source}"
            if i == item.selected_index:
                head = f"» {head}"
            style = "reverse" if i == item.selected_index else ""
            table.add_column(head, style=style, overflow="fold")

        def row(label, fn):
            table.add_row(label, *[fn(c) for c in item.candidates])

        row("title", lambda c: c.title or "—")
        row("album/film", lambda c: c.film or c.release_title or "—")
        row("year", lambda c: str(c.year) if c.year else "—")
        row("type", lambda c: c.release_type.value if hasattr(c.release_type, "value") else str(c.release_type))
        row("singers", lambda c: _names(c, "singers"))
        row("composers", lambda c: _names(c, "composers"))
        row("lyricist", lambda c: _names(c, "lyricists"))
        row("match", lambda c: f"{c.match_score:.2f}")
        if not item.candidates:
            table.add_row("(no candidates — steer the agent below)")
        self.query_one("#table", Static).update(table)

    def _populate_candidates(self, item: ReviewItem) -> None:
        lv = self.query_one("#candidates", ListView)
        lv.clear()
        for i, c in enumerate(item.candidates):
            label = f"[{i}] {c.source} · {c.title or '—'} — {c.film or c.release_title or '—'} ({c.match_score:.2f})"
            lv.append(ListItem(Label(label)))
        if item.candidates:
            lv.index = max(0, min(item.selected_index, len(item.candidates) - 1))

    def _render_reasons(self, item: ReviewItem) -> Text:
        txt = Text()
        txt.append("REASONS\n", style="bold")
        for line in item.reasons or ["(none)"]:
            txt.append(f"  · {line}\n")
        if item.steer_log:
            txt.append("STEERS\n", style="bold")
            for s in item.steer_log:
                txt.append(f"  › {s}\n", style="cyan")
        return txt

    def _render_audio(self, item: ReviewItem) -> Text:
        txt = Text()
        txt.append("AUDIO VERIFY\n", style="bold")
        if not item.audio_report:
            txt.append("  (not run)")
            return txt
        for k, v in item.audio_report.items():
            txt.append(f"  {k}: {v}\n")
        return txt

    # -- events -------------------------------------------------------------
    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        lv = event.list_view
        idx = lv.index
        if idx is None:
            return
        if lv.id == "queue":
            if 0 <= idx < len(self._queue):
                self._show(self._queue[idx])
        elif lv.id == "candidates" and self._current is not None:
            if 0 <= idx < len(self._current.candidates):
                self._current.selected_index = idx
                self._render_table(self._current)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "steer":
            return
        text = event.value.strip()
        event.input.value = ""
        if not self._current:
            self.notify("no track selected", severity="warning")
            return
        if self.judge is None:
            self.notify("agent unavailable (no LLM configured)", severity="warning")
            return
        self._set_busy(True)
        self._run_steer(self._current, text)

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#spinner", LoadingIndicator).display = busy

    @work(thread=True, exclusive=True, group="steer")
    def _run_steer(self, item: ReviewItem, text: str) -> None:
        try:
            added = item.steer(self.judge, text)
        except Exception as exc:  # never let a bad agent turn crash the UI
            self.call_from_thread(self._after_steer, item, None, str(exc))
            return
        self.call_from_thread(self._after_steer, item, added, None)

    def _after_steer(self, item: ReviewItem, added, error) -> None:
        self._set_busy(False)
        if error is not None:
            self.notify(f"agent error: {error}", severity="error")
            return
        if item is self._current:
            self._show(item)
        self._update_queue_row(item)
        n = len(added or [])
        if n:
            self.notify(f"agent added {n} candidate(s)")
        else:
            self.notify("agent proposed nothing new (abstained)")

    # -- actions ------------------------------------------------------------
    def action_accept(self) -> None:
        item = self._current
        if not item or not item.candidates:
            self.notify("nothing to accept", severity="warning")
            return
        lv = self.query_one("#candidates", ListView)
        idx = lv.index if lv.index is not None else item.selected_index
        item.accept(idx)
        self._update_queue_row(item)
        c = item.selected_candidate()
        self.notify(f"accepted [{idx}] {c.title if c else ''}")

    def action_edit(self) -> None:
        item = self._current
        if not item:
            return

        def _done(tags) -> None:
            if tags is not None:
                item.edited_tags = tags
                self.notify("edits saved (accept to apply)")
                if item is self._current:
                    self.query_one("#reasons", Static).update(self._render_reasons(item))

        self.push_screen(EditScreen(item), _done)

    def action_skip(self) -> None:
        item = self._current
        if not item:
            return
        item.skip()
        self._update_queue_row(item)
        self.notify("skipped")

    def action_play(self) -> None:
        item = self._current
        if not item:
            return
        self._play_clip(item.track.path, item.track.duration_sec)

    @work(thread=True, exclusive=True, group="play")
    def _play_clip(self, path: str, duration) -> None:
        seconds = 25
        start = max(0.0, (duration / 2 - seconds / 2)) if duration else 45.0
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                 "-ss", str(start), "-t", str(seconds), path],
                check=False,
            )
        except FileNotFoundError:
            self.call_from_thread(self.notify, "ffplay not found", severity="warning")
        except Exception as exc:
            self.call_from_thread(self.notify, f"play failed: {exc}", severity="error")

    def action_write_all(self) -> None:
        accepted = self.session.accepted_items()
        if not accepted:
            self.notify("no accepted tracks to write", severity="warning")
            return
        written = 0
        errors = 0
        for item in accepted:
            try:
                res = write_tags(item.track.path, item.final_tags(), dry_run=self.dry_run)
                if res.get("error"):
                    errors += 1
                else:
                    written += 1
            except Exception:
                errors += 1
        mode = "planned (dry-run)" if self.dry_run else "written"
        msg = f"{written} {mode}"
        if errors:
            msg += f", {errors} failed"
        self.notify(msg, severity="error" if errors else "information")

    def action_quit_save(self) -> None:
        try:
            path = save_session(self.session)
            self.notify(f"session saved -> {path}")
        except Exception as exc:
            self.notify(f"save failed: {exc}", severity="error")
        self.exit()
