"""immich-go-styled Textual dashboard for the tag → organize workflow.

Visual language borrowed from immich-go's terminal UI: a bordered accent banner,
a row of colored stat tiles with big numbers, per-stage progress bars, and a live
scrolling log — on a dark surface with a violet accent. This module imports
``textual`` at load time, so it is imported *lazily* by
:func:`audio_tagger.tui.dashboard.run_dashboard`; the controller in
:mod:`audio_tagger.tui.dashboard` stays dependency-free and unit-testable.
"""

from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Label, ProgressBar, RichLog, Static

from .dashboard import WorkflowStats, run_workflow

_TILE_ORDER = ["scanned", "auto-tagged", "needs-review", "copied", "duplicates", "errors"]


def _slug(label: str) -> str:
    return label.lower().replace(" ", "-")


class StatTile(Vertical):
    """A single immich-go-style counter tile: big number over a dim label."""

    def __init__(self, label: str, kind: str) -> None:
        super().__init__(classes=f"tile {kind}", id=f"tile-{_slug(label)}")
        self._label = label
        self.current_value = 0

    def compose(self) -> ComposeResult:
        yield Label("0", classes="tile-value", id=f"{self.id}-val")
        yield Label(self._label.upper(), classes="tile-label")

    def set_value(self, value: int) -> None:
        self.current_value = value
        self.query_one(f"#{self.id}-val", Label).update(str(value))


class ImmichDashboard(App):
    """Full-workflow dashboard: Analyze → Organize, styled like immich-go."""

    CSS = """
    Screen { background: #14141b; color: #e7e7ef; }

    #banner {
        height: 3; padding: 0 2; margin: 1 2 0 2;
        background: #1c1c27; color: #a277ff; text-style: bold;
        border: round #a277ff; content-align: left middle;
    }
    #subtitle { height: 1; margin: 0 2 1 2; color: #8b8ba7; }

    #tiles { height: 6; margin: 0 1; }
    .tile {
        width: 1fr; height: 100%; margin: 0 1; padding: 1 1;
        background: #1c1c27; border: round #2c2c3a; content-align: center middle;
    }
    .tile-value { width: 100%; text-align: center; text-style: bold; color: #e7e7ef; }
    .tile-label { width: 100%; text-align: center; color: #6c6c86; }
    .tile.ok    .tile-value { color: #3ddc84; }
    .tile.warn  .tile-value { color: #ffb454; }
    .tile.err   .tile-value { color: #ff5370; }
    .tile.info  .tile-value { color: #7cd1ff; }
    .tile.muted .tile-value { color: #8b8ba7; }

    .stage-row { height: 3; margin: 0 2; }
    .stage-label { width: 14; color: #a277ff; text-style: bold; content-align: left middle; }
    ProgressBar { width: 1fr; }
    Bar > .bar--bar { color: #a277ff; }
    Bar > .bar--complete { color: #3ddc84; }

    #log {
        margin: 1 2; padding: 0 1; height: 1fr;
        background: #1c1c27; border: round #2c2c3a; color: #b7b7cf;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, source: str, dest: str, *, config=None,
                 apply: bool = False, tag_copies: bool = False) -> None:
        super().__init__()
        self.source = source
        self.dest = dest
        self.config = config
        self.apply = apply
        self.tag_copies = tag_copies
        self._tiles: dict[str, StatTile] = {}

    def compose(self) -> ComposeResult:
        yield Static("◆ AUDIO-TAGGER   ·   organize dashboard", id="banner")
        mode = "APPLY" if self.apply else "DRY-RUN"
        yield Static(
            f"{self.source}  →  {self.dest}    [{mode}]    "
            "COPY-ONLY · your original files are never moved, deleted, or modified",
            id="subtitle",
        )

        with Horizontal(id="tiles"):
            for label, _value, kind in WorkflowStats().as_tiles():
                tile = StatTile(label, kind)
                self._tiles[_slug(label)] = tile
                yield tile

        yield Horizontal(Label("ANALYZE", classes="stage-label"),
                         ProgressBar(id="pb-analyze", show_eta=False),
                         classes="stage-row")
        yield Horizontal(Label("ORGANIZE", classes="stage-label"),
                         ProgressBar(id="pb-organize", show_eta=False),
                         classes="stage-row")

        yield RichLog(id="log", highlight=False, markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        # Analyze runs as one blocking pipeline call -> show it as indeterminate.
        self.query_one("#pb-analyze", ProgressBar).update(total=None)
        self._run_pipeline()

    # -- worker -------------------------------------------------------------
    @work(thread=True, exclusive=True)
    def _run_pipeline(self) -> None:
        def on_event(kind, payload):
            self.call_from_thread(self._handle_event, kind, payload)
        try:
            run_workflow(self.source, self.dest, config=self.config,
                         apply=self.apply, tag_copies=self.tag_copies, on_event=on_event)
        except Exception as exc:   # surface, never crash the UI
            self.call_from_thread(self._handle_event, "log", f"[#ff5370]fatal: {exc}[/]")

    # -- UI-thread event handler -------------------------------------------
    def _handle_event(self, kind: str, payload) -> None:
        log = self.query_one("#log", RichLog)
        if kind == "log":
            log.write(payload)
        elif kind == "stage":
            if payload.get("name") == "organize":
                self.query_one("#pb-organize", ProgressBar).update(
                    total=max(1, payload.get("total", 1)), progress=0)
                # analyze is finished by the time organize starts
                self.query_one("#pb-analyze", ProgressBar).update(total=1, progress=1)
        elif kind == "progress":
            if payload.get("stage") == "organize":
                self.query_one("#pb-organize", ProgressBar).update(progress=payload["done"])
        elif kind == "stats":
            self._apply_stats(payload)
        elif kind == "done":
            self._apply_stats(payload)
            self.query_one("#pb-analyze", ProgressBar).update(total=1, progress=1)
            log.write("[#3ddc84]◆ workflow complete — press q to quit[/]")

    def _apply_stats(self, s: WorkflowStats) -> None:
        for label, value, _kind in s.as_tiles():
            tile = self._tiles.get(_slug(label))
            if tile is not None:
                tile.set_value(value)
