"""Dashboard tests: the pure workflow controller (headless) + a Textual smoke test."""

import asyncio
import json
import os

from audio_tagger.tui.dashboard import WorkflowStats, run_workflow


def _make_run_json(tmp_path):
    """A tiny tag --json-out document over three real (fake-bytes) files."""
    src = tmp_path / "src"
    src.mkdir()
    rows = []
    for i, (album, title) in enumerate([
        ("Brahmastra (OST)", "Kesariya"),
        ("Brahmastra (OST)", "Deva Deva"),
        ("Kesari", "Deh Shiva (Male Version)"),
    ]):
        p = src / f"track{i}.mp3"
        p.write_bytes(f"audio-{i}".encode())
        rows.append({
            "track": {"path": str(p), "existing_title": title, "existing_album": album},
            "candidates": [{"source": "x", "title": title, "release_title": album,
                            "release_type": "soundtrack", "film": album}],
            "chosen_index": 0, "tags": {"album": album, "title": title},
            "needs_review": i == 2,   # one track flagged for review
        })
    rj = tmp_path / "run.json"
    rj.write_text(json.dumps(rows), encoding="utf-8")
    return str(rj), str(src)


# --------------------------------------------------------------------------- #
# controller (no Textual)
# --------------------------------------------------------------------------- #
def test_controller_dry_run_counts(tmp_path):
    run_json, _ = _make_run_json(tmp_path)
    events = []
    stats = run_workflow(run_json, str(tmp_path / "lib"), apply=False,
                         on_event=lambda k, p: events.append((k, p)))
    assert stats.scanned == 3
    assert stats.resolved_auto == 2 and stats.needs_review == 1
    assert stats.copied == 0                       # dry-run copies nothing
    assert not os.path.exists(str(tmp_path / "lib"))
    assert any(k == "done" for k, _ in events)
    assert any(k == "stage" and p.get("name") == "organize" for k, p in events)


def test_controller_apply_copies_and_keeps_originals(tmp_path):
    run_json, src = _make_run_json(tmp_path)
    lib = str(tmp_path / "lib")
    stats = run_workflow(run_json, lib, apply=True)
    assert stats.copied == 3
    assert os.path.isfile(os.path.join(lib, "Brahmastra (OST)", "Kesariya.mp3"))
    assert os.path.isfile(os.path.join(lib, "Kesari", "Deh Shiva (Male Version).mp3"))
    # originals all still present and unmodified
    assert len(os.listdir(src)) == 3


def test_stats_tiles_shape():
    tiles = WorkflowStats(scanned=5, resolved_auto=4, needs_review=1, copied=4).as_tiles()
    labels = [t[0] for t in tiles]
    assert labels == ["Scanned", "Auto-tagged", "Needs review", "Copied", "Duplicates", "Errors"]


# --------------------------------------------------------------------------- #
# Textual app smoke test (composes + runs to completion headlessly)
# --------------------------------------------------------------------------- #
def test_dashboard_app_composes_and_completes(tmp_path):
    from audio_tagger.tui.dashboard_app import ImmichDashboard
    from textual.widgets import ProgressBar, Static

    run_json, _ = _make_run_json(tmp_path)

    async def _run():
        app = ImmichDashboard(run_json, str(tmp_path / "lib"), apply=False)
        async with app.run_test() as pilot:
            # banner + all six stat tiles + both progress bars are present
            assert app.query_one("#banner", Static)
            for slug in ["scanned", "auto-tagged", "needs-review", "copied", "duplicates", "errors"]:
                assert app.query_one(f"#tile-{slug}")
            assert app.query_one("#pb-organize", ProgressBar)
            # let the worker finish, then the scanned tile should read 3
            for _ in range(40):
                await pilot.pause(0.05)
                if app._tiles["scanned"].current_value == 3:
                    break
            assert app._tiles["scanned"].current_value == 3
            assert app._tiles["auto-tagged"].current_value == 2

    asyncio.run(_run())
