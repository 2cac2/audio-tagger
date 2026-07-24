"""Organizer tests — the copy-only safety guarantees are the point.

No real audio needed: copy/compare work on bytes, so plain files with audio
extensions exercise the same paths.
"""

import os

from audio_tagger.organize import (
    CopyOp, OrgItem, execute, items_from_review_items, plan, sanitize_segment,
)


def _write(path, data=b"audio-bytes"):
    with open(path, "wb") as fh:
        fh.write(data)
    return path


# --------------------------------------------------------------------------- #
# planning + sanitization
# --------------------------------------------------------------------------- #
def test_plan_builds_album_over_title(tmp_path):
    src = _write(str(tmp_path / "a.mp3"))
    ops = plan([OrgItem(src=src, album="Brahmastra (OST)", title="Kesariya")], str(tmp_path / "lib"))
    assert ops[0].dest.endswith(os.path.join("lib", "Brahmastra (OST)", "Kesariya.mp3"))


def test_plan_track_number_prefix(tmp_path):
    src = _write(str(tmp_path / "a.mp3"))
    ops = plan([OrgItem(src=src, album="Kesari", title="Deh Shiva", track_no=3)], str(tmp_path / "lib"))
    assert os.path.basename(ops[0].dest) == "03 Deh Shiva.mp3"


def test_sanitize_strips_illegal_and_preserves_unicode():
    assert sanitize_segment('A/B:C*?"') == "A_B_C___"   # 5 illegal chars -> 5 underscores
    assert sanitize_segment("केसरिया") == "केसरिया"       # Devanagari preserved
    assert sanitize_segment("  ") == "Unknown"           # empty -> fallback


def test_plan_resolves_in_plan_collisions(tmp_path):
    s1 = _write(str(tmp_path / "1.mp3"))
    s2 = _write(str(tmp_path / "2.mp3"), b"different")
    ops = plan([OrgItem(s1, "Album", "Song"), OrgItem(s2, "Album", "Song")], str(tmp_path / "lib"))
    assert os.path.basename(ops[0].dest) == "Song.mp3"
    assert os.path.basename(ops[1].dest) == "Song (2).mp3"   # no two copies to one path


# --------------------------------------------------------------------------- #
# execute — copy-only
# --------------------------------------------------------------------------- #
def test_dry_run_writes_nothing(tmp_path):
    src = _write(str(tmp_path / "a.mp3"))
    dest_root = str(tmp_path / "lib")
    ops = plan([OrgItem(src, "Album", "Song")], dest_root)
    summary = execute(ops, dry_run=True)
    assert summary["dry_run"] == 1 and summary["copied"] == 0
    assert not os.path.exists(dest_root)                 # nothing created


def test_apply_copies_and_leaves_original_untouched(tmp_path):
    src = _write(str(tmp_path / "a.mp3"), b"original-bytes")
    before_mtime = os.path.getmtime(src)
    dest_root = str(tmp_path / "lib")
    ops = plan([OrgItem(src, "Sholay", "Mehbooba")], dest_root)
    summary = execute(ops, dry_run=False)
    assert summary["copied"] == 1
    dest = os.path.join(dest_root, "Sholay", "Mehbooba.mp3")
    assert os.path.isfile(dest)
    # original still there, unchanged, and a real copy (distinct inode)
    assert os.path.isfile(src)
    assert open(src, "rb").read() == b"original-bytes"
    assert os.path.getmtime(src) == before_mtime
    assert open(dest, "rb").read() == b"original-bytes"
    assert os.path.samefile(src, dest) is False


def test_identical_destination_is_skipped_not_overwritten(tmp_path):
    src = _write(str(tmp_path / "a.mp3"), b"same")
    dest_root = str(tmp_path / "lib")
    ops = plan([OrgItem(src, "Album", "Song")], dest_root)
    execute(ops, dry_run=False)                          # first copy
    ops2 = plan([OrgItem(src, "Album", "Song")], dest_root)
    summary = execute(ops2, dry_run=False)               # second run
    assert summary["skipped_dup"] == 1 and summary["copied"] == 0


def test_different_file_at_destination_is_renamed_never_overwritten(tmp_path):
    dest_root = str(tmp_path / "lib")
    # pre-existing DIFFERENT file already sitting at the target path
    target_dir = os.path.join(dest_root, "Album")
    os.makedirs(target_dir)
    _write(os.path.join(target_dir, "Song.mp3"), b"PRE-EXISTING")
    src = _write(str(tmp_path / "a.mp3"), b"new-content")
    ops = plan([OrgItem(src, "Album", "Song")], dest_root)
    summary = execute(ops, dry_run=False)
    assert summary["renamed"] == 1
    assert open(os.path.join(target_dir, "Song.mp3"), "rb").read() == b"PRE-EXISTING"  # untouched
    assert os.path.isfile(os.path.join(target_dir, "Song (2).mp3"))


def test_src_equal_dest_refused(tmp_path):
    src = _write(str(tmp_path / "x.mp3"))
    op = CopyOp(src=src, dest=src, album="Album")
    summary = execute([op], dry_run=False)
    assert summary["errors"] == 1 and op.status == "error"


def test_missing_source_is_error_not_crash(tmp_path):
    op = CopyOp(src=str(tmp_path / "gone.mp3"), dest=str(tmp_path / "lib/A/x.mp3"), album="A")
    summary = execute([op], dry_run=False)
    assert summary["errors"] == 1


# --------------------------------------------------------------------------- #
# item builders
# --------------------------------------------------------------------------- #
def test_items_from_review_items_uses_resolved_album():
    class _T:
        path = "/music/x.mp3"
        existing_album = None
        existing_title = None

    class _Item:
        track = _T()
        def final_tags(self):
            return {"album": "Brahmastra (Original Motion Picture Soundtrack)", "title": "Kesariya"}

    items = items_from_review_items([_Item()])
    assert items[0].album.startswith("Brahmastra") and items[0].title == "Kesariya"
    assert items[0].src == "/music/x.mp3"
