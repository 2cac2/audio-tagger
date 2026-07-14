"""Review CSV round-trip — write_queue -> (approve) -> read_approvals.

Verifies the queue preserves the full write provenance: year, grouping (film),
the compilation flag, both MusicBrainz ids, and the multi-value credit lists.
"""

import csv

from audio_tagger.models import Candidate, InputTrack, ReleaseType, Resolution
from audio_tagger.review import read_approvals, write_queue


def _resolution():
    c = Candidate(source="musicbrainz", title="Kesariya", release_title="Brahmastra",
                  release_type=ReleaseType.SOUNDTRACK, film="Brahmastra",
                  recording_mbid="rec-1", release_mbid="rel-1")
    tags = {
        "title": "Kesariya",
        "artist": ["Arijit Singh"],
        "albumartist": ["Pritam"],
        "album": "Brahmastra (Original Motion Picture Soundtrack)",
        "grouping": "Brahmastra",
        "year": 2022,
        "composer": ["Pritam"],
        "lyricist": ["Amitabh Bhattacharya"],
        "comp": 1,
        "musicbrainz_trackid": "rec-1",
        "musicbrainz_albumid": "rel-1",
    }
    return Resolution(track=InputTrack(path="/music/kesariya.mp3"), chosen=c,
                      confidence=0.5, agreement=0.5, tags=tags, needs_review=True,
                      reasons=["held for review"])


def _approve_all(path):
    """Flip every row's approved flag to 1 (simulating a human edit)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames
    for row in rows:
        row["approved"] = "1"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_review_roundtrip_preserves_provenance(tmp_path):
    path = str(tmp_path / "review.csv")
    n = write_queue([_resolution()], path)
    assert n == 1

    _approve_all(path)
    approvals = read_approvals(path)

    assert "/music/kesariya.mp3" in approvals
    fields = approvals["/music/kesariya.mp3"]
    assert fields["year"] == 2022
    assert fields["grouping"] == "Brahmastra"
    assert fields["comp"] == 1
    assert fields["musicbrainz_trackid"] == "rec-1"
    assert fields["musicbrainz_albumid"] == "rel-1"
    assert fields["composer"] == ["Pritam"]
    assert fields["lyricist"] == ["Amitabh Bhattacharya"]
    assert fields["artist"] == ["Arijit Singh"]
    assert fields["albumartist"] == ["Pritam"]
    assert fields["title"] == "Kesariya"


def test_unapproved_rows_are_not_returned(tmp_path):
    path = str(tmp_path / "review.csv")
    write_queue([_resolution()], path)   # approved stays 0
    assert read_approvals(path) == {}
