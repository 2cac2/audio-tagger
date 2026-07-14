"""Tests for credit normalization and multi-composer album accounting.

Covers ``audio_tagger.credits`` (canonical acts + cross-source merge) and the
album-level ALBUMARTIST / comp rules in ``audio_tagger.albumize``.
"""

from audio_tagger.albumize import reconcile_albums
from audio_tagger.credits import canonical_acts, merge_credits
from audio_tagger.models import ArtistCredit, Candidate, InputTrack, ReleaseType, Resolution
from audio_tagger.tagmap import build_tags


def _cand(**kw):
    base = dict(source="test", title="Song", release_title="Rel",
                release_type=ReleaseType.SOUNDTRACK, credits=[], match_score=0.8)
    base.update(kw)
    return Candidate(**base)


# --------------------------------------------------------------------------- #
# canonical_acts: fold known duos, guard unrelated same-first-name artists.
# --------------------------------------------------------------------------- #
def test_duo_members_fold_to_single_act():
    assert canonical_acts(["Vishal Dadlani", "Shekhar Ravjiani"]) == ["Vishal-Shekhar"]


def test_duo_folding_dedupes_and_keeps_order():
    # Both members plus the act name all collapse to one entry.
    acts = canonical_acts(["Shekhar Ravjiani", "Vishal Dadlani", "Vishal-Shekhar"])
    assert acts == ["Vishal-Shekhar"]


def test_unrelated_vishal_not_folded():
    # The guard: a lone "Vishal ..." who is not a Vishal-Shekhar member stays put.
    assert canonical_acts(["Vishal Mishra"]) == ["Vishal Mishra"]
    assert canonical_acts(["Vishal Bhardwaj"]) == ["Vishal Bhardwaj"]


def test_trio_folds():
    acts = canonical_acts(["Shankar Mahadevan", "Ehsaan Noorani", "Loy Mendonsa"])
    assert acts == ["Shankar-Ehsaan-Loy"]


# --------------------------------------------------------------------------- #
# merge_credits: union roles across sources, dedupe transliteration drift.
# --------------------------------------------------------------------------- #
def test_merge_credits_unions_roles():
    a = _cand(credits=[ArtistCredit("Arijit Singh", "singer")])
    b = _cand(credits=[ArtistCredit("Pritam", "composer"),
                       ArtistCredit("Amitabh Bhattacharya", "lyricist")])
    merged = merge_credits([a, b])
    by_role = {(c.role, c.name) for c in merged}
    assert ("singer", "Arijit Singh") in by_role
    assert ("composer", "Pritam") in by_role
    assert ("lyricist", "Amitabh Bhattacharya") in by_role


def test_merge_credits_dedupes_similar_names():
    # Same role, transliteration-similar spelling -> one credit (first wins).
    a = _cand(credits=[ArtistCredit("Arijit Singh", "singer")])
    b = _cand(credits=[ArtistCredit("Arijit Sing", "singer")])
    merged = merge_credits([a, b])
    singers = [c.name for c in merged if c.role == "singer"]
    assert singers == ["Arijit Singh"]


def test_merge_credits_backfills_mbid():
    a = _cand(credits=[ArtistCredit("Pritam", "composer", mbid=None)])
    b = _cand(credits=[ArtistCredit("Pritam", "composer", mbid="c1")])
    merged = merge_credits([a, b])
    composers = [c for c in merged if c.role == "composer"]
    assert len(composers) == 1
    assert composers[0].mbid == "c1"


# --------------------------------------------------------------------------- #
# Album-level accounting.
# --------------------------------------------------------------------------- #
def _res(path, singer, composer, film, rtype=ReleaseType.SOUNDTRACK, release_title="Rel"):
    c = _cand(title=path, release_type=rtype, film=film, release_title=release_title,
              credits=[ArtistCredit(singer, "singer"), ArtistCredit(composer, "composer")])
    return Resolution(track=InputTrack(path=path), chosen=c, confidence=0.9,
                      agreement=1.0, tags=build_tags(c), needs_review=False)


def test_two_composer_film_credits_both_not_various_artists():
    rs = [
        _res("t1", "Arijit Singh", "Pritam", "Anthology Film"),
        _res("t2", "Shreya Ghoshal", "Amit Trivedi", "Anthology Film"),
    ]
    reconcile_albums(rs)
    for r in rs:
        assert r.tags["albumartist"] == ["Pritam", "Amit Trivedi"]
        assert r.tags["albumartist"] != ["Various Artists"]
        assert r.tags["comp"] == 0


def test_duo_film_folds_to_single_albumartist():
    rs = [
        _res("t1", "Neha Kakkar", "Vishal Dadlani", "Duo Film"),
        _res("t2", "Benny Dayal", "Shekhar Ravjiani", "Duo Film"),
    ]
    reconcile_albums(rs)
    for r in rs:
        assert r.tags["albumartist"] == ["Vishal-Shekhar"]
        assert r.tags["comp"] == 0


def test_true_compilation_sets_comp_flag():
    rs = [
        _res("t1", "Kishore Kumar", "R.D. Burman", None,
             rtype=ReleaseType.COMPILATION, release_title="Greatest Hits"),
        _res("t2", "Kishore Kumar", "R.D. Burman", None,
             rtype=ReleaseType.COMPILATION, release_title="Greatest Hits"),
    ]
    reconcile_albums(rs)
    assert all(r.tags["comp"] == 1 for r in rs)
