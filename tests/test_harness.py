"""Unit tests for the decision logic — no network, all fakes.

Covers the parts that carry the design's weight:
  * release-preference heuristic beats compilations ("greatest hits" trap)
  * tag mapping follows the user's modeling rule
  * album reconciliation keeps a multi-artist film as ONE album
  * resolver agreement + web-search fallback on disagreement
"""

from audio_tagger.albumize import reconcile_albums
from audio_tagger.heuristics import looks_like_compilation, rank_candidates, release_preference_score
from audio_tagger.models import ArtistCredit, Candidate, InputTrack, ReleaseType, Resolution
from audio_tagger.resolve import Resolver, agreement_score
from audio_tagger.tagmap import build_tags


def _cand(**kw):
    base = dict(source="test", title="Song", release_title="Rel",
                release_type=ReleaseType.ALBUM, credits=[], match_score=0.8)
    base.update(kw)
    return Candidate(**base)


# --------------------------------------------------------------------------- #
# Heuristic: original soundtrack must beat the greatest-hits compilation.
# --------------------------------------------------------------------------- #
def test_soundtrack_beats_compilation():
    ost = _cand(release_title="Sholay", release_type=ReleaseType.SOUNDTRACK,
                film="Sholay", year=1975, match_score=0.85)
    comp = _cand(release_title="R.D. Burman's Greatest Hits",
                 release_type=ReleaseType.COMPILATION, year=2005, match_score=0.95)
    ranked = rank_candidates([comp, ost])
    assert ranked[0] is ost, "original soundtrack should rank above a higher-scored compilation"


def test_textual_compilation_markers_detected():
    for title in ["Superhits of Kishore", "Bollywood Party Mix Vol. 3",
                  "Best Of Lata", "Evergreen Golden Collection"]:
        assert looks_like_compilation(_cand(release_title=title)), title
    assert not looks_like_compilation(_cand(release_title="Sholay"))


def test_compilation_score_is_low():
    comp = _cand(release_type=ReleaseType.COMPILATION, release_title="Greatest Hits")
    assert release_preference_score(comp) <= 0.2


# --------------------------------------------------------------------------- #
# Tag mapping follows the user's modeling rule.
# --------------------------------------------------------------------------- #
def test_tagmap_roles_and_album_for_film_song():
    c = _cand(
        title="Mehbooba",
        release_type=ReleaseType.SOUNDTRACK,
        film="Sholay",
        credits=[
            ArtistCredit("R.D. Burman", "singer"),
            ArtistCredit("R.D. Burman", "composer"),
            ArtistCredit("Anand Bakshi", "lyricist"),
        ],
    )
    tags = build_tags(c)
    assert tags["artist"] == ["R.D. Burman"]                 # playback singer
    assert tags["composer"] == ["R.D. Burman"]               # music director
    assert tags["lyricist"] == ["Anand Bakshi"]              # lyricist
    assert tags["albumartist"] == ["R.D. Burman"]            # composer -> albumartist
    assert "Original Motion Picture Soundtrack" in tags["album"]


def test_tagmap_single_uses_title_as_album():
    c = _cand(title="Brown Munde", release_title="Brown Munde",
              release_type=ReleaseType.SINGLE,
              credits=[ArtistCredit("AP Dhillon", "singer")])
    tags = build_tags(c)
    assert tags["album"] == "Brown Munde"                    # true single -> title
    assert tags["artist"] == ["AP Dhillon"]


def test_tagmap_albumartist_falls_back_to_producer():
    c = _cand(title="X", release_type=ReleaseType.ALBUM, release_title="Album",
              credits=[ArtistCredit("Singer", "singer"),
                       ArtistCredit("Producer P", "producer")])
    tags = build_tags(c)
    assert tags["albumartist"] == ["Producer P"]             # no composer -> producer


# --------------------------------------------------------------------------- #
# Album reconciliation: a film with different singers stays ONE album.
# --------------------------------------------------------------------------- #
def _res(path, singer, composer, film):
    c = _cand(title=path, release_type=ReleaseType.SOUNDTRACK, film=film,
              credits=[ArtistCredit(singer, "singer"), ArtistCredit(composer, "composer")])
    r = Resolution(track=InputTrack(path=path), chosen=c, confidence=0.9,
                   agreement=1.0, tags=build_tags(c), needs_review=False)
    return r


def test_multi_artist_film_stays_one_album():
    rs = [
        _res("t1", "Kishore Kumar", "R.D. Burman", "Sholay"),
        _res("t2", "Lata Mangeshkar", "R.D. Burman", "Sholay"),
        _res("t3", "Manna Dey", "R.D. Burman", "Sholay"),
    ]
    reconcile_albums(rs)
    albums = {tuple(r.tags["albumartist"]) for r in rs}
    names = {r.tags["album"] for r in rs}
    keys = {r.tags["album_key"] for r in rs}
    assert len(albums) == 1, "single music director -> one album-artist for all tracks"
    assert albums == {("R.D. Burman",)}
    assert len(names) == 1 and len(keys) == 1
    # New semantics: varying playback singers across a film is NOT a compilation.
    assert all(r.tags["comp"] == 0 for r in rs), "varying singers must not set comp"


def test_multiple_music_directors_both_in_albumartist():
    # New semantics: two known music directors on one film are BOTH credited as
    # a multi-value ALBUMARTIST (first-seen order) — never "Various Artists" —
    # and the film is still comp=0 (not a compilation).
    rs = [
        _res("t1", "Arijit Singh", "Pritam", "Some Film"),
        _res("t2", "Shreya Ghoshal", "A.R. Rahman", "Some Film"),
    ]
    reconcile_albums(rs)
    assert all(r.tags["albumartist"] == ["Pritam", "A.R. Rahman"] for r in rs)
    assert not any(r.tags["albumartist"] == ["Various Artists"] for r in rs)
    assert all(r.tags["comp"] == 0 for r in rs)


# --------------------------------------------------------------------------- #
# Resolver: agreement, and web-search fallback on disagreement.
# --------------------------------------------------------------------------- #
class _FakeSource:
    def __init__(self, name, cands):
        self.name = name
        self._c = cands

    def search(self, track):
        return self._c


def test_agreement_high_when_sources_concur():
    a = _cand(source="musicbrainz", recording_mbid="rec-1")
    b = _cand(source="itunes", recording_mbid="rec-1")
    assert agreement_score(a, [a, b]) == 1.0


def test_agreement_low_when_sources_conflict():
    a = _cand(source="musicbrainz", title="Song A", recording_mbid="rec-1")
    b = _cand(source="itunes", title="Totally Different", recording_mbid="rec-9")
    assert agreement_score(a, [a, b]) == 0.5


class _FakeFallback:
    name = "websearch"
    def __init__(self):
        self.called = False
    def search(self, track, hints=None):
        self.called = True
        return [_cand(source="websearch", title="Resolved", recording_mbid="rec-1",
                      release_type=ReleaseType.SOUNDTRACK, film="Film",
                      credits=[ArtistCredit("Singer", "singer")], match_score=0.5)]


def test_fallback_triggered_on_disagreement():
    s1 = _FakeSource("musicbrainz", [_cand(source="musicbrainz", title="A", recording_mbid="r1")])
    s2 = _FakeSource("itunes", [_cand(source="itunes", title="B", recording_mbid="r2")])
    fb = _FakeFallback()
    r = Resolver([s1, s2], websearch_fallback=fb, agreement_floor=0.9).resolve(InputTrack(path="x"))
    assert fb.called, "disagreeing sources should trigger the web-search tie-breaker"
    assert r.chosen is not None


def test_fallback_triggered_when_no_structured_results():
    fb = _FakeFallback()
    r = Resolver([], websearch_fallback=fb).resolve(InputTrack(path="x"))
    assert fb.called and r.used_websearch_fallback
