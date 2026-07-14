"""Resolver gating tests — the tightened confidence/agreement gate.

Covers the design's core fix: a single responding source can no longer clear the
auto-write gate; two independent sources can; audio verification overrides the
source floor; ``_same_recording`` no longer treats missing singers as agreement
across dissimilar releases; ISRC / recording-MBID shortcuts settle identity.
"""

from audio_tagger.models import ArtistCredit, Candidate, InputTrack, ReleaseType
from audio_tagger.resolve import Resolver, _same_recording, count_agreeing_sources
from audio_tagger.verify import VerifyReport


def _cand(**kw):
    base = dict(source="test", title="Kesariya", release_title="Brahmastra",
                release_type=ReleaseType.SOUNDTRACK, film="Brahmastra",
                credits=[ArtistCredit("Arijit Singh", "singer")], match_score=0.9)
    base.update(kw)
    return Candidate(**base)


class _FakeSource:
    def __init__(self, name, cands):
        self.name = name
        self._c = cands

    def search(self, track):
        return list(self._c)


# --------------------------------------------------------------------------- #
# Single vs. two independent sources.
# --------------------------------------------------------------------------- #
def test_single_source_no_longer_auto_passes():
    s = _FakeSource("musicbrainz", [_cand(source="musicbrainz", recording_mbid="rec-1")])
    r = Resolver([s], min_independent_sources=2).resolve(InputTrack(path="x"))
    assert r.n_agreeing_sources == 1
    assert r.confidence >= 0.80          # high confidence...
    assert r.needs_review is True        # ...but held: only one independent source


def test_two_agreeing_sources_auto_pass():
    s1 = _FakeSource("musicbrainz", [_cand(source="musicbrainz", recording_mbid="rec-1")])
    s2 = _FakeSource("itunes", [_cand(source="itunes", recording_mbid="rec-1")])
    r = Resolver([s1, s2], min_independent_sources=2).resolve(InputTrack(path="x"))
    assert r.n_agreeing_sources == 2
    assert r.confidence >= 0.80
    assert r.needs_review is False       # two independent sources -> auto-write


# --------------------------------------------------------------------------- #
# Audio verification overrides the source floor.
# --------------------------------------------------------------------------- #
def test_audio_verification_overrides_source_floor():
    cand = _cand(source="jiosaavn", recording_mbid="rec-1", raw={"language": "hindi"})
    s = _FakeSource("jiosaavn", [cand])

    def verifier(track, candidates):
        # Clip agrees with the candidate's declared language -> corroboration.
        return VerifyReport(language="hindi", confidence=0.9)

    r = Resolver([s], min_independent_sources=2, verifier=verifier).resolve(
        InputTrack(path="x"))
    assert r.n_agreeing_sources == 1
    assert r.audio_verified is True
    assert r.needs_review is False       # audio corroboration satisfies the floor


def test_audio_contradiction_does_not_verify():
    cand = _cand(source="jiosaavn", recording_mbid="rec-1", raw={"language": "hindi"})
    s = _FakeSource("jiosaavn", [cand])

    def verifier(track, candidates):
        return VerifyReport(language="punjabi", confidence=0.9)  # disagrees

    r = Resolver([s], min_independent_sources=2, verifier=verifier).resolve(
        InputTrack(path="x"))
    assert r.audio_verified is False
    assert r.needs_review is True


# --------------------------------------------------------------------------- #
# _same_recording tightening + identifier shortcuts.
# --------------------------------------------------------------------------- #
def test_missing_singers_title_only_across_dissimilar_releases_is_not_a_match():
    a = Candidate(source="a", title="Kesariya", release_title="Some Album",
                  release_type=ReleaseType.ALBUM, credits=[])
    b = Candidate(source="b", title="Kesariya", release_title="Party Mix Collection",
                  release_type=ReleaseType.COMPILATION, credits=[])
    assert _same_recording(a, b) is False


def test_missing_singers_but_same_release_is_a_match():
    a = Candidate(source="a", title="Kesariya", release_title="Brahmastra",
                  release_type=ReleaseType.SOUNDTRACK, credits=[])
    b = Candidate(source="b", title="Kesariya", release_title="Brahmastra",
                  release_type=ReleaseType.SOUNDTRACK, credits=[])
    assert _same_recording(a, b) is True


def test_isrc_shortcut_matches_across_different_titles():
    a = Candidate(source="a", title="Kesariya", release_title="X",
                  release_type=ReleaseType.ALBUM, raw={"isrc": "IN-S18-22-61234"})
    b = Candidate(source="b", title="Completely Different Name", release_title="Y",
                  release_type=ReleaseType.SINGLE, raw={"isrc": "INS182261234"})
    assert _same_recording(a, b) is True


def test_isrc_shortcut_distinguishes_different_recordings():
    a = Candidate(source="a", title="Kesariya", release_title="X",
                  release_type=ReleaseType.ALBUM, raw={"isrc": "INS182261234"})
    b = Candidate(source="b", title="Kesariya", release_title="X",
                  release_type=ReleaseType.ALBUM, raw={"isrc": "USABC0000001"})
    assert _same_recording(a, b) is False


def test_count_agreeing_excludes_agent():
    top = _cand(source="musicbrainz", recording_mbid="rec-1")
    same_db = _cand(source="itunes", recording_mbid="rec-1")
    agent = _cand(source="agent", recording_mbid="rec-1")
    # Two DBs + agent all agree, but the agent is not an independent source.
    assert count_agreeing_sources(top, [top, same_db, agent]) == 2
