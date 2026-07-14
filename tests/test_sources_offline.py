"""Offline source-adapter tests — HTTP monkeypatched to canned fixtures.

No network, no real ``requests`` / ``musicbrainzngs`` needed: a fake module is
injected into ``sys.modules`` so the lazy ``import requests`` / ``import
musicbrainzngs`` inside each adapter picks up our stub.
"""

import json
import os
import sys
import types

import pytest

from audio_tagger.models import InputTrack, ReleaseType

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as fh:
        return json.load(fh)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _install_requests(monkeypatch, get=None, post=None):
    fake = types.ModuleType("requests")
    if get is not None:
        fake.get = get
    if post is not None:
        fake.post = post
    monkeypatch.setitem(sys.modules, "requests", fake)


# --------------------------------------------------------------------------- #
# JioSaavn: film-as-album, composer from `music`, compilation demotion.
# --------------------------------------------------------------------------- #
def _jiosaavn_get_factory():
    search = _load("jiosaavn_search.json")
    filmi = _load("jiosaavn_song.json")
    comp = _load("jiosaavn_compilation.json")

    def fake_get(url, params=None, timeout=None, headers=None):
        if "/api/search/songs" in url:
            return FakeResponse(search)
        if "/api/songs/song1" in url:
            return FakeResponse(filmi)
        if "/api/songs/song2" in url:
            return FakeResponse(comp)
        return FakeResponse({}, status=404)

    return fake_get


def test_jiosaavn_maps_film_and_composer(monkeypatch):
    _install_requests(monkeypatch, get=_jiosaavn_get_factory())
    from audio_tagger.sources.jiosaavn import JioSaavnSource

    src = JioSaavnSource(base_url="http://localhost:3500")
    track = InputTrack(path="x", existing_title="Kesariya", existing_artist="Arijit Singh")
    cands = src.search(track)

    filmi = [c for c in cands if c.film]
    assert filmi, "expected a filmi candidate with the film as album"
    fc = filmi[0]
    assert fc.film == "Brahmastra"
    assert fc.release_title == "Brahmastra"
    assert fc.release_type == ReleaseType.SOUNDTRACK
    assert fc.composers == ["Pritam"]          # music director from `music`
    assert fc.singers == ["Arijit Singh"]
    assert fc.raw.get("language") == "hindi"


def test_jiosaavn_demotes_compilation_dupe(monkeypatch):
    _install_requests(monkeypatch, get=_jiosaavn_get_factory())
    from audio_tagger.sources.jiosaavn import JioSaavnSource

    src = JioSaavnSource(base_url="http://localhost:3500")
    track = InputTrack(path="x", existing_title="Kesariya", existing_artist="Arijit Singh")
    cands = src.search(track)

    comps = [c for c in cands if c.release_type == ReleaseType.COMPILATION]
    assert comps, "the 'Superhits' album must be demoted to a compilation"
    assert comps[0].film is None


def test_jiosaavn_match_score_varies_with_input(monkeypatch):
    _install_requests(monkeypatch, get=_jiosaavn_get_factory())
    from audio_tagger.sources.jiosaavn import JioSaavnSource

    src = JioSaavnSource(base_url="http://localhost:3500")
    good = src.search(InputTrack(path="x", existing_title="Kesariya",
                                 existing_artist="Arijit Singh"))
    bad = src.search(InputTrack(path="y", existing_title="Zzzz Nonsense Query",
                                existing_artist="Nobody At All"))
    good_film = [c for c in good if c.film][0]
    bad_film = [c for c in bad if c.film][0]
    assert good_film.match_score > bad_film.match_score


# --------------------------------------------------------------------------- #
# Deezer: ISRC exposed in raw, record_type mapped.
# --------------------------------------------------------------------------- #
def test_deezer_exposes_isrc(monkeypatch):
    search = _load("deezer_search.json")
    album = _load("deezer_album.json")
    track_detail = _load("deezer_track.json")

    def fake_get(url, params=None, timeout=None, headers=None):
        if url.startswith("https://api.deezer.com/search"):
            return FakeResponse(search)
        if "/album/" in url:
            return FakeResponse(album)
        if "/track/" in url:
            return FakeResponse(track_detail)
        return FakeResponse({}, status=404)

    _install_requests(monkeypatch, get=fake_get)
    from audio_tagger.sources.deezer import DeezerSource

    src = DeezerSource()
    cands = src.search(InputTrack(path="x", existing_title="Kesariya",
                                  existing_artist="Arijit Singh"))
    assert cands
    c = cands[0]
    assert c.title == "Kesariya"
    assert c.raw.get("isrc") == "INS182261234"
    assert c.release_type == ReleaseType.ALBUM
    assert c.year == 2022


def test_deezer_extracts_film_and_uses_neutral_role(monkeypatch):
    """Deezer '(From "X")' titles set the film + soundtrack type, and the lone
    top-level artist (a composer for filmi tracks) is a neutral 'performer',
    not a falsely-asserted singer."""
    search = {"data": [{
        "id": 111, "title": 'Kesariya (From "Brahmastra")',
        "artist": {"name": "Pritam"},               # Deezer's artist = music director
        "album": {"id": 222, "title": "Kesariya (From \"Brahmastra\")"},
    }]}

    def fake_get(url, params=None, timeout=None, headers=None):
        if url.startswith("https://api.deezer.com/search"):
            return FakeResponse(search)
        if "/album/" in url:
            return FakeResponse({"record_type": "single", "release_date": "2022-07-17"})
        if "/track/" in url:
            return FakeResponse({"isrc": "INS172203702", "release_date": "2022-07-17"})
        return FakeResponse({}, status=404)

    _install_requests(monkeypatch, get=fake_get)
    from audio_tagger.sources.deezer import DeezerSource

    c = DeezerSource().search(InputTrack(path="x", existing_title="Kesariya",
                                         existing_artist="Arijit Singh"))[0]
    assert c.film == "Brahmastra"
    assert c.title == "Kesariya"                     # film suffix stripped from title
    assert c.release_type == ReleaseType.SOUNDTRACK  # film recovered -> soundtrack
    assert c.credits[0].role == "performer"          # NOT "singer"
    assert c.raw.get("role_unverified") is True


# --------------------------------------------------------------------------- #
# YouTube: trust filter (label / Topic / verified only) + LLM parse of credits.
# --------------------------------------------------------------------------- #
class _FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL: a context manager returning canned entries."""
    entries: list = []

    def __init__(self, opts=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, query, download=False):
        return {"entries": list(self.__class__.entries)}


def _install_ytdlp(monkeypatch, entries):
    _FakeYDL.entries = entries
    fake = types.ModuleType("yt_dlp")
    fake.YoutubeDL = _FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)


class _FakeLLM:
    """Minimal llm: returns a fixed JSON body from .chat, ignoring the prompt."""
    def __init__(self, payload):
        self._payload = payload

    def chat(self, messages, tools=None):
        return {"role": "assistant", "content": self._payload}


_YT_ENTRIES = [
    {   # official label upload — KEEP
        "id": "vid1", "title": 'Kesariya - Brahmastra | Pritam | Arijit Singh',
        "channel": "Sony Music India", "channel_is_verified": True,
        "duration": 268, "description": "Singer: Arijit Singh\nMusic: Pritam\nLyrics: Amitabh Bhattacharya\nMovie: Brahmastra",
    },
    {   # fan/cover channel, wrong duration — DROP (untrusted + duration mismatch)
        "id": "vid2", "title": "Kesariya cover (slowed reverb)",
        "channel": "Random Fan Covers", "channel_is_verified": False,
        "duration": 120, "description": "my cover guys",
    },
    {   # "- Topic" art-track — KEEP
        "id": "vid3", "title": "Kesariya",
        "channel": "Arijit Singh - Topic", "channel_is_verified": False,
        "duration": 268, "description": "Provided to YouTube by Sony Music",
    },
]


def test_youtube_trust_filter_and_llm_parse(monkeypatch):
    _install_ytdlp(monkeypatch, _YT_ENTRIES)
    from audio_tagger.sources.youtube import YouTubeSource

    parsed = ('{"title":"Kesariya","film":"Brahmastra","year":2022,'
              '"label":"Sony Music India","singers":["Arijit Singh"],'
              '"composers":["Pritam"],"lyricists":["Amitabh Bhattacharya"],'
              '"lyrics":"kesariya tera ishq hai piya"}')
    src = YouTubeSource(llm=_FakeLLM(parsed), duration_tolerance_sec=5)
    track = InputTrack(path="x", existing_title="Kesariya",
                       existing_artist="Arijit Singh", duration_sec=268.0)
    cands = src.search(track)

    channels = {c.raw.get("channel") for c in cands}
    assert "Random Fan Covers" not in channels, "untrusted/duration-mismatch must be dropped"
    assert "Sony Music India" in channels and "Arijit Singh - Topic" in channels

    label = [c for c in cands if c.raw.get("channel") == "Sony Music India"][0]
    assert label.source == "youtube"
    assert label.film == "Brahmastra"
    assert label.release_type == ReleaseType.SOUNDTRACK
    assert label.singers == ["Arijit Singh"]
    assert label.composers == ["Pritam"]
    assert label.lyricists == ["Amitabh Bhattacharya"]
    assert label.raw.get("lyrics") == "kesariya tera ishq hai piya"


def test_youtube_no_llm_falls_back_to_title_pattern(monkeypatch):
    _install_ytdlp(monkeypatch, [_YT_ENTRIES[0]])
    from audio_tagger.sources.youtube import YouTubeSource

    src = YouTubeSource(llm=None)   # agent off -> degraded title-pattern parse
    cands = src.search(InputTrack(path="x", existing_title="Kesariya", duration_sec=268.0))
    assert cands
    c = cands[0]
    assert c.title == "Kesariya"
    assert c.film == "Brahmastra"          # recovered from "Song - Film | ..." title
    assert c.singers == []                 # roles unknown without the LLM


def test_youtube_degrades_without_ytdlp(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # force ImportError inside search
    from audio_tagger.sources.youtube import YouTubeSource
    assert YouTubeSource(llm=_FakeLLM("{}")).search(InputTrack(path="x", existing_title="X")) == []


# --------------------------------------------------------------------------- #
# MusicBrainz: work-rels populate composer + lyricist.
# --------------------------------------------------------------------------- #
def test_musicbrainz_populates_roles(monkeypatch, tmp_path):
    mb_search = _load("mb_search.json")
    mb_recording = _load("mb_recording.json")

    fake_mb = types.ModuleType("musicbrainzngs")
    fake_mb.set_useragent = lambda *a, **k: None
    fake_mb.search_recordings = lambda query, limit=5: mb_search
    fake_mb.get_recording_by_id = lambda rid, includes=None: {"recording": mb_recording}
    monkeypatch.setitem(sys.modules, "musicbrainzngs", fake_mb)

    from audio_tagger.sources.musicbrainz import MusicBrainzSource

    src = MusicBrainzSource(cache_dir=str(tmp_path))
    cands = src.search(InputTrack(path="x", existing_title="Kesariya",
                                  existing_artist="Arijit Singh"))
    assert cands
    c = cands[0]
    assert c.recording_mbid == "rec-mb-1"
    assert c.singers == ["Arijit Singh"]
    assert c.composers == ["Pritam"]                 # from work-relation-list
    assert c.lyricists == ["Amitabh Bhattacharya"]   # from work-relation-list
    assert c.film == "Brahmastra"
    assert c.release_type == ReleaseType.SOUNDTRACK


def test_musicbrainz_seed_mbids_looked_up(monkeypatch, tmp_path):
    mb_recording = _load("mb_recording.json")
    calls = {"search": 0, "byid": []}

    def _search(query, limit=5):
        calls["search"] += 1
        return {"recording-list": []}

    def _byid(rid, includes=None):
        calls["byid"].append(rid)
        return {"recording": mb_recording}

    fake_mb = types.ModuleType("musicbrainzngs")
    fake_mb.set_useragent = lambda *a, **k: None
    fake_mb.search_recordings = _search
    fake_mb.get_recording_by_id = _byid
    monkeypatch.setitem(sys.modules, "musicbrainzngs", fake_mb)

    from audio_tagger.sources.musicbrainz import MusicBrainzSource

    src = MusicBrainzSource(cache_dir=str(tmp_path))
    track = InputTrack(path="x", existing_title="Kesariya")
    cands = src.search(track, seed_mbids=["rec-mb-1"])
    assert cands and cands[0].recording_mbid == "rec-mb-1"
    assert "rec-mb-1" in calls["byid"], "AcoustID-seeded MBID must be fetched"


# --------------------------------------------------------------------------- #
# AcoustID: fingerprint lookup -> recording MBID + real audio score.
# --------------------------------------------------------------------------- #
def test_acoustid_fingerprint_lookup(monkeypatch):
    payload = _load("acoustid_lookup.json")

    def fake_post(url, data=None, timeout=None):
        return FakeResponse(payload)

    _install_requests(monkeypatch, post=fake_post)
    from audio_tagger.sources.acoustid_source import AcoustIDSource

    src = AcoustIDSource(api_key="testkey")
    track = InputTrack(path="x", existing_title="Kesariya",
                       duration_sec=210.0, acoustid_fingerprint="AQADtMk=")
    cands = src.search(track)
    assert cands
    c = cands[0]
    assert c.recording_mbid == "rec-mb-1"
    assert c.match_score == pytest.approx(0.95)      # real AcoustID score carried
    assert c.release_type == ReleaseType.SOUNDTRACK
    assert c.film == "Brahmastra"


def test_acoustid_self_disables_without_key(monkeypatch):
    monkeypatch.delenv("ACOUSTID_KEY", raising=False)
    from audio_tagger.sources.acoustid_source import AcoustIDSource

    src = AcoustIDSource(api_key=None)
    assert src.enabled is False
    assert src.search(InputTrack(path="x", duration_sec=200.0,
                                 acoustid_fingerprint="AQAD")) == []
