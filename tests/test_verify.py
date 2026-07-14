"""Audio-verification tests — fake LLM, no ffmpeg, no network.

``verify_clip`` needs only an object with ``chat(messages, tools=None) -> dict``;
``extract_clip`` degrades to ``None`` when ffmpeg is unavailable (monkeypatched).
"""

import json

from audio_tagger.models import ArtistCredit, Candidate, ReleaseType
from audio_tagger.verify import VerifyReport, consistency, extract_clip, verify_clip
import audio_tagger.verify.audio_clip as audio_clip


def _cand(language=None, singers=None):
    raw = {"language": language} if language else {}
    credits = [ArtistCredit(n, "singer") for n in (singers or [])]
    return Candidate(source="jiosaavn", title="Kesariya", release_title="Brahmastra",
                     release_type=ReleaseType.SOUNDTRACK, credits=credits, raw=raw)


class FakeLLM:
    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        return {"role": "assistant", "content": json.dumps(self._payload)}


# --------------------------------------------------------------------------- #
# verify_clip: parse the omni model's JSON report.
# --------------------------------------------------------------------------- #
def test_verify_clip_parses_report():
    llm = FakeLLM({
        "language": "hindi", "vocalist_gender": "male", "vocalist_count": 1,
        "live_or_studio": "studio", "confidence": 0.8, "notes": "solo male vocal",
    })
    report = verify_clip(llm, b"RIFFxxxxWAVEfake-audio-bytes", [_cand("hindi")])
    assert isinstance(report, VerifyReport)
    assert report.language == "hindi"
    assert report.vocalist_gender == "male"
    assert report.vocalist_count == 1
    assert report.live_or_studio == "studio"
    assert llm.calls == 1


def test_verify_clip_none_paths():
    llm = FakeLLM({"language": "hindi", "confidence": 0.5})
    assert verify_clip(None, b"bytes") is None       # no llm
    assert verify_clip(llm, b"") is None             # no clip
    assert llm.calls == 0


# --------------------------------------------------------------------------- #
# consistency: language is the dominant signal.
# --------------------------------------------------------------------------- #
def test_consistency_language_match_is_positive():
    report = VerifyReport(language="hindi", confidence=0.9)
    assert consistency(report, _cand("hindi")) == 1.0


def test_consistency_language_mismatch_penalized():
    report = VerifyReport(language="hindi", confidence=0.9)
    assert consistency(report, _cand("punjabi")) == -1.0


def test_consistency_hindi_urdu_are_agreement():
    # Hindustani register — urdu folds to hindi, so this is not a penalty.
    report = VerifyReport(language="urdu", confidence=0.9)
    assert consistency(report, _cand("hindi")) == 1.0


def test_consistency_unknown_language_is_neutral():
    report = VerifyReport(language=None, confidence=0.5)
    assert consistency(report, _cand(None)) == 0.0


# --------------------------------------------------------------------------- #
# extract_clip: graceful degradation without ffmpeg.
# --------------------------------------------------------------------------- #
def test_extract_clip_none_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(audio_clip.shutil, "which", lambda name: None)
    assert extract_clip("/some/track.mp3") is None


def test_extract_clip_none_on_empty_path():
    assert extract_clip("") is None
