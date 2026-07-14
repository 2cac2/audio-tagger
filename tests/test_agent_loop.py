"""AgentJudge tests driven by a scripted fake LLM — fully offline.

The judge only needs an object with ``chat(messages, tools=None) -> dict`` and an
optional toolbox exposing ``openai_tools()`` / ``call(name, args)``. No ``openai``
or ``mcp`` import happens.
"""

import json

from audio_tagger.agent.loop import AgentJudge
from audio_tagger.config import HarnessConfig
from audio_tagger.models import ArtistCredit, Candidate, InputTrack, ReleaseType


def _cfg(max_turns=6):
    cfg = HarnessConfig()
    cfg.llm.max_agent_turns = max_turns
    return cfg


def _hint(**kw):
    base = dict(source="musicbrainz", title="Kesariya", release_title="Brahmastra",
                release_type=ReleaseType.SOUNDTRACK, film="Brahmastra",
                credits=[ArtistCredit("Wrong Singer", "singer")], match_score=0.4)
    base.update(kw)
    return Candidate(**base)


def _verdict_msg(**payload):
    return {"role": "assistant", "content": json.dumps(payload)}


class FakeLLM:
    """Pops scripted assistant messages in order; records each call's messages."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        if self._responses:
            return self._responses.pop(0)
        return {"role": "assistant", "content": "{}"}


class FakeToolbox:
    def __init__(self):
        self.calls = []

    def openai_tools(self):
        return [{"type": "function",
                 "function": {"name": "web_search", "parameters": {}}}]

    def call(self, name, args):
        self.calls.append((name, args))
        return "web result: original film is Brahmastra (2022)"


_TOOL_CALL_MSG = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "web_search", "arguments": "{\"q\": \"kesariya\"}"},
    }],
}


# --------------------------------------------------------------------------- #
# Tool call, then a final verdict.
# --------------------------------------------------------------------------- #
def test_tool_call_then_verdict():
    llm = FakeLLM([
        _TOOL_CALL_MSG,
        _verdict_msg(decision="pick", index=0, fields={}, reason="ok", confidence=0.5),
    ])
    tb = FakeToolbox()
    judge = AgentJudge(_cfg(), llm=llm, toolbox=tb)
    out = judge.search(InputTrack(path="x"), hints=[_hint()])
    assert len(out) == 1
    assert out[0].source == "agent"
    assert len(llm.calls) == 2
    assert tb.calls and tb.calls[0][0] == "web_search"


# --------------------------------------------------------------------------- #
# Invalid JSON -> single repair -> abstain.
# --------------------------------------------------------------------------- #
def test_invalid_json_repair_then_abstain():
    llm = FakeLLM([
        {"role": "assistant", "content": "sorry, I am not sure what to do here"},
        {"role": "assistant", "content": "still not json at all"},
    ])
    judge = AgentJudge(_cfg(), llm=llm)
    out = judge.search(InputTrack(path="x"), hints=[_hint()])
    assert out == []
    assert len(llm.calls) == 2       # original + one repair pass, then abstain


def test_invalid_json_repair_then_valid():
    llm = FakeLLM([
        {"role": "assistant", "content": "let me think..."},
        _verdict_msg(decision="pick", index=0, fields={}, reason="fixed", confidence=0.4),
    ])
    judge = AgentJudge(_cfg(), llm=llm)
    out = judge.search(InputTrack(path="x"), hints=[_hint()])
    assert len(out) == 1
    assert len(llm.calls) == 2


# --------------------------------------------------------------------------- #
# Max-turn bound: an always-tool-calling model stops at max_agent_turns chats.
# --------------------------------------------------------------------------- #
def test_max_turn_bound():
    always_tool = FakeLLM([dict(_TOOL_CALL_MSG) for _ in range(20)])
    judge = AgentJudge(_cfg(max_turns=3), llm=always_tool, toolbox=FakeToolbox())
    out = judge.search(InputTrack(path="x"), hints=[_hint()])
    assert out == []
    assert len(always_tool.calls) == 3


# --------------------------------------------------------------------------- #
# Confidence cap: the agent can never exceed the configured cap.
# --------------------------------------------------------------------------- #
def test_confidence_is_capped():
    llm = FakeLLM([
        _verdict_msg(decision="pick", index=0, fields={}, reason="sure", confidence=0.99),
    ])
    cfg = _cfg()
    judge = AgentJudge(cfg, llm=llm)
    out = judge.search(InputTrack(path="x"), hints=[_hint()])
    assert out[0].match_score == cfg.thresholds.agent_confidence_cap
    assert out[0].match_score <= 0.60


# --------------------------------------------------------------------------- #
# A pick edits the chosen structured candidate.
# --------------------------------------------------------------------------- #
def test_pick_edits_the_hint():
    llm = FakeLLM([
        _verdict_msg(decision="pick", index=0,
                     fields={"title": "Kesariya",
                             "singers": ["Arijit Singh"],
                             "composers": ["Pritam"]},
                     reason="corrected credits", confidence=0.5),
    ])
    judge = AgentJudge(_cfg(), llm=llm)
    out = judge.search(InputTrack(path="x"), hints=[_hint(title="Kesaria")])
    assert out[0].title == "Kesariya"
    assert out[0].singers == ["Arijit Singh"]
    assert out[0].composers == ["Pritam"]


def test_pick_bad_index_returns_empty():
    llm = FakeLLM([
        _verdict_msg(decision="pick", index=9, fields={}, reason="oops", confidence=0.5),
    ])
    judge = AgentJudge(_cfg(), llm=llm)
    out = judge.search(InputTrack(path="x"), hints=[_hint()])
    assert out == []


def test_new_candidate_created():
    llm = FakeLLM([
        _verdict_msg(decision="new_candidate",
                     fields={"title": "Kesariya", "film": "Brahmastra",
                             "singers": ["Arijit Singh"], "release_type": "soundtrack"},
                     reason="not in candidates", confidence=0.9),
    ])
    judge = AgentJudge(_cfg(), llm=llm)
    out = judge.search(InputTrack(path="x"), hints=[])
    assert len(out) == 1
    assert out[0].source == "agent"
    assert out[0].title == "Kesariya"
    assert out[0].match_score <= 0.60          # still capped


def test_abstain_returns_empty():
    llm = FakeLLM([_verdict_msg(decision="abstain", reason="unsure")])
    judge = AgentJudge(_cfg(), llm=llm)
    assert judge.search(InputTrack(path="x"), hints=[_hint()]) == []


# --------------------------------------------------------------------------- #
# Steering text reaches the LLM messages.
# --------------------------------------------------------------------------- #
def test_steer_text_reaches_messages():
    llm = FakeLLM([_verdict_msg(decision="abstain", reason="ok")])
    judge = AgentJudge(_cfg(), llm=llm)
    judge.search(InputTrack(path="x"), hints=[_hint()],
                 steer="this is the film version from Rockstar 2011")
    blob = json.dumps(llm.calls[0])
    assert "Rockstar 2011" in blob


def test_no_llm_returns_empty():
    judge = AgentJudge(_cfg(), llm=None)
    assert judge.search(InputTrack(path="x"), hints=[_hint()]) == []
