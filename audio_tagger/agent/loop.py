"""The LLM judge — a bounded reconciler loop over structured candidates.

:class:`AgentJudge` is the harness's fallback tie-breaker, wired where the old
web-search fallback used to sit: the resolver calls ``search(track, hints=...)``
when the structured sources are empty or disagree, and the TUI calls it with a
free-text ``steer`` message. It never talks to files or the network directly —
it drives a local LLM (``llm``) and, optionally, MCP tools (``toolbox``).

The contract every consumer relies on:
  * output is a ``list[Candidate]`` with ``source="agent"``;
  * every returned candidate's ``match_score`` is capped at
    ``cfg.thresholds.agent_confidence_cap`` (default 0.60) — the agent can break
    a tie but can never single-handedly clear the 0.80 auto-write gate;
  * a ``pick`` verdict *edits/boosts* an existing structured candidate
    (transliteration + role fixes) rather than fabricating a new one;
  * uncertainty resolves to ``abstain`` -> empty list -> the track goes to review.

It is fully driveable offline: pass any object with
``chat(messages, tools=None) -> dict`` as ``llm`` (see the module tests) — no
``openai`` import happens here.
"""

from __future__ import annotations

import dataclasses
import json

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from .schema import safe_json, validate_verdict

_SYSTEM = (
    "You are a metadata reconciler for mainstream Hindi/Bollywood, Punjabi and "
    "Indipop songs. You are given a local file's existing tags and a short list "
    "of candidate identifications from structured music databases. Your job is to "
    "decide which candidate is correct and to fix its credits.\n\n"
    "RULES:\n"
    "1. Strongly prefer the ORIGINAL film soundtrack / album over any "
    "'Greatest Hits', 'Superhits', 'Top 50', 'Party Mix' or other compilation. "
    "A compilation is almost never the right release for a film song.\n"
    "2. Normalize transliteration when comparing (humein/hume/humey are the same "
    "word; pyaar/pyar; zindagi/jindagi). Spelling drift is not a different song.\n"
    "3. Assign roles precisely: playback singers -> role 'singer'; the music "
    "director/composer -> role 'composer'; the lyricist -> role 'lyricist'. Keep "
    "every distinct singer; do not merge duet singers into one string.\n"
    "4. ABSTAIN when you are not confident. A wrong tag is worse than no tag.\n"
    "5. You MAY call the provided web_search / fetch tools, but ONLY when the "
    "candidates genuinely conflict or are missing roles you need. Do not search "
    "when the candidates already agree.\n\n"
    "When you are done, reply with a SINGLE JSON object and nothing else:\n"
    "{\n"
    '  "decision": "pick" | "new_candidate" | "abstain",\n'
    '  "index": <int, the candidate number to pick — required for "pick">,\n'
    '  "fields": {\n'
    '     "title": str, "release_title": str, "film": str|null, "year": int|null,\n'
    '     "release_type": "soundtrack|album|single|compilation|ep|unknown",\n'
    '     "singers": [str], "composers": [str], "lyricists": [str]\n'
    "  },\n"
    '  "reason": str,\n'
    '  "confidence": 0.0-1.0\n'
    "}\n"
    "For 'pick', include in 'fields' only the values you are CORRECTING; omit the "
    "rest and the candidate's own values are kept. For 'new_candidate', fill in "
    "everything you know. For 'abstain', give a 'reason'."
)

_REPAIR = (
    "Your previous message was not a single valid JSON object matching the "
    "required schema. Reply now with ONLY the JSON object (no prose, no code "
    "fence). If you cannot decide, use {\"decision\": \"abstain\", \"reason\": \"...\"}."
)

_ROLE_KEYS = (
    ("singers", "singer"),
    ("composers", "composer"),
    ("lyricists", "lyricist"),
    ("producers", "producer"),
    ("mixers", "mixer"),
)


class AgentJudge:
    """LLM reconciler implementing the resolver's fallback ``search`` protocol."""

    name = "agent"

    def __init__(self, cfg, llm=None, toolbox=None):
        self.cfg = cfg
        self.llm = llm
        self.toolbox = toolbox
        llm_cfg = getattr(cfg, "llm", None)
        self.max_turns = int(getattr(llm_cfg, "max_agent_turns", 6) or 6)
        thr = getattr(cfg, "thresholds", None)
        self.cap = float(getattr(thr, "agent_confidence_cap", 0.60) or 0.60)

    # -- public entry point -------------------------------------------------
    def search(
        self,
        track: InputTrack,
        hints: list[Candidate] | None = None,
        steer: str | None = None,
        audio_report=None,
    ) -> list[Candidate]:
        """Judge one track; return agent-sourced, confidence-capped candidates.

        ``hints``        : ranked structured candidates (top of the list first).
        ``steer``        : optional free-text human guidance (from the TUI).
        ``audio_report`` : optional audio-verify evidence (dataclass or dict).
        """
        if self.llm is None:
            return []
        hints = list(hints or [])
        messages = self._build_messages(track, hints, steer, audio_report)
        tools = self._tools()
        verdict = self._run_loop(messages, tools)
        if verdict is None:
            return []
        return self._verdict_to_candidates(verdict, hints)

    # -- message construction ----------------------------------------------
    def _tools(self):
        if self.toolbox is None:
            return None
        try:
            tools = self.toolbox.openai_tools()
            return tools or None
        except Exception:
            return None

    def _build_messages(self, track, hints, steer, audio_report) -> list[dict]:
        facts = {
            "existing_title": track.existing_title,
            "existing_artist": track.existing_artist,
            "existing_album": track.existing_album,
            "duration_sec": track.duration_sec,
        }
        cand_json = [self._compact_candidate(i, c) for i, c in enumerate(hints[:6])]

        user_parts = [
            "FILE TAGS:\n" + json.dumps(facts, ensure_ascii=False),
            "CANDIDATES (index is what you pick):\n"
            + json.dumps(cand_json, ensure_ascii=False, indent=1),
        ]
        report = self._compact_report(audio_report)
        if report:
            user_parts.append("AUDIO VERIFICATION EVIDENCE:\n" + report)
        if steer:
            user_parts.append("HUMAN GUIDANCE (trust this strongly):\n" + str(steer))
        user_parts.append(
            "Decide now. Reply with the single JSON verdict object described above."
        )

        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    @staticmethod
    def _compact_candidate(index: int, c: Candidate) -> dict:
        rtype = c.release_type.value if isinstance(c.release_type, ReleaseType) else str(c.release_type)
        return {
            "index": index,
            "source": c.source,
            "title": c.title,
            "release_title": c.release_title,
            "release_type": rtype,
            "film": c.film,
            "year": c.year,
            "singers": c.singers,
            "composers": c.composers,
            "lyricists": c.lyricists,
            "match_score": round(c.match_score, 3),
        }

    @staticmethod
    def _compact_report(audio_report) -> str:
        if audio_report is None:
            return ""
        if dataclasses.is_dataclass(audio_report) and not isinstance(audio_report, type):
            try:
                return json.dumps(dataclasses.asdict(audio_report), ensure_ascii=False)
            except Exception:
                pass
        if isinstance(audio_report, dict):
            try:
                return json.dumps(audio_report, ensure_ascii=False)
            except Exception:
                return str(audio_report)
        return str(audio_report)

    # -- the bounded loop ---------------------------------------------------
    def _run_loop(self, messages: list[dict], tools):
        repaired = False
        turns = 0
        while turns < self.max_turns:
            turns += 1
            try:
                msg = self.llm.chat(messages, tools=tools)
            except Exception:
                return None
            if not isinstance(msg, dict):
                from .llm import _message_to_dict
                msg = _message_to_dict(msg)

            tool_calls = msg.get("tool_calls") or []
            # Dispatch tools only if we have somewhere to send them and turns left
            # to still produce a verdict afterwards.
            if tool_calls and tools and self.toolbox is not None and turns < self.max_turns:
                messages.append(self._assistant_echo(msg))
                for tc in tool_calls:
                    name, args = self._parse_tool_call(tc)
                    result = self._dispatch(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", "") if isinstance(tc, dict) else "",
                        "content": result,
                    })
                continue

            # Otherwise treat this message as the final verdict text.
            verdict = validate_verdict(safe_json(_content_text(msg.get("content"))))
            if verdict is not None:
                return verdict

            # One JSON-repair pass, then abstain.
            if not repaired and turns < self.max_turns:
                repaired = True
                messages.append(self._assistant_echo(msg))
                messages.append({"role": "user", "content": _REPAIR})
                continue
            return None
        return None

    @staticmethod
    def _assistant_echo(msg: dict) -> dict:
        """A clean assistant turn to append before tool results / repair prompt."""
        echo: dict = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            echo["tool_calls"] = msg["tool_calls"]
        return echo

    @staticmethod
    def _parse_tool_call(tc) -> tuple[str, dict]:
        if not isinstance(tc, dict):
            return "", {}
        fn = tc.get("function") or {}
        name = fn.get("name", "") or ""
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            args = safe_json(raw_args) or {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        return name, args

    def _dispatch(self, name: str, args: dict) -> str:
        if not name or self.toolbox is None:
            return ""
        try:
            return self.toolbox.call(name, args) or ""
        except Exception:
            return ""

    # -- verdict -> candidates ---------------------------------------------
    def _verdict_to_candidates(self, verdict, hints: list[Candidate]) -> list[Candidate]:
        cap_score = min(self.cap, verdict.confidence) if verdict.confidence else self.cap

        if verdict.decision == "abstain":
            return []

        if verdict.decision == "pick":
            idx = verdict.index
            if idx is None or not (0 <= idx < len(hints)):
                return []
            return [self._corrected_pick(hints[idx], verdict, cap_score)]

        # new_candidate
        cand = self._new_candidate(verdict, cap_score)
        return [cand] if cand else []

    def _corrected_pick(self, base: Candidate, verdict, score: float) -> Candidate:
        """Copy the chosen structured candidate and apply the judge's fixes."""
        f = verdict.fields or {}
        credits = self._credits_from_fields(f)
        raw = dict(base.raw or {})
        raw["agent_verdict"] = {
            "decision": "pick",
            "reason": verdict.reason,
            "confidence": verdict.confidence,
        }
        return dataclasses.replace(
            base,
            source="agent",
            title=f.get("title") or base.title,
            release_title=f.get("release_title") or base.release_title,
            release_type=_parse_rtype(f.get("release_type"), base.release_type),
            film=f.get("film") if "film" in f else base.film,
            year=_parse_year(f.get("year")) if f.get("year") is not None else base.year,
            credits=credits if credits else list(base.credits),
            match_score=score,
            raw=raw,
        )

    def _new_candidate(self, verdict, score: float) -> Candidate | None:
        f = verdict.fields or {}
        title = (f.get("title") or "").strip()
        if not title:
            return None
        credits = self._credits_from_fields(f)
        release_title = (f.get("release_title") or f.get("film") or title)
        return Candidate(
            source="agent",
            title=title,
            release_title=release_title,
            release_type=_parse_rtype(f.get("release_type"), ReleaseType.UNKNOWN),
            credits=credits,
            film=f.get("film"),
            year=_parse_year(f.get("year")),
            match_score=score,
            raw={"agent_verdict": {
                "decision": "new_candidate",
                "reason": verdict.reason,
                "confidence": verdict.confidence,
            }},
        )

    @staticmethod
    def _credits_from_fields(fields: dict) -> list[ArtistCredit]:
        credits: list[ArtistCredit] = []
        for key, role in _ROLE_KEYS:
            for nm in fields.get(key) or []:
                if isinstance(nm, str) and nm.strip():
                    credits.append(ArtistCredit(nm.strip(), role))
        # Also accept an explicit [{name, role}] list.
        for c in fields.get("credits") or []:
            if isinstance(c, dict) and (c.get("name") or "").strip():
                credits.append(ArtistCredit(c["name"].strip(), c.get("role", "singer")))
        return credits


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _content_text(content) -> str:
    """Flatten a message ``content`` (str or OpenAI content-part list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(part.get("text") or "")
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return str(content)


def _parse_rtype(val, default):
    if val is None:
        return default
    try:
        return ReleaseType(str(val).strip().lower())
    except (ValueError, AttributeError):
        return default


def _parse_year(val):
    if val is None:
        return None
    try:
        return int(str(val)[:4])
    except (TypeError, ValueError):
        return None
