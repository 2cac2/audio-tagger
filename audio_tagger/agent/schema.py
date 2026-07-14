"""Verdict schema + JSON-repair helper for the LLM judge.

The agent (and the audio-verify model) talk back in JSON. Local LLMs served by
vLLM / llama.cpp are not always disciplined about it — they wrap the object in
prose, fence it in ```json blocks, or emit trailing commentary. ``safe_json``
digs the first well-formed object out of that noise, and ``validate_verdict``
turns a raw dict into a typed :class:`Verdict` or ``None`` (which the loop reads
as "abstain / try a repair pass").

Standard library only, so the package still imports with nothing installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# The three moves the judge is allowed to make.
DECISIONS = {"pick", "new_candidate", "abstain"}

# Loose synonyms a model might emit for the two non-abstain moves.
_DECISION_ALIASES = {
    "new": "new_candidate",
    "candidate": "new_candidate",
    "propose": "new_candidate",
    "create": "new_candidate",
    "choose": "pick",
    "select": "pick",
    "keep": "pick",
    "skip": "abstain",
    "unsure": "abstain",
    "unknown": "abstain",
    "none": "abstain",
}


@dataclass
class Verdict:
    """A validated decision from the judge.

    ``decision``   : one of ``pick`` / ``new_candidate`` / ``abstain``.
    ``index``      : for ``pick``, which structured candidate to correct/keep.
    ``fields``     : corrected/proposed fields (title, release_title, film, year,
                     release_type, singers/composers/lyricists, ...).
    ``reason``     : short human-readable justification (surfaced in review).
    ``confidence`` : the model's own 0..1 confidence (capped downstream).
    """

    decision: str
    index: int | None = None
    fields: dict = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _first_json_object(s: str) -> str | None:
    """Return the first *balanced* ``{...}`` substring, honoring JSON strings.

    A brace inside a quoted string ("Kal Ho Naa Ho {reprise}") must not throw
    off the depth count, so we track string state and escapes.
    """
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def safe_json(text: str) -> dict | None:
    """Extract the first JSON object from possibly-fenced, prose-wrapped text.

    Tolerates ```` ```json ```` fences and leading/trailing commentary. Returns
    the parsed dict, or ``None`` if nothing parseable is present. Never raises.
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip()

    # Peel a leading code fence (```json / ``` ...), and any trailing fence.
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        fence = s.rfind("```")
        if fence >= 0:
            s = s[:fence]
        s = s.strip()

    obj = _first_json_object(s)
    if obj is not None:
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: greedy first-brace..last-brace slice.
    start, end = s.find("{"), s.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(s[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Verdict validation
# ---------------------------------------------------------------------------

def _coerce_int(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_verdict(d: dict | None) -> Verdict | None:
    """Turn a raw dict into a :class:`Verdict`, or ``None`` if unusable.

    ``None`` signals the loop to attempt a single JSON-repair pass and then
    abstain. A ``pick`` without an integer ``index`` is treated as invalid,
    since it has nothing to correct.
    """
    if not isinstance(d, dict):
        return None

    decision = str(d.get("decision", "")).strip().lower()
    decision = decision.replace("-", "_").replace(" ", "_")
    decision = _DECISION_ALIASES.get(decision, decision)
    if decision not in DECISIONS:
        return None

    index = _coerce_int(d.get("index"))

    fields = d.get("fields")
    if not isinstance(fields, dict):
        fields = {}

    reason = str(d.get("reason", "") or "").strip()

    try:
        confidence = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # A "pick" must say which candidate it is picking.
    if decision == "pick" and index is None:
        return None

    return Verdict(
        decision=decision,
        index=index,
        fields=fields,
        reason=reason,
        confidence=confidence,
    )
