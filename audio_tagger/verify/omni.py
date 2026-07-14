"""Audio verification via a local omni LLM — evidence, never a verdict.

A ~25 second clip (see :mod:`.audio_clip`) is sent to the omni model as a
base64 ``input_audio`` content part with a strict-JSON prompt. The model reports
only what it can *hear* — sung language, vocalist gender, how many distinct lead
voices, live vs studio — which :func:`verify_clip` parses into a
:class:`VerifyReport`.

:func:`consistency` then scores how well a structured candidate agrees with that
report, returning a value in ``[-1, 1]``: a clip that sounds Punjabi against a
candidate the databases label Hindi drives the score negative; agreement drives
it positive. The resolver folds this in as a small ``±`` nudge and, on agreement,
may set ``audio_verified`` — the audio is corroborating evidence, and can never
pick a candidate on its own.

No third-party import happens at module load: ``verify_clip`` only needs an
object exposing ``chat(messages, tools=None) -> dict`` (the real
:class:`~audio_tagger.agent.llm.LocalLLM` or an offline fake), and the base64
packing is delegated to :func:`audio_tagger.agent.llm.audio_content_part`,
imported lazily inside the call.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.schema import safe_json
from ..models import Candidate

# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class VerifyReport:
    """What the omni model heard in the clip. All fields optional / best-effort.

    ``language``        : sung language, lowercased (``"hindi"``, ``"punjabi"``).
    ``vocalist_gender`` : ``"male" | "female" | "mixed" | "instrumental" | None``.
    ``vocalist_count``  : number of distinct lead voices, or ``None``.
    ``live_or_studio``  : ``"live" | "studio" | None``.
    ``confidence``      : the model's own 0..1 confidence in the report.
    ``notes``           : one short free-text line of supporting detail.
    """

    language: str | None = None
    vocalist_gender: str | None = None
    vocalist_count: int | None = None
    live_or_studio: str | None = None
    confidence: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert audio analyst for mainstream South Asian popular music "
    "(Hindi/Bollywood, Punjabi, Indipop). You are given a short audio clip. "
    "Report ONLY what you can actually hear in the audio — never guess from a "
    "title or artist name. Reply with a SINGLE strict JSON object and nothing "
    "else (no prose, no code fence)."
)

_SCHEMA_HINT = (
    "Return exactly this JSON shape:\n"
    "{\n"
    '  "language": "the sung language in lowercase (e.g. hindi, punjabi, '
    'english, tamil, telugu) or null if you cannot tell",\n'
    '  "vocalist_gender": "male | female | mixed | instrumental | null",\n'
    '  "vocalist_count": <integer count of distinct lead voices, or null>,\n'
    '  "live_or_studio": "live | studio | null",\n'
    '  "confidence": <number 0.0-1.0, how sure you are overall>,\n'
    '  "notes": "one short sentence of supporting detail"\n'
    "}"
)


# ---------------------------------------------------------------------------
# verify_clip
# ---------------------------------------------------------------------------

def verify_clip(llm, clip_bytes: bytes, candidates=None) -> VerifyReport | None:
    """Ask the omni model about ``clip_bytes``; return a report or ``None``.

    Exactly one ``llm.chat`` call. Returns ``None`` — never raising — when there
    is no LLM, no clip, the call fails, or the reply carries no parseable JSON.
    ``candidates`` (optional) only seeds a light "possible language(s)" hint; the
    model is told not to let it override what it hears.
    """
    if llm is None or not clip_bytes:
        return None

    # Lazy: the package must import with only the stdlib present.
    from ..agent.llm import audio_content_part

    user_text = _SCHEMA_HINT
    context = _candidate_context(candidates or [])
    if context:
        user_text += (
            "\n\nFor reference only (do NOT let this override what you hear): "
            + context
        )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                audio_content_part(clip_bytes, "wav"),
            ],
        },
    ]

    try:
        msg = llm.chat(messages)
    except Exception:
        return None

    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    data = safe_json(_content_text(content))
    if not data:
        return None
    return _report_from_dict(data)


def _candidate_context(candidates) -> str:
    """Compact hint listing distinct declared languages across candidates."""
    langs: list[str] = []
    for c in list(candidates)[:5]:
        raw = getattr(c, "raw", None)
        if isinstance(raw, dict):
            lang = raw.get("language")
            if lang:
                token = str(lang).strip()
                if token and token.lower() not in {x.lower() for x in langs}:
                    langs.append(token)
    if langs:
        return "candidate databases suggest language(s): " + ", ".join(langs)
    return ""


def _report_from_dict(d: dict) -> VerifyReport:
    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    notes = d.get("notes")
    return VerifyReport(
        language=_clean_str(d.get("language")),
        vocalist_gender=_clean_str(d.get("vocalist_gender")),
        vocalist_count=_coerce_int(d.get("vocalist_count")),
        live_or_studio=_clean_str(d.get("live_or_studio")),
        confidence=max(0.0, min(1.0, conf)),
        notes=str(notes).strip() if notes is not None else "",
    )


# ---------------------------------------------------------------------------
# Consistency scoring
# ---------------------------------------------------------------------------

# Languages that are close enough acoustically to count as agreement.
_LANG_ALIASES = {
    "hin": "hindi",
    "hindustani": "hindi",
    "urdu": "hindi",          # Hindustani register — indistinguishable in song
    "hi": "hindi",
    "pun": "punjabi",
    "panjabi": "punjabi",
    "pa": "punjabi",
    "tam": "tamil",
    "tel": "telugu",
    "ben": "bengali",
    "bangla": "bengali",
    "guj": "gujarati",
    "mar": "marathi",
    "kan": "kannada",
    "mal": "malayalam",
    "eng": "english",
    "en": "english",
}


def consistency(report: VerifyReport, candidate: Candidate) -> float:
    """Score audio/metadata agreement for ``candidate`` in ``[-1, 1]``.

    Language is the dominant signal (weight 1.0): if the candidate's declared
    language and the report's heard language are both known and disagree, that
    dimension contributes ``-1`` — a clip whose only usable signal is a language
    mismatch scores ``-1.0`` overall; a match scores ``+1.0``; an unknown on
    either side is neutral. Vocalist count (vs the number of credited singers)
    and gender/instrumental cues add weaker corroboration. Returns ``0.0`` when
    no dimension can be compared. Never raises.
    """
    if report is None or candidate is None:
        return 0.0

    signals: list[tuple[float, float]] = []  # (weight, score in [-1, 1])

    # 1) Language — the primary signal.
    match = _lang_match(_candidate_language(candidate), report.language)
    if match is True:
        signals.append((1.0, 1.0))
    elif match is False:
        signals.append((1.0, -1.0))

    # 2) Vocalist count vs number of credited singers (soft — a duet may show
    #    only one voice in a 25s window, so this only nudges).
    cand_count = len(getattr(candidate, "singers", []) or [])
    rep_count = report.vocalist_count
    if cand_count > 0 and isinstance(rep_count, int) and rep_count > 0:
        diff = abs(rep_count - cand_count)
        signals.append((0.4, max(-1.0, 1.0 - float(diff))))

    # 3) Gender / instrumental.
    gender = (report.vocalist_gender or "").strip().lower()
    if gender == "instrumental":
        # Credited singers but the clip has no vocals is a real contradiction.
        signals.append((0.5, -1.0) if cand_count > 0 else (0.3, 0.5))
    elif gender in ("male", "female", "mixed"):
        cand_gender = _candidate_gender(candidate)
        if cand_gender:
            signals.append((0.4, 1.0 if _gender_match(cand_gender, gender) else -1.0))
        elif cand_count > 0:
            # Audio has a voice and the candidate credits singers — mild agree.
            signals.append((0.2, 0.3))

    if not signals:
        return 0.0
    total_w = sum(w for w, _ in signals)
    agg = sum(w * s for w, s in signals) / total_w
    return max(-1.0, min(1.0, agg))


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


def _clean_str(value) -> str | None:
    """Lowercase/strip a scalar; ``None`` for empty or null-ish placeholders."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in ("null", "none", "unknown", "n/a", "na"):
        return None
    return s


def _coerce_int(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        # tolerate "2" / "2 voices" style strings
        try:
            import re
            m = re.search(r"-?\d+", str(value))
            return int(m.group()) if m else None
        except (TypeError, ValueError):
            return None


def _candidate_language(candidate) -> str | None:
    raw = getattr(candidate, "raw", None)
    if isinstance(raw, dict):
        return _clean_str(raw.get("language"))
    return None


def _candidate_gender(candidate) -> str | None:
    """Best-effort gender hint from a candidate's raw payload, if any."""
    raw = getattr(candidate, "raw", None)
    if isinstance(raw, dict):
        return _clean_str(raw.get("vocalist_gender") or raw.get("gender"))
    return None


def _canon_lang(s) -> str:
    """Canonicalize a language name; '' when empty/unknown."""
    val = _clean_str(s)
    if not val:
        return ""
    if val in _LANG_ALIASES:
        return _LANG_ALIASES[val]
    # "hindi film", "sung in punjabi" -> the first recognized language token wins.
    known = set(_LANG_ALIASES.values())
    for tok in val.replace("/", " ").split():
        if tok in _LANG_ALIASES:
            return _LANG_ALIASES[tok]
        if tok in known:
            return tok
    return val


def _lang_match(a, b):
    """``True``/``False`` if both languages are known, else ``None`` (neutral)."""
    ca, cb = _canon_lang(a), _canon_lang(b)
    if not ca or not cb:
        return None
    if ca == cb:
        return True
    return ca in cb or cb in ca


def _gender_match(a: str, b: str) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return True
    # "mixed" is compatible with a single-gender reading of a duet clip.
    return "mixed" in (a, b)
