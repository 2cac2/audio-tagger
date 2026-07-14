"""LLM parse of a YouTube description into structured credits.

Official Bollywood / Punjabi label uploads (T-Series, Sony Music India, Zee
Music, Speed Records, …) and auto-generated "… - Topic" art-tracks put the full
credit chain in the video description — "Singer:", "Music:", "Lyrics:", "Movie:",
"Music Label:" — and often the whole lyric. Formats vary enough that a regex is
brittle, so the :class:`~audio_tagger.sources.youtube.YouTubeSource` hands the
description to the local LLM and asks for strict JSON. This is the "LLM for
parsing" step; the cross-source *comparison* is still the job of
:class:`~audio_tagger.agent.loop.AgentJudge`.

Only the standard library is imported here; the ``llm`` argument is any object
with ``.chat(messages, tools=None) -> dict`` (the real :class:`LocalLLM` or a
fake), so this is fully offline-testable.
"""

from __future__ import annotations

from .schema import safe_json

_SYSTEM = (
    "You extract song credits from a YouTube video title + description for Indian "
    "(Hindi/Bollywood, Punjabi, indipop) music. Return ONLY a JSON object with keys: "
    "title (string), film (string or null — the movie/album this song is FROM), "
    "year (integer or null), label (string or null), "
    "singers (array of playback-singer names), composers (array of music "
    "director/composer names), lyricists (array of lyricist names), "
    "lyrics (string or null — the full lyric text if the description contains it). "
    "Distinguish the roles carefully: 'Music'/'Composed by' -> composers, "
    "'Singer'/'Vocals' -> singers, 'Lyrics'/'Lyricist'/'Written by' -> lyricists. "
    "If a field is not stated, use null or an empty array — never guess a name."
)


def _as_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_youtube_description(llm, description: str, title: str,
                             channel: str | None = None,
                             file_hints: dict | None = None) -> dict:
    """Parse a description into ``{title, film, year, label, singers,
    composers, lyricists, lyrics}`` using ``llm``. Returns ``{}`` on any failure.

    ``file_hints`` (existing tag title/artist) is passed as context so the model
    can prefer the credit chain that matches the user's file.
    """
    if llm is None:
        return {}
    user = (
        f"Video title: {title}\n"
        f"Channel: {channel or 'unknown'}\n"
        f"File hints: {file_hints or {}}\n\n"
        f"Description:\n{(description or '')[:4000]}"
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        msg = llm.chat(messages)
    except Exception:
        return {}
    content = msg.get("content") if isinstance(msg, dict) else None
    data = safe_json(content or "")
    if not isinstance(data, dict):
        return {}

    lyrics = data.get("lyrics")
    lyrics = lyrics.strip() if isinstance(lyrics, str) and lyrics.strip() else None
    return {
        "title": (str(data.get("title")).strip() if data.get("title") else title),
        "film": (str(data.get("film")).strip() if data.get("film") else None),
        "year": _as_int(data.get("year")),
        "label": (str(data.get("label")).strip() if data.get("label") else None),
        "singers": _as_str_list(data.get("singers")),
        "composers": _as_str_list(data.get("composers")),
        "lyricists": _as_str_list(data.get("lyricists")),
        "lyrics": lyrics,
    }
