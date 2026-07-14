"""Web-search fallback — the tie-breaker of last resort.

Used only when the structured sources return nothing or disagree. It is
deliberately dependency-injected: you pass in

  search_fn(query: str) -> list[dict]     # {title, url, snippet}
  llm_fn(system: str, user: str) -> str   # returns a JSON string

so it can be wired to whatever you have — a web-search MCP tool, SerpAPI,
DuckDuckGo, plus any LLM. Nothing here calls a specific provider, which
keeps the harness portable and testable with fakes.

The LLM is constrained to *extract* structured fields from the search
snippets and to abstain (return nulls) rather than invent — the same
no-hallucination rule the whole harness runs on. Its output still passes
back through confidence gating, so a shaky web guess lands in review, not
silently onto your files.
"""

from __future__ import annotations

import json

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from .base import query_string

_SYSTEM = (
    "You identify Indian film / Punjabi / indipop songs from web search snippets. "
    "Return ONLY JSON matching the schema. Distinguish playback singers (role 'singer'), "
    "the music director/composer (role 'composer') and the lyricist (role 'lyricist'). "
    "If the song is from a film, set 'film' to the film name and 'release_type' to "
    "'soundtrack'. Prefer the ORIGINAL soundtrack/album over any 'greatest hits', "
    "'superhits' or compilation. If you are unsure of a field, use null — never guess."
)

_SCHEMA_HINT = {
    "title": "string",
    "release_title": "string",
    "release_type": "soundtrack|album|single|compilation|unknown",
    "film": "string|null",
    "year": "int|null",
    "credits": [{"name": "string", "role": "singer|composer|lyricist|producer|mixer"}],
    "confidence": "0..1",
}


class WebSearchFallback:
    name = "websearch"

    def __init__(self, search_fn, llm_fn, max_results: int = 6):
        self.search_fn = search_fn
        self.llm_fn = llm_fn
        self.max_results = max_results

    def search(self, track: InputTrack, hints: list[Candidate] | None = None) -> list[Candidate]:
        q = query_string(track) or (track.existing_title or "")
        if not q:
            return []
        results = (self.search_fn(f"{q} song singers music director film album") or [])[: self.max_results]
        if not results:
            return []

        snippets = "\n".join(
            f"- {r.get('title','')}: {r.get('snippet','')} ({r.get('url','')})" for r in results
        )
        hint_txt = ""
        if hints:
            hint_txt = "\nStructured sources disagreed between:\n" + "\n".join(
                f"  * {h.title} / {h.release_title} [{h.release_type.value}] via {h.source}"
                for h in hints
            )
        user = (
            f"Query: {q}{hint_txt}\n\nSearch results:\n{snippets}\n\n"
            f"Return JSON with this shape (values are types):\n{json.dumps(_SCHEMA_HINT)}"
        )

        raw = self.llm_fn(_SYSTEM, user)
        data = _safe_json(raw)
        if not data or not data.get("title"):
            return []

        credits = [
            ArtistCredit(c["name"], c.get("role", "singer"))
            for c in data.get("credits", []) if c.get("name")
        ]
        try:
            rtype = ReleaseType(data.get("release_type", "unknown"))
        except ValueError:
            rtype = ReleaseType.UNKNOWN

        return [Candidate(
            source=self.name,
            title=data["title"],
            release_title=data.get("release_title") or data.get("film") or data["title"],
            release_type=rtype,
            credits=credits,
            film=data.get("film"),
            year=data.get("year"),
            # Web guesses are capped so they can nudge but never dominate a
            # structured match; the LLM's own stated confidence scales it.
            match_score=min(0.6, float(data.get("confidence", 0.4))),
            raw={"web_results": results, "llm": data},
        )]


def _safe_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start : end + 1]) if start >= 0 else None
    except (json.JSONDecodeError, ValueError):
        return None
