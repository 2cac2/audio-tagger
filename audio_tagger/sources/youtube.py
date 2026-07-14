"""YouTube source — label/Topic descriptions parsed by the LLM.

Official Bollywood / Punjabi label channels (T-Series, Sony Music India, Zee
Music Company, Saregama, YRF; Speed Records, White Hill Music, …) and the
auto-generated "… - Topic" art-tracks publish the full credit chain in the video
description, and often the lyrics too. That is label-authoritative data for
exactly the things this harness cares about: which film the song is from, and
who the playback singers vs music director vs lyricist are.

The hard part is trust — YouTube search is polluted with covers, fan reuploads
and "slowed+reverb" edits. So this source is deliberately narrow: it keeps a
result ONLY if the channel is a known label, a "… - Topic" art-track, or a
verified channel, AND (when the file's duration is known) the video duration
matches. Survivors are handed to the LLM for structured extraction
(:func:`~audio_tagger.agent.parse.parse_youtube_description`); the cross-source
comparison stays with the :class:`~audio_tagger.agent.loop.AgentJudge`.

``yt_dlp`` is imported lazily inside :meth:`search`, so the package still imports
with only the standard library present; without it (or without an LLM) the
source degrades gracefully rather than raising.
"""

from __future__ import annotations

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from ..translit import extract_film_from_title, sim
from .base import query_string

# Known official label / music-company channels (normalized, lowercase). Not
# exhaustive — extend via config ``sources.youtube.trusted_channels``.
_DEFAULT_TRUSTED = {
    "t-series", "sony music india", "zee music company", "saregama music",
    "saregama", "yrf", "yrf music", "tips official", "tips music",
    "speed records", "white hill music", "geet mp3", "t-series apnapunjab",
    "sony music south", "aditya music", "zee music punjabi", "eros now music",
    "times music", "venus", "shemaroo", "muzik247",
}


def _norm_channel(name: str | None) -> str:
    return (name or "").strip().lower()


class YouTubeSource:
    name = "youtube"

    def __init__(self, llm=None, max_results: int = 6,
                 trusted_channels=None, duration_tolerance_sec: int = 5,
                 cookiefile: str | None = None, extractor_args=None):
        self.llm = llm
        self.max_results = max_results
        self.duration_tolerance_sec = duration_tolerance_sec
        # cookiefile / extractor_args let the source work on bot-gated (datacenter)
        # hosts: a Netscape cookies.txt for a logged-in session, or a PO-token
        # provider base_url (e.g. {"youtubepot-bgutilhttp": {"base_url": [...]}}).
        # Both are optional; on a residential IP no mitigation is needed.
        self.cookiefile = cookiefile
        self.extractor_args = extractor_args
        trusted = set(_DEFAULT_TRUSTED)
        if trusted_channels:
            trusted |= {_norm_channel(c) for c in trusted_channels}
        self.trusted = trusted
        self.enabled = True   # degrades inside search() if yt_dlp is missing

    # -- trust helpers ------------------------------------------------------
    def _is_topic(self, channel: str | None) -> bool:
        return _norm_channel(channel).endswith("- topic")

    def _is_official(self, channel: str | None) -> bool:
        return _norm_channel(channel) in self.trusted

    def _trusted(self, entry: dict) -> tuple[bool, str]:
        """Return (is_trusted, tier) where tier is 'official' | 'topic' | 'verified'."""
        channel = entry.get("channel") or entry.get("uploader")
        if self._is_official(channel):
            return True, "official"
        if self._is_topic(channel):
            return True, "topic"
        if entry.get("channel_is_verified"):
            return True, "verified"
        return False, ""

    def _duration_ok(self, track: InputTrack, entry: dict) -> bool:
        dur = entry.get("duration")
        if not track.duration_sec or not dur:
            return True   # unknown on either side — don't reject on duration
        try:
            return abs(float(dur) - float(track.duration_sec)) <= self.duration_tolerance_sec
        except (TypeError, ValueError):
            return True

    # -- search -------------------------------------------------------------
    def search(self, track: InputTrack) -> list[Candidate]:
        try:
            from yt_dlp import YoutubeDL  # lazy: package imports without yt_dlp
        except Exception:
            return []
        q = query_string(track)
        if not q:
            return []

        opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "extract_flat": False, "noplaylist": True,
                # A single unavailable/private video in the batch must not sink
                # the whole search — bad entries become None and are skipped.
                "ignoreerrors": True}
        if self.cookiefile:
            opts["cookiefile"] = self.cookiefile
        if self.extractor_args:
            opts["extractor_args"] = self.extractor_args
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch{self.max_results}:{q}", download=False)
            entries = (info or {}).get("entries") or []
        except Exception:
            return []

        out: list[Candidate] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            trusted, tier = self._trusted(entry)
            if not trusted:
                continue
            # Official-label and "- Topic" uploads are authoritative even when the
            # music video is a shorter edit than the album track, so only the
            # weaker "verified" tier is duration-gated (catches a verified channel
            # hosting unrelated content, e.g. a TV-show clip).
            if tier == "verified" and not self._duration_ok(track, entry):
                continue
            cand = self._to_candidate(track, entry, tier)
            if cand:
                out.append(cand)
        return out

    # -- candidate assembly -------------------------------------------------
    def _to_candidate(self, track: InputTrack, entry: dict, tier: str) -> Candidate | None:
        raw_title = (entry.get("title") or "").strip()
        if not raw_title:
            return None
        channel = entry.get("channel") or entry.get("uploader")
        description = entry.get("description") or ""

        if self.llm is not None:
            from ..agent.parse import parse_youtube_description  # lazy: avoid import cycle
            parsed = parse_youtube_description(
                self.llm, description, raw_title, channel,
                file_hints={"title": track.existing_title, "artist": track.existing_artist},
            )
        else:
            parsed = self._fallback_parse(raw_title)
        if not parsed:
            parsed = self._fallback_parse(raw_title)

        title = parsed.get("title") or raw_title
        film = parsed.get("film")
        credits: list[ArtistCredit] = []
        for name in parsed.get("singers", []):
            credits.append(ArtistCredit(name, "singer"))
        for name in parsed.get("composers", []):
            credits.append(ArtistCredit(name, "composer"))
        for name in parsed.get("lyricists", []):
            credits.append(ArtistCredit(name, "lyricist"))

        rtype = ReleaseType.SOUNDTRACK if film else ReleaseType.UNKNOWN
        return Candidate(
            source=self.name,
            title=title,
            release_title=film or parsed.get("label") or "",
            release_type=rtype,
            credits=credits,
            film=film,
            year=parsed.get("year"),
            match_score=self._score(track, title, tier, entry),
            raw={
                "channel": channel,
                "channel_is_verified": bool(entry.get("channel_is_verified")),
                "is_topic": tier == "topic",
                "is_official": tier == "official",
                "trust_tier": tier,
                "video_id": entry.get("id"),
                "duration": entry.get("duration"),
                "label": parsed.get("label"),
                "lyrics": parsed.get("lyrics"),
            },
        )

    def _score(self, track: InputTrack, title: str, tier: str, entry: dict) -> float:
        """Trust tier sets the ceiling; title similarity and duration refine it."""
        base = 0.85 if tier in ("official", "topic") else 0.70
        title_sim = sim(track.existing_title, title) if track.existing_title else 0.6
        score = base * (0.55 + 0.45 * title_sim)
        # A confirmed duration match (both sides known and within tolerance) nudges up.
        if track.duration_sec and entry.get("duration") and self._duration_ok(track, entry):
            score = min(1.0, score + 0.05)
        return round(max(0.0, min(base, score)), 3)

    def _fallback_parse(self, raw_title: str) -> dict:
        """No-LLM degraded parse: recover title + film from the label title pattern.

        Official titles look like ``Song - Film | Actors | Music | Singer | ...``
        or ``Song (From "Film")``. We can salvage the title and film without the
        description's role labels; roles stay empty (the judge/other sources fill
        them). Better a weak title+film candidate than nothing.
        """
        title, film = extract_film_from_title(raw_title)
        title = title or raw_title
        if film is None:
            # "Song - Film | ..." pattern.
            head = raw_title.split("|", 1)[0].strip()
            if " - " in head:
                left, right = head.split(" - ", 1)
                title, film = left.strip(), right.strip() or None
            else:
                title = head
        return {"title": title, "film": film, "year": None, "label": None,
                "singers": [], "composers": [], "lyricists": [], "lyrics": None}
