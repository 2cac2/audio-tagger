"""Cross-source resolution.

Pipeline per track:
  1. Query the structured sources (MusicBrainz, iTunes, Spotify, Discogs).
  2. Rank every candidate with the release-preference heuristic.
  3. Measure cross-source *agreement* on the top pick.
  4. If sources agree strongly  -> accept (auto-write if confident enough).
     If they disagree or are empty -> web-search fallback tie-breaker.
  5. Confidence-gate: anything below threshold goes to the review queue.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from .heuristics import rank_candidates, release_preference_score
from .models import Candidate, InputTrack, Resolution
from .tagmap import build_tags


def _norm(s: str | None) -> str:
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum() or ch.isspace()).strip()


def _sim(a: str | None, b: str | None) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _same_recording(a: Candidate, b: Candidate) -> bool:
    """Do two candidates from different sources describe the same recording?"""
    if a.recording_mbid and b.recording_mbid:
        return a.recording_mbid == b.recording_mbid
    title_ok = _sim(a.title, b.title) >= 0.85
    # Agreement on *any* singer strengthens the match (order/spelling varies).
    singers_a = {_norm(x) for x in a.singers}
    singers_b = {_norm(x) for x in b.singers}
    singer_ok = bool(singers_a & singers_b) or not (singers_a and singers_b)
    return title_ok and singer_ok


def agreement_score(top: Candidate, all_candidates: list[Candidate]) -> float:
    """Fraction of *distinct sources* whose best candidate matches ``top``."""
    by_source: dict[str, Candidate] = {}
    for c in all_candidates:
        # keep each source's own best (list is already ranked best-first)
        by_source.setdefault(c.source, c)

    if not by_source:
        return 0.0
    agreeing = sum(1 for c in by_source.values() if _same_recording(top, c))
    return agreeing / len(by_source)


def combined_confidence(top: Candidate, agreement: float) -> float:
    """Blend the source match score, release desirability and agreement."""
    return round(
        0.35 * top.match_score
        + 0.25 * release_preference_score(top)
        + 0.40 * agreement,
        3,
    )


class Resolver:
    def __init__(
        self,
        sources: list,
        websearch_fallback=None,
        auto_threshold: float = 0.80,
        agreement_floor: float = 0.5,
    ):
        """
        sources            : list of Source adapters (see sources/base.py)
        websearch_fallback : optional WebSearchFallback, used only on disagreement
        auto_threshold     : confidence at/above which we auto-write (else review)
        agreement_floor    : below this agreement we trigger the web fallback
        """
        self.sources = sources
        self.websearch_fallback = websearch_fallback
        self.auto_threshold = auto_threshold
        self.agreement_floor = agreement_floor

    def _gather(self, track: InputTrack) -> list[Candidate]:
        candidates: list[Candidate] = []
        for src in self.sources:
            try:
                candidates.extend(src.search(track) or [])
            except Exception as exc:  # a flaky source must not sink the track
                candidates.append(_error_marker(src, exc))
        return [c for c in candidates if not c.raw.get("_error")]

    def resolve(self, track: InputTrack) -> Resolution:
        candidates = self._gather(track)
        reasons: list[str] = []
        used_fallback = False

        if not candidates:
            reasons.append("no structured source returned a candidate")
            if self.websearch_fallback:
                candidates = self.websearch_fallback.search(track) or []
                used_fallback = True
                reasons.append("invoked web-search fallback (empty structured results)")

        if not candidates:
            return Resolution(
                track=track, chosen=None, confidence=0.0, agreement=0.0,
                needs_review=True, used_websearch_fallback=used_fallback,
                reasons=reasons + ["unidentified — queued for manual review"],
            )

        ranked = rank_candidates(candidates)
        top = ranked[0]
        agreement = agreement_score(top, ranked)

        # Sources disagree -> ask the web to break the tie, then re-rank.
        if agreement < self.agreement_floor and self.websearch_fallback and not used_fallback:
            reasons.append(
                f"sources disagreed (agreement={agreement:.2f}) — web-search tie-breaker"
            )
            extra = self.websearch_fallback.search(track, hints=ranked[:3]) or []
            if extra:
                used_fallback = True
                ranked = rank_candidates(candidates + extra)
                top = ranked[0]
                agreement = agreement_score(top, ranked)

        confidence = combined_confidence(top, agreement)
        needs_review = confidence < self.auto_threshold
        reasons.append(
            f"chose '{top.title}' / '{top.release_title}' from {top.source} "
            f"(match={top.match_score:.2f}, agreement={agreement:.2f}, conf={confidence:.2f})"
        )
        if top.release_type.value == "compilation":
            reasons.append("WARNING: best candidate is a compilation — verify original album")

        return Resolution(
            track=track,
            chosen=top,
            confidence=confidence,
            agreement=agreement,
            tags=build_tags(top),
            used_websearch_fallback=used_fallback,
            needs_review=needs_review,
            reasons=reasons,
        )


def _error_marker(src, exc: Exception) -> Candidate:
    from .models import ReleaseType
    return Candidate(
        source=getattr(src, "name", "unknown"),
        title="", release_title="", release_type=ReleaseType.UNKNOWN,
        raw={"_error": str(exc)},
    )
