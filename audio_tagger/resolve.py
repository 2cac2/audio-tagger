"""Cross-source resolution.

Pipeline per track:
  1. Query the structured sources (MusicBrainz, iTunes, Spotify, Discogs).
  2. Rank every candidate with the release-preference heuristic.
  3. Measure cross-source *agreement* on the top pick and count how many
     *independent* sources actually back it.
  4. If the sources are empty or disagree -> ask the fallback judge (AgentJudge,
     the old web-search slot) to break the tie, then re-rank.
  5. Confidence-gate: below threshold, or backed by too few independent sources
     and not audio-verified, goes to the review queue.

Audio verification is *evidence only*: when the gate is at risk (too few
independent sources, or the top candidates disagree on language/film) a short
clip is analysed by the omni model and the result nudges the chosen candidate's
score by +/-0.1 and may set ``audio_verified`` — it can never pick alone.

Only the standard library is imported at module load; the audio-verify layer is
imported lazily inside the verification hook.
"""

from __future__ import annotations

from . import translit
from .credits import canonical_acts
from .heuristics import (
    looks_like_compilation,
    rank_candidates,
    release_preference_score,
)
from .models import Candidate, InputTrack, Resolution
from .tagmap import build_tags

# How strongly the audio must (dis)agree before it nudges the score.
_AUDIO_AGREE = 0.25
_AUDIO_DISAGREE = -0.25


def _norm(s: str | None) -> str:
    """Light, transliteration-aware normalization (delegated to ``translit``)."""
    return translit._light_norm(s)


def _sim(a: str | None, b: str | None) -> float:
    """Transliteration-tolerant similarity (delegated to ``translit.sim``)."""
    return translit.sim(a, b)


def _norm_isrc(value) -> str:
    """Uppercase an ISRC and drop separators (``US-ABC-12-34567`` forms)."""
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _acts_overlap(acts_a: list[str], acts_b: list[str]) -> bool:
    """True if any canonical act on each side is transliteration-equal."""
    for x in acts_a:
        for y in acts_b:
            if translit.sim(x, y) >= 0.9:
                return True
    return False


def _film_match(a: Candidate, b: Candidate) -> bool:
    """Both candidates name a film and the two names are close."""
    return bool(a.film) and bool(b.film) and translit.sim(a.film, b.film) >= 0.85


def _same_recording(a: Candidate, b: Candidate) -> bool:
    """Do two candidates from different sources describe the same recording?

    Fast, high-confidence shortcuts first: an equal recording-MBID or an equal
    ISRC settles it outright. Otherwise fall back to title/credit comparison.
    Singer names are folded to their canonical composer/act form and compared
    transliteration-tolerantly. The old "no singers on one side == agreement"
    loophole is gone: when either side lacks singer credits we demand a very
    strong title match *and* a corroborating film-or-release-title match before
    calling it the same recording.
    """
    # 1) Identifiers win outright.
    if a.recording_mbid and b.recording_mbid:
        return a.recording_mbid == b.recording_mbid

    isrc_a = _norm_isrc((a.raw or {}).get("isrc"))
    isrc_b = _norm_isrc((b.raw or {}).get("isrc"))
    if isrc_a and isrc_b:
        return isrc_a == isrc_b

    # 2) Textual match, transliteration-tolerant.
    title_sim = translit.sim(a.title, b.title)
    acts_a = canonical_acts(a.singers)
    acts_b = canonical_acts(b.singers)

    if acts_a and acts_b:
        # Both sides credit singers: a solid title match plus a shared act.
        return title_sim >= 0.85 and _acts_overlap(acts_a, acts_b)

    # One (or both) sides have no singer credits — no free pass. Require a near
    # -exact title AND a corroborating film or release-title agreement.
    if title_sim < 0.92:
        return False
    release_sim = translit.sim(a.release_title, b.release_title)
    return _film_match(a, b) or release_sim >= 0.6


def _by_source_best(all_candidates: list[Candidate]) -> dict[str, Candidate]:
    """Each source's own best candidate (list is already ranked best-first)."""
    by_source: dict[str, Candidate] = {}
    for c in all_candidates:
        by_source.setdefault(c.source, c)
    return by_source


def agreement_score(top: Candidate, all_candidates: list[Candidate]) -> float:
    """Fraction of *distinct sources* whose best candidate matches ``top``."""
    by_source = _by_source_best(all_candidates)
    if not by_source:
        return 0.0
    agreeing = sum(1 for c in by_source.values() if _same_recording(top, c))
    return agreeing / len(by_source)


def count_agreeing_sources(top: Candidate, all_candidates: list[Candidate]) -> int:
    """Number of *distinct, independent* sources whose best pick matches ``top``.

    The agent judge (``source == "agent"``) is a tie-breaker, not an independent
    corroborating database, so it never counts toward the source floor.
    """
    by_source = _by_source_best(all_candidates)
    return sum(
        1
        for name, c in by_source.items()
        if name != "agent" and _same_recording(top, c)
    )


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
        fallback=None,
        auto_threshold: float = 0.80,
        agreement_floor: float = 0.5,
        websearch_fallback=None,
        min_independent_sources: int = 2,
        verifier=None,
        llm=None,
    ):
        """
        sources                 : list of Source adapters (see sources/base.py)
        fallback / websearch_fallback
                                : optional tie-breaker judge (AgentJudge, or the
                                  legacy web-search fallback). Both names are
                                  accepted; ``fallback`` wins when both given.
                                  Used only when sources are empty or disagree.
        auto_threshold          : confidence at/above which we auto-write.
        agreement_floor         : below this agreement we invoke the fallback.
        min_independent_sources : distinct structured sources that must back the
                                  pick before it can auto-write without audio.
        verifier                : optional callable ``(track, candidates) ->
                                  VerifyReport|None`` audio-verify hook. When
                                  absent but ``llm`` is set, the built-in
                                  clip->omni pipeline is used instead.
        llm                     : optional omni LLM for the built-in verifier.
        """
        self.sources = sources
        # Backward-compatible: keep ``websearch_fallback`` as the live attribute
        # (existing tests/cli pass it by that name), but prefer ``fallback``.
        self.websearch_fallback = fallback or websearch_fallback
        self.fallback = self.websearch_fallback
        self.auto_threshold = auto_threshold
        self.agreement_floor = agreement_floor
        self.min_independent_sources = min_independent_sources
        self.verifier = verifier
        self.llm = llm

    # -- gathering ----------------------------------------------------------
    def _gather(self, track: InputTrack) -> list[Candidate]:
        candidates: list[Candidate] = []
        for src in self.sources:
            try:
                candidates.extend(src.search(track) or [])
            except Exception as exc:  # a flaky source must not sink the track
                candidates.append(_error_marker(src, exc))
        return [c for c in candidates if not c.raw.get("_error")]

    # -- audio evidence -----------------------------------------------------
    def _can_verify(self) -> bool:
        return self.verifier is not None or self.llm is not None

    def _audio_report(self, track: InputTrack, candidates: list[Candidate]):
        """Produce a VerifyReport (or None) — via the hook or the built-in path.

        Never raises: any failure (no ffmpeg, no llm, bad clip) yields ``None``.
        """
        try:
            if self.verifier is not None:
                return self.verifier(track, candidates)
            if self.llm is None:
                return None
            from .verify.audio_clip import extract_clip
            from .verify.omni import verify_clip

            clip = extract_clip(track.path)
            if not clip:
                return None
            return verify_clip(self.llm, clip, candidates)
        except Exception:
            return None

    def _apply_audio_evidence(
        self, track: InputTrack, top: Candidate, ranked: list[Candidate], reasons: list[str]
    ) -> bool:
        """Nudge ``top.match_score`` by +/-0.1 from an audio clip. Returns whether
        the audio *corroborated* the pick (sets ``audio_verified``).
        """
        report = self._audio_report(track, ranked)
        if report is None:
            return False
        try:
            from .verify.omni import consistency

            score = consistency(report, top)
        except Exception:
            return False

        if score >= _AUDIO_AGREE:
            top.match_score = min(1.0, top.match_score + 0.1)
            reasons.append(
                f"audio verification CORROBORATES the pick (consistency={score:.2f})"
            )
            return True
        if score <= _AUDIO_DISAGREE:
            top.match_score = max(0.0, top.match_score - 0.1)
            reasons.append(
                f"audio verification CONTRADICTS the pick (consistency={score:.2f})"
            )
            return False
        reasons.append(
            f"audio verification inconclusive (consistency={score:.2f})"
        )
        return False

    def _top_conflicts(self, top: Candidate, ranked: list[Candidate]) -> bool:
        """Do the leading candidates disagree on language or film?

        Only compares against candidates from a *different* source, so a source
        contradicting itself is ignored. A known/known language mismatch or two
        clearly different film names counts as a conflict worth verifying.
        """
        top_lang = _lang_token((top.raw or {}).get("language"))
        for c in ranked:
            if c is top or c.source == top.source:
                continue
            lang = _lang_token((c.raw or {}).get("language"))
            if top_lang and lang and top_lang != lang and top_lang not in lang and lang not in top_lang:
                return True
            if top.film and c.film and translit.sim(top.film, c.film) < 0.6:
                return True
        return False

    # -- main ---------------------------------------------------------------
    def resolve(self, track: InputTrack) -> Resolution:
        candidates = self._gather(track)
        reasons: list[str] = []
        used_fallback = False
        agent_used = False

        if not candidates:
            reasons.append("no structured source returned a candidate")
            if self.fallback:
                candidates = self.fallback.search(track) or []
                used_fallback = True
                agent_used = True
                reasons.append("invoked fallback judge (empty structured results)")

        if not candidates:
            return Resolution(
                track=track, chosen=None, confidence=0.0, agreement=0.0,
                needs_review=True, used_websearch_fallback=used_fallback,
                agent_used=agent_used,
                reasons=reasons + ["unidentified — queued for manual review"],
            )

        ranked = rank_candidates(candidates)
        top = ranked[0]
        agreement = agreement_score(top, ranked)

        # Sources disagree -> ask the fallback judge to break the tie, re-rank.
        if agreement < self.agreement_floor and self.fallback and not used_fallback:
            reasons.append(
                f"sources disagreed (agreement={agreement:.2f}) — fallback tie-breaker"
            )
            extra = self.fallback.search(track, hints=ranked[:3]) or []
            if extra:
                used_fallback = True
                agent_used = True
                ranked = rank_candidates(candidates + extra)
                top = ranked[0]
                agreement = agreement_score(top, ranked)

        n_agreeing = count_agreeing_sources(top, ranked)

        # Audio verification as EVIDENCE — only when the gate is genuinely at
        # risk (too few independent sources, or the leaders disagree on
        # language/film) and we actually have a way to verify.
        audio_verified = False
        source_floor_fails = n_agreeing < self.min_independent_sources
        if (source_floor_fails or self._top_conflicts(top, ranked)) and self._can_verify():
            audio_verified = self._apply_audio_evidence(track, top, ranked, reasons)

        confidence = combined_confidence(top, agreement)

        # New gate: below threshold, OR backed by too few independent sources
        # and not corroborated by audio. This closes the single-source auto-pass.
        needs_review = confidence < self.auto_threshold or (
            n_agreeing < self.min_independent_sources and not audio_verified
        )

        reasons.append(
            f"chose '{top.title}' / '{top.release_title}' from {top.source} "
            f"(match={top.match_score:.2f}, agreement={agreement:.2f}, "
            f"sources={n_agreeing}, conf={confidence:.2f})"
        )
        if n_agreeing < self.min_independent_sources and not audio_verified:
            reasons.append(
                f"only {n_agreeing} independent source(s) backing the pick "
                f"(need {self.min_independent_sources}) and no audio corroboration"
            )
        if top.release_type.value == "compilation" or looks_like_compilation(top):
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
            n_agreeing_sources=n_agreeing,
            audio_verified=audio_verified,
            agent_used=agent_used,
        )


def _lang_token(value) -> str:
    """Lowercase a raw language value for a coarse conflict check ('' if empty)."""
    return str(value or "").strip().lower()


def _error_marker(src, exc: Exception) -> Candidate:
    from .models import ReleaseType
    return Candidate(
        source=getattr(src, "name", "unknown"),
        title="", release_title="", release_type=ReleaseType.UNKNOWN,
        raw={"_error": str(exc)},
    )
