"""Review-session state for the interactive TUI.

This module is the pure-Python heart of the review flow: it loads a session
from a resolutions JSON (the ``tag --json-out`` format) or, failing that, by
re-resolving a library path, wraps every reviewable track in a mutable
:class:`ReviewItem`, and round-trips the whole session to ``review.json`` so a
review can be paused and resumed.

It reuses the existing pipeline — ``tagmap.build_tags`` to turn a chosen
candidate into a tag dict, ``writer.write_tags`` (via the app) to persist, and
:class:`~audio_tagger.agent.loop.AgentJudge` for steering — and adds no new
resolution logic of its own. Everything here imports with only the standard
library present (``textual`` lives in :mod:`audio_tagger.tui.app`, imported only
when the UI actually runs).

Public surface
--------------
* :class:`ReviewSession`, :class:`ReviewItem` — the in-memory model.
* :func:`load_session` — build a session from a JSON file or a library path.
* :func:`save_session` — persist a session to ``review.json`` (resume-safe).
* :func:`candidate_to_dict` / :func:`candidate_from_dict`,
  :func:`track_to_dict` / :func:`track_from_dict`,
  :func:`item_to_dict` / :func:`item_from_dict` — serialization helpers.
* :func:`short_reason`, :data:`STATUS_GLYPHS` — small display helpers.

Steering
--------
Free-text human guidance reaches the agent through :meth:`ReviewItem.steer`,
which calls ``judge.search(track, hints=<ranked candidates>, steer=<text>,
audio_report=<report>)`` and merges the returned (confidence-capped,
agent-sourced) candidates back into the item. The agent can never
single-handedly clear the auto-write gate, and nothing is written until a human
accepts it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ..models import ArtistCredit, Candidate, InputTrack, ReleaseType
from ..tagmap import build_tags

# --- review status --------------------------------------------------------

STATUS_UNRESOLVED = "unresolved"
STATUS_AGENT = "agent-proposed"
STATUS_ACCEPTED = "accepted"
STATUS_SKIPPED = "skipped"

STATUS_GLYPHS = {
    STATUS_UNRESOLVED: "○",   # ○  open
    STATUS_AGENT: "◆",        # ◆  agent proposed
    STATUS_ACCEPTED: "✓",     # ✓  accepted
    STATUS_SKIPPED: "–",      # –  skipped
}

_SESSION_VERSION = 1


# ---------------------------------------------------------------------------
# small serialization helpers
# ---------------------------------------------------------------------------

def _rtype_value(rt) -> str:
    return rt.value if isinstance(rt, ReleaseType) else str(rt or "unknown")


def _parse_rtype(val) -> ReleaseType:
    if isinstance(val, ReleaseType):
        return val
    try:
        return ReleaseType(str(val).strip().lower())
    except (ValueError, AttributeError):
        return ReleaseType.UNKNOWN


def _jsonable(obj):
    """Return ``obj`` if it can be JSON-serialized, else a safe fallback."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if _is_json_scalar(v)}
        return None


def _is_json_scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, dict))


def _credits_from_dict(d: dict) -> list[ArtistCredit]:
    """Rebuild credits, preferring an explicit ``credits`` list, else role lists."""
    out: list[ArtistCredit] = []
    raw_credits = d.get("credits")
    if isinstance(raw_credits, list) and raw_credits:
        for cr in raw_credits:
            if isinstance(cr, dict) and (cr.get("name") or "").strip():
                out.append(ArtistCredit(
                    name=str(cr["name"]).strip(),
                    role=str(cr.get("role") or "singer"),
                    mbid=cr.get("mbid"),
                ))
        if out:
            return out
    # Fall back to the by-role name lists a compact candidate carries.
    for key, role in (
        ("singers", "singer"), ("composers", "composer"),
        ("lyricists", "lyricist"), ("producers", "producer"),
        ("mixers", "mixer"),
    ):
        for nm in d.get(key) or []:
            if isinstance(nm, str) and nm.strip():
                out.append(ArtistCredit(name=nm.strip(), role=role))
    return out


def candidate_to_dict(c: Candidate | None) -> dict | None:
    """Serialize a :class:`Candidate` to a JSON-safe dict.

    Emits both an authoritative ``credits`` list and the flattened per-role name
    lists (``singers``/``composers``/…) so a human reading the JSON — or a
    consumer that only knows the compact shape — gets what it needs.
    """
    if c is None:
        return None
    return {
        "source": c.source,
        "title": c.title,
        "release_title": c.release_title,
        "release_type": _rtype_value(c.release_type),
        "film": c.film,
        "year": c.year,
        "release_date": c.release_date,
        "is_various_artists": bool(c.is_various_artists),
        "recording_mbid": c.recording_mbid,
        "release_mbid": c.release_mbid,
        "match_score": round(float(c.match_score or 0.0), 4),
        "singers": c.singers,
        "composers": c.composers,
        "lyricists": c.lyricists,
        "producers": c.producers,
        "mixers": c.mixers,
        "credits": [
            {"name": cr.name, "role": cr.role, "mbid": cr.mbid} for cr in c.credits
        ],
        "raw": _jsonable(c.raw) or {},
    }


def candidate_from_dict(d) -> Candidate | None:
    """Rebuild a :class:`Candidate` from either the compact or full dict shape."""
    if not isinstance(d, dict):
        return None
    return Candidate(
        source=str(d.get("source") or ""),
        title=str(d.get("title") or ""),
        release_title=str(d.get("release_title") or ""),
        release_type=_parse_rtype(d.get("release_type")),
        credits=_credits_from_dict(d),
        film=d.get("film"),
        year=d.get("year"),
        release_date=d.get("release_date"),
        is_various_artists=bool(d.get("is_various_artists", False)),
        recording_mbid=d.get("recording_mbid"),
        release_mbid=d.get("release_mbid"),
        match_score=float(d.get("match_score") or 0.0),
        raw=d.get("raw") if isinstance(d.get("raw"), dict) else {},
    )


def track_to_dict(t: InputTrack) -> dict:
    return {
        "path": t.path,
        "existing_title": t.existing_title,
        "existing_artist": t.existing_artist,
        "existing_album": t.existing_album,
        "duration_sec": t.duration_sec,
        "acoustid_fingerprint": t.acoustid_fingerprint,
    }


def track_from_dict(d) -> InputTrack:
    d = d or {}
    return InputTrack(
        path=str(d.get("path") or ""),
        existing_title=d.get("existing_title"),
        existing_artist=d.get("existing_artist"),
        existing_album=d.get("existing_album"),
        duration_sec=d.get("duration_sec"),
        acoustid_fingerprint=d.get("acoustid_fingerprint"),
    )


def _audio_report_to_dict(report) -> dict | None:
    """Normalize a VerifyReport (dataclass), dict, or None to a JSON dict."""
    if report is None:
        return None
    if isinstance(report, dict):
        return _jsonable(report)
    import dataclasses
    if dataclasses.is_dataclass(report) and not isinstance(report, type):
        try:
            return _jsonable(dataclasses.asdict(report))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# candidate signature (for dedupe on merge)
# ---------------------------------------------------------------------------

def _cand_sig(c: Candidate) -> tuple:
    singers = "|".join(s.lower() for s in c.singers)
    comps = "|".join(s.lower() for s in c.composers)
    return (
        (c.source or "").lower(),
        (c.title or "").lower(),
        (c.release_title or "").lower(),
        c.recording_mbid or "",
        singers,
        comps,
    )


# ---------------------------------------------------------------------------
# ReviewItem
# ---------------------------------------------------------------------------

@dataclass
class ReviewItem:
    """One reviewable track: its candidates, provenance, and review state."""

    track: InputTrack
    candidates: list[Candidate] = field(default_factory=list)
    chosen_index: int = -1          # index of the resolver's top pick, or -1
    selected_index: int = 0         # index the human currently has highlighted
    confidence: float = 0.0
    agreement: float = 0.0
    n_agreeing_sources: int = 0
    audio_verified: bool = False
    agent_used: bool = False
    used_websearch_fallback: bool = False
    needs_review: bool = True
    reasons: list[str] = field(default_factory=list)
    audio_report: dict | None = None
    resolver_tags: dict = field(default_factory=dict)  # post-albumize tag dict
    status: str = STATUS_UNRESOLVED
    edited_tags: dict | None = None
    steer_log: list[str] = field(default_factory=list)

    # -- accessors ----------------------------------------------------------
    def selected_candidate(self) -> Candidate | None:
        if 0 <= self.selected_index < len(self.candidates):
            return self.candidates[self.selected_index]
        return None

    def chosen_candidate(self) -> Candidate | None:
        if 0 <= self.chosen_index < len(self.candidates):
            return self.candidates[self.chosen_index]
        return None

    def title(self) -> str:
        c = self.chosen_candidate() or self.selected_candidate()
        if c and c.title:
            return c.title
        return self.track.existing_title or os.path.basename(self.track.path)

    # -- final tags ---------------------------------------------------------
    def final_tags(self) -> dict:
        """The tag dict a write would apply for the current selection.

        Precedence: inline human edits > the resolver's reconciled tags (only
        when the highlighted candidate is still the resolver's own pick, so
        album-level reconciliation survives) > a fresh ``build_tags`` of the
        highlighted candidate.
        """
        if self.edited_tags:
            return dict(self.edited_tags)
        if self.selected_index == self.chosen_index and self.resolver_tags:
            return dict(self.resolver_tags)
        c = self.selected_candidate()
        if c is None:
            return dict(self.resolver_tags)
        return build_tags(c)

    # -- mutations ----------------------------------------------------------
    def accept(self, index: int | None = None) -> None:
        if index is not None and 0 <= index < len(self.candidates):
            self.selected_index = index
        self.status = STATUS_ACCEPTED

    def skip(self) -> None:
        self.status = STATUS_SKIPPED

    def merge_candidates(self, new: list[Candidate]) -> list[Candidate]:
        """Append not-yet-seen candidates; highlight the first newly added one."""
        seen = {_cand_sig(c) for c in self.candidates}
        added: list[Candidate] = []
        first_new: int | None = None
        for c in new or []:
            if c is None:
                continue
            sig = _cand_sig(c)
            if sig in seen:
                continue
            seen.add(sig)
            self.candidates.append(c)
            added.append(c)
            if first_new is None:
                first_new = len(self.candidates) - 1
        if first_new is not None:
            self.selected_index = first_new
        return added

    def steer(self, judge, steer_text: str) -> list[Candidate]:
        """Send human guidance to the agent judge and merge its proposals.

        This is the single point where free-text steering reaches the agent:
        it forwards ``steer=steer_text`` (plus the current ranked candidates as
        ``hints`` and any ``audio_report``) to ``judge.search``. Returned
        candidates are already ``source="agent"`` and confidence-capped by the
        judge; here they are only deduped, merged, and surfaced for the human.
        Returns the list of *newly added* candidates.
        """
        text = (steer_text or "").strip()
        if not text or judge is None:
            return []
        ranked = list(self.candidates)
        new = judge.search(
            self.track, hints=ranked, steer=text, audio_report=self.audio_report,
        ) or []
        self.steer_log.append(text)
        added = self.merge_candidates(new)
        if added:
            self.status = STATUS_AGENT
        return added

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return item_to_dict(self)


def item_to_dict(item: ReviewItem) -> dict:
    """Serialize a ReviewItem, resume-safe and json-out compatible."""
    return {
        "track": track_to_dict(item.track),
        "chosen": candidate_to_dict(item.chosen_candidate()),
        "candidates": [candidate_to_dict(c) for c in item.candidates],
        "chosen_index": item.chosen_index,
        "selected_index": item.selected_index,
        "confidence": item.confidence,
        "agreement": item.agreement,
        "n_agreeing_sources": item.n_agreeing_sources,
        "audio_verified": item.audio_verified,
        "agent_used": item.agent_used,
        "used_websearch_fallback": item.used_websearch_fallback,
        "needs_review": item.needs_review,
        "reasons": list(item.reasons),
        "audio_report": item.audio_report,
        "tags": item.resolver_tags,
        "status": item.status,
        "edited_tags": item.edited_tags,
        "steer_log": list(item.steer_log),
    }


def _find_candidate(cands: list[Candidate], target: Candidate) -> int:
    """Locate ``target`` within ``cands`` by identity-ish match, else -1."""
    if target.recording_mbid:
        for i, c in enumerate(cands):
            if c.recording_mbid and c.recording_mbid == target.recording_mbid:
                return i
    tsig = _cand_sig(target)
    for i, c in enumerate(cands):
        if _cand_sig(c) == tsig:
            return i
    return -1


def item_from_dict(d: dict) -> ReviewItem:
    """Rebuild a ReviewItem from either a saved session or a json-out entry.

    Tolerant of both shapes: a saved session carries ``chosen_index`` and the
    TUI fields (``status``/``selected_index``/``edited_tags``); a fresh json-out
    entry carries a ``chosen`` candidate that is reconciled back into the
    candidate list here.
    """
    cands = [candidate_from_dict(c) for c in (d.get("candidates") or [])]
    cands = [c for c in cands if c is not None]

    chosen_index = d.get("chosen_index")
    if chosen_index is None:
        chosen = candidate_from_dict(d.get("chosen")) if d.get("chosen") else None
        if chosen is not None:
            idx = _find_candidate(cands, chosen)
            if idx < 0:
                cands.insert(0, chosen)
                idx = 0
            chosen_index = idx
        else:
            chosen_index = 0 if cands else -1
    chosen_index = int(chosen_index)

    default_sel = chosen_index if chosen_index >= 0 else 0
    selected_index = int(d.get("selected_index", default_sel))
    if cands:
        selected_index = max(0, min(selected_index, len(cands) - 1))
    else:
        selected_index = 0

    tags = d.get("tags")
    if not isinstance(tags, dict):
        tags = {}
    edited = d.get("edited_tags")
    if not isinstance(edited, dict):
        edited = None

    return ReviewItem(
        track=track_from_dict(d.get("track")),
        candidates=cands,
        chosen_index=chosen_index,
        selected_index=selected_index,
        confidence=float(d.get("confidence") or 0.0),
        agreement=float(d.get("agreement") or 0.0),
        n_agreeing_sources=int(d.get("n_agreeing_sources") or 0),
        audio_verified=bool(d.get("audio_verified", False)),
        agent_used=bool(d.get("agent_used", False)),
        used_websearch_fallback=bool(d.get("used_websearch_fallback", False)),
        needs_review=bool(d.get("needs_review", True)),
        reasons=list(d.get("reasons") or []),
        audio_report=d.get("audio_report") if isinstance(d.get("audio_report"), dict) else None,
        resolver_tags=tags,
        status=str(d.get("status") or STATUS_UNRESOLVED),
        edited_tags=edited,
        steer_log=list(d.get("steer_log") or []),
    )


def item_from_resolution(res) -> ReviewItem:
    """Build a ReviewItem from a live :class:`~audio_tagger.models.Resolution`.

    ``candidates`` on a Resolution is not part of the shared contract, so we
    read it defensively (``getattr``) and always fold ``chosen`` in.
    """
    chosen = getattr(res, "chosen", None)
    cands = list(getattr(res, "candidates", None) or [])
    if chosen is not None and not cands:
        cands = [chosen]
    chosen_index = _find_candidate(cands, chosen) if chosen is not None else -1
    if chosen is not None and chosen_index < 0:
        cands.insert(0, chosen)
        chosen_index = 0
    sel = chosen_index if chosen_index >= 0 else 0
    return ReviewItem(
        track=res.track,
        candidates=cands,
        chosen_index=chosen_index,
        selected_index=sel,
        confidence=float(getattr(res, "confidence", 0.0) or 0.0),
        agreement=float(getattr(res, "agreement", 0.0) or 0.0),
        n_agreeing_sources=int(getattr(res, "n_agreeing_sources", 0) or 0),
        audio_verified=bool(getattr(res, "audio_verified", False)),
        agent_used=bool(getattr(res, "agent_used", False)),
        used_websearch_fallback=bool(getattr(res, "used_websearch_fallback", False)),
        needs_review=bool(getattr(res, "needs_review", True)),
        reasons=list(getattr(res, "reasons", None) or []),
        audio_report=_audio_report_to_dict(getattr(res, "audio_report", None)),
        resolver_tags=dict(getattr(res, "tags", None) or {}),
    )


# ---------------------------------------------------------------------------
# ReviewSession
# ---------------------------------------------------------------------------

@dataclass
class ReviewSession:
    """The whole review: every item, plus where it loaded from / saves to."""

    items: list[ReviewItem] = field(default_factory=list)
    source: str | None = None
    save_path: str = "review.json"
    dry_run: bool = True

    def queue_items(self) -> list[ReviewItem]:
        """Items surfaced in the review queue (those flagged for review).

        Falls back to *all* items if nothing is flagged, so opening a session
        never shows an empty queue.
        """
        pending = [it for it in self.items if it.needs_review]
        return pending or list(self.items)

    def accepted_items(self) -> list[ReviewItem]:
        return [it for it in self.items if it.status == STATUS_ACCEPTED]

    def to_dict(self) -> dict:
        return {
            "version": _SESSION_VERSION,
            "source": self.source,
            "dry_run": self.dry_run,
            "items": [item_to_dict(it) for it in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict, source: str | None = None) -> "ReviewSession":
        items = [item_from_dict(d) for d in (data.get("items") or [])]
        return cls(
            items=items,
            source=data.get("source") or source,
            dry_run=bool(data.get("dry_run", True)),
        )

    def save(self, path: str | None = None) -> str:
        target = path or self.save_path
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        self.save_path = target
        return target


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _session_from_json(data, source: str) -> ReviewSession:
    """Build a session from parsed JSON (saved session, or json-out list)."""
    if isinstance(data, dict) and "items" in data:
        # A previously-saved review session — resume it, saving back in place.
        session = ReviewSession.from_dict(data, source=source)
        session.save_path = source
        return session

    # A json-out document: a bare list, or {"resolutions": [...]}.
    if isinstance(data, dict):
        rows = data.get("resolutions") or data.get("results") or []
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError(f"unrecognized resolutions JSON in {source!r}")
    items = [item_from_dict(d) for d in rows if isinstance(d, dict)]
    return ReviewSession(items=items, source=source, save_path=_default_save_path(source))


def _default_save_path(source: str | None) -> str:
    if not source:
        return "review.json"
    d = os.path.dirname(os.path.abspath(source))
    return os.path.join(d, "review.json")


def _session_from_reresolve(source: str, config=None) -> ReviewSession:
    """Re-resolve a library path into a fresh review session.

    Reuses the real pipeline (scan -> sources -> rank -> agreement -> gate ->
    album reconciliation) via the resolver's public helpers, so the TUI adds no
    resolution logic. Network sources are best-effort; anything that fails is
    skipped rather than fatal.
    """
    from ..scan import scan_path
    tracks = scan_path(source, fingerprint=False)
    if not tracks:
        return ReviewSession(items=[], source=source, save_path=_default_save_path(source))

    sources = _build_sources(config)
    items = [_resolve_one(t, sources) for t in tracks]

    # Album-level reconciliation mirrors the tag pipeline, so grouped tracks
    # share ALBUM/ALBUMARTIST/comp. It operates on Resolution-shaped objects;
    # we hand it lightweight shims and copy the reconciled tags back.
    try:
        from ..albumize import reconcile_albums
        shims = [_ResolutionShim(it) for it in items]
        reconcile_albums(shims)
        for it, sh in zip(items, shims):
            it.resolver_tags = dict(sh.tags)
            it.reasons = list(sh.reasons)
    except Exception:
        pass

    return ReviewSession(items=items, source=source, save_path=_default_save_path(source))


def _build_sources(config):
    """Best-effort structured sources for re-resolution (keyless first)."""
    # Prefer the CLI's own builder when present, so we track its source set.
    try:
        from ..cli import _build_sources as cli_build
        srcs = cli_build()
        if srcs:
            return srcs
    except Exception:
        pass
    srcs = []
    for modname, clsname in (
        ("musicbrainz", "MusicBrainzSource"),
        ("itunes", "ITunesSource"),
    ):
        try:
            mod = __import__(f"audio_tagger.sources.{modname}", fromlist=[clsname])
            srcs.append(getattr(mod, clsname)())
        except Exception:
            pass
    return srcs


def _resolve_one(track, sources) -> ReviewItem:
    """Resolve a single track using the resolver's public scoring helpers."""
    from ..heuristics import rank_candidates
    from ..resolve import agreement_score, combined_confidence

    gathered: list[Candidate] = []
    for src in sources:
        try:
            gathered.extend(src.search(track) or [])
        except Exception:
            continue
    gathered = [c for c in gathered if not (isinstance(c.raw, dict) and c.raw.get("_error"))]

    if not gathered:
        return ReviewItem(
            track=track, candidates=[], chosen_index=-1, needs_review=True,
            reasons=["no structured source returned a candidate"],
        )

    ranked = rank_candidates(gathered)
    top = ranked[0]
    agreement = agreement_score(top, ranked)
    confidence = combined_confidence(top, agreement)
    return ReviewItem(
        track=track,
        candidates=ranked,
        chosen_index=0,
        selected_index=0,
        confidence=confidence,
        agreement=agreement,
        needs_review=confidence < 0.80,
        reasons=[
            f"chose '{top.title}' / '{top.release_title}' from {top.source} "
            f"(match={top.match_score:.2f}, agreement={agreement:.2f}, "
            f"conf={confidence:.2f})"
        ],
        resolver_tags=build_tags(top),
    )


class _ResolutionShim:
    """Duck-typed Resolution wrapper so ``reconcile_albums`` can mutate tags."""

    def __init__(self, item: ReviewItem):
        self._item = item
        self.chosen = item.selected_candidate() or item.chosen_candidate()
        self.tags = dict(item.resolver_tags)
        self.reasons = list(item.reasons)


def load_session(source: str, config=None) -> ReviewSession:
    """Load a review session from a JSON file or by re-resolving a path.

    * ``source`` ending in ``.json`` and existing  -> parse it. A document with
      an ``items`` key is a saved session (resumed and saved back in place);
      otherwise it is treated as a ``tag --json-out`` resolutions document.
    * any other existing path (a directory or an audio file) -> re-resolve with
      the structured pipeline.
    """
    if source and os.path.isfile(source) and source.lower().endswith(".json"):
        with open(source, encoding="utf-8") as fh:
            data = json.load(fh)
        return _session_from_json(data, source)
    return _session_from_reresolve(source, config)


def save_session(session: ReviewSession, path: str | None = None) -> str:
    """Persist ``session`` to ``path`` (or its ``save_path``); return the path."""
    return session.save(path)


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------

def short_reason(item: ReviewItem, width: int = 58) -> str:
    """A one-line reason for the queue row."""
    if item.status == STATUS_ACCEPTED:
        base = "accepted"
    elif item.status == STATUS_SKIPPED:
        base = "skipped"
    elif item.status == STATUS_AGENT:
        base = "agent proposed a candidate"
    elif item.reasons:
        base = item.reasons[-1]
    elif not item.candidates:
        base = "unidentified"
    else:
        base = "needs review"
    base = " ".join(str(base).split())
    if len(base) > width:
        return base[: width - 1] + "…"
    return base
