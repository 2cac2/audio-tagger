"""Command-line entry point.

    python -m audio_tagger tag   /music --dry-run
    python -m audio_tagger tag   /music --apply           # writes tags + review.csv
    python -m audio_tagger tag   /music --json-out run.json  # dump for the TUI
    python -m audio_tagger review run.json                 # interactive review TUI
    python -m audio_tagger apply-reviews review.csv        # writes approved rows
    python -m audio_tagger rollback /music                 # restore from backups

Sources activate based on the loaded ``HarnessConfig`` (per-source ``enabled``
flags + available deps/keys); the harness runs with whatever is present (at
minimum MusicBrainz + iTunes + Deezer, all keyless). The LLM agent judge and
audio verification are opt-in and degrade to off when their endpoint/tools are
unavailable.

Heavy modules (config's yaml, the source adapters, the agent/verify/tui layers)
are imported lazily inside the functions that use them, so ``import
audio_tagger.cli`` stays cheap and never requires third-party libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .albumize import reconcile_albums
from .lyrics import choose_writable, fetch_lyrics
from .resolve import Resolver
from .review import read_approvals, write_queue
from .scan import scan_path
from .writer import rollback, write_lyrics_tag, write_tags


# ---------------------------------------------------------------------------
# builders (config-driven)
# ---------------------------------------------------------------------------

def _enabled(cfg, name: str) -> bool:
    """Is source ``name`` enabled in the config (defaulting to True)?"""
    sc = cfg.sources.get(name)
    return bool(getattr(sc, "enabled", True))


def _build_sources(cfg, no_fingerprint: bool = False, llm=None) -> list:
    """Construct the enabled source adapters in resolution order.

    Order: AcoustID (fingerprint) first, then MusicBrainz, JioSaavn, YouTube,
    iTunes, Deezer; Discogs and Spotify only when explicitly enabled (and
    credentialed). Any adapter that fails to construct (missing dep) is skipped
    with a warning. ``llm`` (when the agent is on) is injected into the YouTube
    source so it can parse label descriptions into structured credits.
    """
    from .sources import (
        AcoustIDSource,
        DeezerSource,
        DiscogsSource,
        ITunesSource,
        JioSaavnSource,
        MusicBrainzSource,
        SpotifySource,
        YouTubeSource,
    )

    out: list = []

    def add(name: str, factory):
        if not _enabled(cfg, name):
            return
        try:
            src = factory()
        except Exception as exc:  # missing dep / bad config — skip, don't crash
            print(f"[warn] {name} disabled: {exc}", file=sys.stderr)
            return
        if not getattr(src, "enabled", True):
            print(f"[note] {name} disabled (no key/creds)", file=sys.stderr)
            return
        out.append(src)

    ac = cfg.sources.get("acoustid")
    if not no_fingerprint:
        add("acoustid", lambda: AcoustIDSource(api_key=getattr(ac, "api_key", None)))

    add("musicbrainz", lambda: MusicBrainzSource())

    js = cfg.sources.get("jiosaavn")
    add("jiosaavn", lambda: JioSaavnSource(base_url=getattr(js, "base_url", None)))

    yt = cfg.sources.get("youtube")
    yt_extra = getattr(yt, "extra", {}) or {}
    # Optional bot-gate mitigations for datacenter hosts: a cookies.txt, and/or a
    # PO-token provider base_url (the bgutil HTTP provider yt-dlp auto-discovers
    # at 127.0.0.1:4416 needs nothing here; only set base_url for a custom host).
    yt_pot_url = yt_extra.get("pot_base_url")
    yt_extractor_args = (
        {"youtubepot-bgutilhttp": {"base_url": [yt_pot_url]}} if yt_pot_url else None
    )
    add("youtube", lambda: YouTubeSource(
        llm=llm,
        max_results=int(yt_extra.get("max_results", 6)),
        trusted_channels=yt_extra.get("trusted_channels"),
        duration_tolerance_sec=int(yt_extra.get("duration_tolerance_sec", 5)),
        cookiefile=yt_extra.get("cookiefile"),
        extractor_args=yt_extractor_args,
    ))

    add("itunes", lambda: ITunesSource())
    add("deezer", lambda: DeezerSource())

    dc = cfg.sources.get("discogs")
    add("discogs", lambda: DiscogsSource(token=getattr(dc, "api_key", None)))

    sp = cfg.sources.get("spotify")
    add("spotify", lambda: SpotifySource(
        client_id=getattr(sp, "api_key", None),
        client_secret=(getattr(sp, "extra", {}) or {}).get("client_secret"),
        enabled=True,
    ))

    return out


def _build_llm(cfg):
    """Construct a ``LocalLLM`` for the configured endpoint (no network yet)."""
    from .agent import LocalLLM
    return LocalLLM(cfg.llm)


def _build_verifier(cfg, llm, mode: str):
    """A ``(track, candidates) -> VerifyReport|None`` audio-verify hook, or None.

    ``mode`` is ``off`` (no hook), ``auto`` or ``always``. The Resolver decides
    *when* to invoke the hook (on gate-risk for ``auto``); the hook itself just
    extracts a clip and asks the omni model. Never raises.
    """
    if mode == "off" or llm is None:
        return None
    v = getattr(cfg, "verify", None)
    offset = getattr(v, "offset", "middle")
    seconds = int(getattr(v, "clip_seconds", 25) or 25)
    sr = int(getattr(v, "sample_rate", 16000) or 16000)

    def verifier(track, candidates):
        try:
            from .verify import extract_clip, verify_clip
            clip = extract_clip(track.path, offset=offset, seconds=seconds, sr=sr)
            if not clip:
                return None
            return verify_clip(llm, clip, candidates)
        except Exception:
            return None

    return verifier


# ---------------------------------------------------------------------------
# tag
# ---------------------------------------------------------------------------

def cmd_tag(args):
    from .config import HarnessConfig
    cfg = HarnessConfig.load(args.config)

    auto_threshold = args.threshold if args.threshold is not None else cfg.thresholds.auto

    tracks = scan_path(args.path, fingerprint=not args.no_fingerprint)
    if args.limit:
        tracks = tracks[: args.limit]
    print(f"scanned {len(tracks)} track(s)")

    # -- LLM / agent / verifier wiring -------------------------------------
    llm = None
    judge = None
    verifier = None
    want_agent = args.agent  # True (--agent), False (--no-agent), or None (auto)

    if want_agent is not False or args.verify_audio != "off":
        try:
            llm = _build_llm(cfg)
        except Exception as exc:
            print(f"[note] LLM unavailable: {exc}", file=sys.stderr)
            llm = None

    if want_agent is not False and llm is not None:
        use = want_agent
        if use is None:  # auto: on iff the endpoint answers a ping
            try:
                use = bool(llm.ping())
            except Exception:
                use = False
            if not use:
                print("[note] agent off: LLM endpoint did not respond to /models ping")
        if use:
            try:
                from .agent import AgentJudge, McpToolbox
                judge = AgentJudge(cfg, llm=llm, toolbox=McpToolbox(cfg.mcp))
                print("[note] agent judge enabled")
            except Exception as exc:
                print(f"[note] agent disabled: {exc}", file=sys.stderr)
                judge = None

    if args.verify_audio != "off":
        verifier = _build_verifier(cfg, llm, args.verify_audio)
        if verifier is not None:
            print(f"[note] audio verification: {args.verify_audio}")

    resolver = Resolver(
        sources=_build_sources(cfg, args.no_fingerprint, llm=llm if judge is not None else None),
        fallback=judge,
        auto_threshold=auto_threshold,
        agreement_floor=cfg.thresholds.agreement_floor,
        min_independent_sources=cfg.thresholds.min_independent_sources,
        verifier=verifier,
        llm=llm if verifier is not None else None,
    )

    resolutions = [resolver.resolve(t) for t in tracks]
    resolutions = reconcile_albums(resolutions)

    auto = [r for r in resolutions if not r.needs_review]
    print(f"auto-write: {len(auto)}   review: {len(resolutions) - len(auto)}")

    # -- per-track status line ---------------------------------------------
    for r in resolutions:
        flags = []
        if r.agent_used:
            flags.append("agent")
        if r.audio_verified:
            flags.append("audio-verified")
        tag = "AUTO " if not r.needs_review else "REVIEW"
        suffix = f" [{','.join(flags)}]" if flags else ""
        print(
            f"[{tag}] {os.path.basename(r.track.path)}  "
            f"conf={r.confidence:.2f} sources={r.n_agreeing_sources}{suffix}"
        )

    # -- write auto-passed tracks (+ lyrics) -------------------------------
    for r in resolutions:
        if r.needs_review or not r.chosen:
            continue
        if args.lyrics:
            _handle_lyrics(cfg, r, apply=args.apply)
        res = write_tags(r.track.path, r.tags, dry_run=not args.apply)
        _print_diff(res, r)

    n = write_queue(resolutions, args.review_out)
    print(f"wrote {args.review_out} ({n} rows to review)")

    if args.json_out:
        _dump_json(resolutions, args.json_out)
        print(f"wrote {args.json_out} (full resolutions for the review TUI)")

    if not args.apply:
        print("dry-run: no files changed. Re-run with --apply to write.")


def _handle_lyrics(cfg, r, apply: bool):
    """Fetch lyrics for a resolved track: synced -> .lrc sidecar; else plain -> tag."""
    got = fetch_lyrics(r.chosen, r.track.duration_sec)
    synced, plain = choose_writable(got)
    if synced:
        _save_lrc(r.track.path, synced)
    elif plain and getattr(cfg.lyrics, "write_plain_to_tag", True):
        write_lyrics_tag(r.track.path, plain, dry_run=not apply)


def _dump_json(resolutions, path: str):
    """Serialize resolutions (with their chosen candidate) into TUI-loadable JSON."""
    from .tui.state import item_from_resolution, item_to_dict
    docs = [item_to_dict(item_from_resolution(r)) for r in resolutions]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# review (TUI)
# ---------------------------------------------------------------------------

def cmd_review(args):
    from .tui import run_tui
    run_tui(args.source, config=args.config, apply=args.apply)


# ---------------------------------------------------------------------------
# apply-reviews
# ---------------------------------------------------------------------------

def cmd_apply_reviews(args):
    approvals = read_approvals(args.csv)
    for path, fields in approvals.items():
        # read_approvals already mirrors a build_tags result (year/grouping/comp/
        # MBIDs included); drop empty/None/zero cells so we only write real values.
        tags = {k: v for k, v in fields.items() if v}
        res = write_tags(path, tags, dry_run=args.dry_run)
        print(res.get("path"), "->", "dry-run" if args.dry_run else "written")
    print(f"applied {len(approvals)} approved row(s)")


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------

def cmd_rollback(args):
    tracks = scan_path(args.path, fingerprint=False)
    restored = sum(1 for t in tracks if rollback(t.path))
    print(f"rolled back {restored}/{len(tracks)} file(s)")


# ---------------------------------------------------------------------------
# organize (copy-only)
# ---------------------------------------------------------------------------

def cmd_organize(args):
    """Copy tracks into a clean <dest>/<Album>/<Title> tree. COPY-ONLY.

    Sources are never moved, deleted, or modified. With ``--from-json`` the
    resolved album/title from a ``tag --json-out`` file drive the layout;
    otherwise each file's *current* tags do. ``--tag-copies`` (json mode only)
    writes the resolved tags onto the COPIES afterward, so a fully tagged,
    organized library is produced while the originals stay untouched.
    """
    from . import organize as org

    items, tags_list = _organize_items(args)
    if not items:
        print("nothing to organize (no audio files / resolutions found)")
        return

    ops = org.plan(items, args.dest)
    print(f"planned {len(ops)} copy operation(s) into {os.path.abspath(args.dest)}")
    print("COPY-ONLY: your original files are never moved, deleted, or modified.\n")

    def _progress(op):
        if op.status in ("copied", "renamed", "dry_run"):
            rel = os.path.relpath(op.dest, os.path.abspath(args.dest))
            tag = "COPY " if not args.apply else ("COPY " if op.status != "renamed" else "COPY*")
            print(f"  [{ 'DRY' if args.apply is False else tag.strip() }] {rel}")

    summary = org.execute(ops, dry_run=not args.apply, on_progress=_progress if args.verbose else None)

    # Optionally tag the COPIES (never the originals) with the resolved tags.
    if args.tag_copies and tags_list and args.apply:
        tagged = 0
        for op, tags in zip(ops, tags_list):
            if op.status in ("copied", "renamed") and tags:
                write_tags(op.dest, tags, dry_run=False)
                tagged += 1
        print(f"tagged {tagged} copied file(s) in the new tree")

    _print_org_summary(ops, summary, applied=args.apply)


def _organize_items(args):
    """Return (items, tags_list). ``tags_list`` is parallel resolved tags or []."""
    from . import organize as org
    path = args.path
    use_json = args.from_json or (os.path.isfile(path) and path.lower().endswith(".json"))
    if use_json:
        from .tui.state import load_session
        session = load_session(path, config=None)
        items = org.items_from_review_items(session.items)
        tags_list = [it.final_tags() for it in session.items]
        return items, tags_list
    tracks = scan_path(path, fingerprint=False)
    return org.items_from_paths([t.path for t in tracks]), []


def _print_org_summary(ops, summary, applied: bool):
    from collections import Counter
    albums = Counter(op.album for op in ops)
    verb = "copied" if applied else "would copy"
    print()
    print(f"{'APPLIED' if applied else 'DRY-RUN'} — {len(albums)} album folder(s):")
    for album, n in sorted(albums.items()):
        print(f"    {album}/   ({n} track(s))")
    s = summary
    done = s["copied"] + s["renamed"] if applied else s["dry_run"]
    mb = s["bytes"] / (1024 * 1024)
    print(f"\n  {verb}: {done}   duplicates skipped: {s['skipped_dup']}   "
          f"renamed (no overwrite): {s['renamed']}   errors: {s['errors']}"
          + (f"   ({mb:.1f} MB)" if applied else ""))
    if not applied:
        print("  dry-run: nothing copied. Re-run with --apply to perform the copies.")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_lrc(audio_path: str, synced: str):
    lrc = os.path.splitext(audio_path)[0] + ".lrc"
    with open(lrc, "w", encoding="utf-8") as fh:
        fh.write(synced)


def _print_diff(res: dict, r):
    tag = "DRY" if res.get("dry_run") else "SET"
    print(f"[{tag}] {res.get('path')}")
    for line in r.reasons[-2:]:
        print(f"      · {line}")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="audio_tagger", description="AI metadata harness for Indian music"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tag", help="scan, resolve and (optionally) write tags")
    t.add_argument("path")
    t.add_argument("--config", default=None, help="path to config.yaml (else ./config.yaml or defaults)")
    t.add_argument("--apply", action="store_true", help="write tags (default is dry-run)")
    t.add_argument("--threshold", type=float, default=None, help="auto-write confidence floor (overrides config)")
    t.add_argument("--limit", type=int, default=None, help="only process the first N scanned tracks")
    t.add_argument("--lyrics", action="store_true", help="also fetch lyrics (.lrc synced, else plain into tag)")
    t.add_argument("--no-fingerprint", action="store_true")
    t.add_argument("--review-out", default="review.csv")
    t.add_argument("--json-out", default=None, help="dump full resolutions to JSON for the review TUI")
    # --agent / --no-agent share a dest, default None -> decide by LLM ping.
    t.add_argument("--agent", dest="agent", action="store_true", default=None,
                   help="use the LLM agent judge as fallback (default: on if the LLM answers a ping)")
    t.add_argument("--no-agent", dest="agent", action="store_false",
                   help="never use the LLM agent judge")
    t.add_argument("--verify-audio", choices=["auto", "always", "off"], default="auto",
                   help="audio verification: auto (on gate-risk), always, or off")
    t.set_defaults(func=cmd_tag)

    rv = sub.add_parser("review", help="interactive review TUI over a path/run.json/review.json")
    rv.add_argument("source", help="a --json-out file, a saved review.json, or a library path")
    rv.add_argument("--config", default=None)
    rv.add_argument("--apply", action="store_true", help="write on accept (default is dry-run)")
    rv.set_defaults(func=cmd_review)

    a = sub.add_parser("apply-reviews", help="write approved rows from a review CSV")
    a.add_argument("csv")
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(func=cmd_apply_reviews)

    rb = sub.add_parser("rollback", help="restore original tags from backups")
    rb.add_argument("path")
    rb.set_defaults(func=cmd_rollback)

    og = sub.add_parser("organize",
                        help="copy tracks into <Album>/ folders (COPY-ONLY; originals untouched)")
    og.add_argument("path", help="a library path/file, or a tag --json-out file (with --from-json)")
    og.add_argument("--dest", required=True, help="destination library root (copies are written here)")
    og.add_argument("--from-json", action="store_true",
                    help="treat path as a tag --json-out file and organize by the resolved album")
    og.add_argument("--tag-copies", action="store_true",
                    help="also write the resolved tags onto the COPIES (json mode + --apply)")
    og.add_argument("--apply", action="store_true", help="perform the copies (default is a dry-run plan)")
    og.add_argument("--verbose", action="store_true", help="print each planned/copied file")
    og.set_defaults(func=cmd_organize)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
