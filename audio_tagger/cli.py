"""Command-line entry point.

    python -m audio_tagger tag   /music --dry-run
    python -m audio_tagger tag   /music --apply           # writes tags + review.csv
    python -m audio_tagger apply-reviews review.csv        # writes approved rows
    python -m audio_tagger rollback /music                 # restore from backups

Sources activate based on available deps / env keys; the harness runs with
whatever is present (at minimum MusicBrainz + iTunes, both keyless).
"""

from __future__ import annotations

import argparse
import sys

from .albumize import reconcile_albums
from .lyrics import fetch_lyrics
from .resolve import Resolver
from .review import read_approvals, write_queue
from .scan import scan_path
from .writer import rollback, write_tags


def _build_sources(no_fingerprint: bool = False):
    sources = []
    # Keyless sources first.
    try:
        from .sources.musicbrainz import MusicBrainzSource
        sources.append(MusicBrainzSource())
    except Exception as e:
        print(f"[warn] musicbrainz disabled: {e}", file=sys.stderr)
    try:
        from .sources.itunes import ITunesSource
        sources.append(ITunesSource())
    except Exception as e:
        print(f"[warn] itunes disabled: {e}", file=sys.stderr)
    # Keyed sources self-disable if creds are missing.
    try:
        from .sources.spotify import SpotifySource
        sp = SpotifySource()
        if sp.enabled:
            sources.append(sp)
    except Exception:
        pass
    try:
        from .sources.discogs import DiscogsSource
        dc = DiscogsSource()
        if dc.enabled:
            sources.append(dc)
    except Exception:
        pass
    return sources


def _build_fallback():
    """Wire the web-search fallback if the runtime provides search+llm callables.

    In a Claude Code / MCP context you would pass real callables here. Absent
    that wiring we return None and the harness simply routes disagreements to
    the review queue instead of the web.
    """
    try:
        from .runtime_fallback import make_fallback  # user-provided, optional
        return make_fallback()
    except Exception:
        return None


def cmd_tag(args):
    tracks = scan_path(args.path, fingerprint=not args.no_fingerprint)
    print(f"scanned {len(tracks)} track(s)")
    resolver = Resolver(
        sources=_build_sources(args.no_fingerprint),
        websearch_fallback=_build_fallback(),
        auto_threshold=args.threshold,
    )
    resolutions = [resolver.resolve(t) for t in tracks]
    resolutions = reconcile_albums(resolutions)

    auto = [r for r in resolutions if not r.needs_review]
    print(f"auto-write: {len(auto)}   review: {len(resolutions) - len(auto)}")

    for r in resolutions:
        if r.needs_review or not r.chosen:
            continue
        if args.lyrics:
            got = fetch_lyrics(r.chosen, r.track.duration_sec)
            if got and got.get("synced"):
                _save_lrc(r.track.path, got["synced"])
        res = write_tags(r.track.path, r.tags, dry_run=not args.apply)
        _print_diff(res, r)

    n = write_queue(resolutions, args.review_out)
    print(f"wrote {args.review_out} ({n} rows to review)")
    if not args.apply:
        print("dry-run: no files changed. Re-run with --apply to write.")


def cmd_apply_reviews(args):
    approvals = read_approvals(args.csv)
    for path, fields in approvals.items():
        tags = {k: v for k, v in fields.items() if v}
        res = write_tags(path, tags, dry_run=args.dry_run)
        print(res.get("path"), "->", "dry-run" if args.dry_run else "written")
    print(f"applied {len(approvals)} approved row(s)")


def cmd_rollback(args):
    tracks = scan_path(args.path, fingerprint=False)
    restored = sum(1 for t in tracks if rollback(t.path))
    print(f"rolled back {restored}/{len(tracks)} file(s)")


def _save_lrc(audio_path: str, synced: str):
    import os
    lrc = os.path.splitext(audio_path)[0] + ".lrc"
    with open(lrc, "w", encoding="utf-8") as fh:
        fh.write(synced)


def _print_diff(res: dict, r):
    tag = "DRY" if res.get("dry_run") else "SET"
    print(f"[{tag}] {res.get('path')}")
    for line in r.reasons[-2:]:
        print(f"      · {line}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="audio_tagger", description="AI metadata harness for Indian music")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tag", help="scan, resolve and (optionally) write tags")
    t.add_argument("path")
    t.add_argument("--apply", action="store_true", help="write tags (default is dry-run)")
    t.add_argument("--threshold", type=float, default=0.80, help="auto-write confidence floor")
    t.add_argument("--lyrics", action="store_true", help="also fetch synced lyrics")
    t.add_argument("--no-fingerprint", action="store_true")
    t.add_argument("--review-out", default="review.csv")
    t.set_defaults(func=cmd_tag)

    a = sub.add_parser("apply-reviews", help="write approved rows from a review CSV")
    a.add_argument("csv")
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(func=cmd_apply_reviews)

    rb = sub.add_parser("rollback", help="restore original tags from backups")
    rb.add_argument("path")
    rb.set_defaults(func=cmd_rollback)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
