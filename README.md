# audio-tagger

An AI metadata harness for **Indian** music libraries — Bollywood film songs,
Punjabi, and indipop — where generic taggers fall down.

It fixes the three things that go wrong with mainstream taggers on this
catalogue:

1. **The "greatest hits" trap.** Songs get filed under *"XYZ's Superhits"* or
   *"Bollywood Party Mix"* because that compilation is the first result. This
   harness explicitly prefers the **original film soundtrack / album** and
   penalizes compilations.
2. **Collaborations flattened into one name.** Multiple playback singers,
   the music director, and the lyricist are separate roles — they land in
   separate tags, not mashed into `ARTIST`.
3. **One film split into many albums.** Players group by *(album +
   album-artist)*; when singers differ per track the film shatters. An
   **album-reconciliation pass** writes one consistent album-artist + album +
   compilation flag across the whole film so it stays together.

## Why this design (and not "LLM web-searches each song")

The reliable part of tagging — *which recording is this* — is solved by
**structured databases + acoustic fingerprints**, not by free-text web
search. Web search gives prose; MusicBrainz gives stable IDs for artists,
release-groups, recordings, and roles. So the LLM is **not** the matcher.

Instead:

```
 scan files ──► structured cascade ──► rank (prefer soundtrack) ──► agree?
   (tags,         MusicBrainz              beat "greatest hits"       │
   fingerprint)   iTunes                                              ├─ yes ─► confidence gate ─► write / review
                  Spotify (key)                                       │
                  Discogs (key)                                       └─ no / empty ─► WEB-SEARCH FALLBACK
                                                                           (LLM extracts from snippets,
                                                                            constrained to abstain, capped
                                                                            score) ─► re-rank ─► gate
```

The LLM only ever **chooses among / extracts** candidates and is capped so it
can nudge but never override a solid structured match. It never invents
metadata — the single biggest source of brittleness in naive approaches.

Fingerprinting (AcoustID) is only ~85% accurate on Indian film music vs.
near-perfect on Western pop, so the residual is real — which is exactly why
low-confidence matches go to a **review queue** instead of silently onto your
files.

## Tag model (your rules, encoded in `tagmap.py` + `albumize.py`)

| Tag | Value |
|-----|-------|
| `ARTIST` | playback singer(s) — multi-valued, real collaborations preserved |
| `ALBUMARTIST` | music director/composer → producer → mixer (decided **once per album**) |
| `ALBUM` | film soundtrack (film song) · else the album · else the song title (true single) |
| `COMPOSER` | music director |
| `LYRICIST` | lyricist |
| `COMPILATION` (`TCMP`/`cpil`) | set to 1 on multi-artist albums so players keep the film as one album |

## Sources

| Source | Key? | Strength for this library |
|--------|------|---------------------------|
| MusicBrainz | no | only source with structured singer/composer/lyricist roles + soundtrack type |
| iTunes Search | no | strong Bollywood/Punjabi/indipop coverage, names the film album |
| Spotify | id/secret | modern Bollywood/Punjabi, distinct collaboration artists |
| Discogs | token | older HMV/Saregama/T-Series pressings, rich vintage credits |
| **Web search** | pluggable | **fallback tie-breaker** on disagreement/empty results |
| Lyrics: LRCLIB → Genius | no / token | synced `.lrc` first, Genius (Hindi/Punjabi) fallback |

The harness runs with **just the two keyless sources** (MusicBrainz + iTunes);
everything else is additive.

## Install

```bash
pip install -e .                 # core (MusicBrainz + iTunes + writing)
pip install -e ".[fingerprint]"  # + AcoustID  (also needs the `fpcalc` binary)
pip install -e ".[spotify]"      # + Spotify/Discogs HTTP
pip install -e ".[fallback]"     # + example web-search fallback deps
```

Optional keys (sources self-disable without them):

```bash
export SPOTIFY_CLIENT_ID=...  SPOTIFY_CLIENT_SECRET=...
export DISCOGS_TOKEN=...
export GENIUS_TOKEN=...
```

To enable the web-search fallback, copy `audio_tagger/runtime_fallback.py.example`
to `runtime_fallback.py` and wire in your search + LLM callables (an Anthropic +
DuckDuckGo example is included).

## Use

```bash
# 1. Dry run — see what it WOULD do, nothing is written:
python -m audio_tagger tag /path/to/music

# 2. Apply high-confidence matches, queue the rest, fetch synced lyrics:
python -m audio_tagger tag /path/to/music --apply --lyrics

# 3. Eyeball review.csv, set approved=1 (fix any field), then:
python -m audio_tagger apply-reviews review.csv

# Undo everything (restores tags from per-file .tags.bak.json backups):
python -m audio_tagger rollback /path/to/music
```

Every write is backed up to a `*.tags.bak.json` sidecar first, so it is fully
reversible.

## beets (optional heavy engine)

`config/beets-config.yaml` is a conservative beets setup (fingerprint-first,
`timid: yes`, original-year preferred, no silent auto-accept) if you'd rather
use beets for the file management and this harness for the Indian-specific
disambiguation. See the comments in that file.

## Layout

```
audio_tagger/
  models.py       normalized Candidate / Resolution shapes
  scan.py         read files + AcoustID fingerprint
  sources/        musicbrainz · itunes · spotify · discogs · websearch(fallback)
  heuristics.py   release-preference scoring (the "greatest hits" fix)
  resolve.py      cross-source agreement + fallback + confidence gating
  tagmap.py       your role/album tag rules
  albumize.py     album reconciliation (the "one film, one album" fix)
  lyrics.py       LRCLIB -> Genius
  writer.py       reversible multi-valued tag writing (mutagen)
  review.py       CSV review queue
  cli.py          `tag` / `apply-reviews` / `rollback`
tests/            unit tests for all decision logic (no network)
```

## Status

Decision logic (heuristics, tag mapping, album reconciliation, resolver
agreement + fallback) is implemented and unit-tested (`pytest`, 12 tests, no
network). The source adapters make real API calls but haven't been run against
a live library here — point it at a small sample in `--dry-run` first.
