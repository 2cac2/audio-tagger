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
   the music director(s), and the lyricist are separate roles — they land in
   separate tags, not mashed into `ARTIST`.
3. **One film split into many albums.** Players group by *(album +
   album-artist)*; when singers differ per track the film shatters. An
   **album-reconciliation pass** writes one consistent album-artist + album
   across the whole film so it stays together.

## Architecture — structured sources first, LLM as judge

Research on this niche is clear: a *pure* web-search LLM tagger is the wrong
shape. Web results for Bollywood songs are polluted with lyric-site SEO spam and
cover-orchestra "greatest hits", and an LLM working from a filename will
confidently tag a cover as the original. The reliable part of tagging — *which
recording is this* — is solved by **structured databases + acoustic
fingerprints**, not free-text search. So the LLM is **not** the matcher.

```
scan files (tags + AcoustID fingerprint)
   │
   ▼
structured cascade ──► AcoustID → MusicBrainz (recording MBIDs seed MB,
   │                    JioSaavn (Hindi film ground truth), iTunes, Deezer (ISRC)
   ▼
translit-normalized similarity ──► rank (prefer soundtrack, penalize compilations)
   │
   ▼
agreement (tightened) ──► confidence gate
   │                         conf ≥ 0.80  AND  (≥2 independent agreeing sources
   │                                            OR audio-verified)
   ├─ PASS ─► tagmap → albumize → lyrics → write (dry-run by default)
   │
   └─ FAIL ─► audio verify (25s clip → omni LLM: language / vocalist gender+count)
              LLM judge (agent loop + MCP web-search/fetch tools)
                 → validated JSON verdict: pick · new_candidate (capped 0.60) · abstain
              re-rank, re-gate → still failing → review queue (TUI / CSV)
```

The agent only ever **chooses among / corrects** structured candidates and its
confidence is **capped** (0.60): it breaks ties, never single-handedly
auto-applies. Audio verification is **evidence only** — it nudges scores ±0.1
and can satisfy the two-source floor, but never picks a winner alone.

Fingerprinting (AcoustID) is only ~85% accurate on Indian film music vs.
near-perfect on Western pop, so the residual is real — which is exactly why
low-confidence matches go to a **review queue** instead of silently onto your
files.

## Tag model (`tagmap.py` + `albumize.py`)

| Tag | Value |
|-----|-------|
| `ARTIST` | playback singer(s) — multi-valued, real collaborations preserved |
| `ALBUMARTIST` | composer act(s), decided **once per album**: 1 act → that act; 2–3 known acts → **all of them** as a multi-value field (never "Various Artists" for a film with known composers); >3 or none → single singer act, else "Various Artists" |
| `ALBUM` | film soundtrack (film song) · else the album · else the song title (true single) |
| `COMPOSER` | music director(s) — composer duos/trios folded to the credited act (Vishal–Shekhar, Sachin–Jigar, Shankar–Ehsaan–Loy, …) |
| `LYRICIST` | lyricist |
| `COMPILATION` (`TCMP`/`cpil`) | set to 1 **only** when the release itself is a compilation — varying playback singers across a film is normal and does **not** set it. Album cohesion comes from identical `ALBUM` + `ALBUMARTIST` + album key. |

## Sources

| Source | Key? | Strength for this library |
|--------|------|---------------------------|
| AcoustID | free key | fingerprint → the *actual* recording MBID (catches covers/remasters); seeds MusicBrainz |
| MusicBrainz | no | structured singer/composer/lyricist roles (via work-rels `includes`) + soundtrack type |
| JioSaavn | self-hosted API | ground truth for Hindi film music: film-as-album, year, label, music director, per-song language |
| iTunes Search | no | strong Bollywood/Punjabi/indipop coverage, names the film album |
| Deezer | no | keyless fallback + **ISRC** bridge for cross-source recording identity |
| Spotify / Discogs | key | **demoted / off by default** (Spotify's 2024–2026 dev lockdown makes it a liability) |
| Lyrics: LRCLIB | no | synced `.lrc` first; plain lyrics written into the tag |

The harness runs with the keyless sources alone; everything else is additive and
each source self-disables without its key.

## Install

```bash
pip install -e .                 # core (MusicBrainz + iTunes + writing + config)
pip install -e ".[sources]"      # + HTTP sources (JioSaavn, Deezer, Discogs, ...)
pip install -e ".[fingerprint]"  # + AcoustID  (also needs the `fpcalc` binary)
pip install -e ".[agent]"        # + local LLM judge (openai) + MCP toolbox
pip install -e ".[tui]"          # + interactive review TUI (textual)
pip install -e ".[dev]"          # + pytest
```

Every third-party library is lazy-imported, so the package imports (and the full
offline test suite runs) with only the standard library present.

Configuration lives in `config.yaml` (see `config/config.example.yaml`). API keys
and endpoints can also come from the environment:

```bash
export ACOUSTID_API_KEY=...
export LLM_BASE_URL=...  LLM_MODEL=...   # local vLLM / llama.cpp OpenAI endpoint
export SPOTIFY_CLIENT_ID=...  SPOTIFY_CLIENT_SECRET=...  DISCOGS_TOKEN=...
```

See **[docs/SETUP.md](docs/SETUP.md)** for the full stack: launching a local omni
LLM (vLLM / llama.cpp), the AcoustID free-key signup, MusicBrainz etiquette,
self-hosting the JioSaavn API, and the `docker/docker-compose.yml` bring-up
(SearXNG + search MCP, Firecrawl, jiosaavn-api).

## Use

```bash
# 1. Dry run — see what it WOULD do, nothing is written:
python -m audio_tagger tag /path/to/music

# 2. Apply high-confidence matches, queue the rest, fetch lyrics,
#    let the agent judge break ties and dump resolutions for the TUI:
python -m audio_tagger tag /path/to/music --apply --lyrics \
    --verify-audio auto --json-out run.json

# 3. Interactive review of the held tracks (steer the agent per track):
python -m audio_tagger review run.json          # or a library path / review.json
python -m audio_tagger review run.json --apply  # write on accept

# 4. Or script it: eyeball review.csv, set approved=1, then:
python -m audio_tagger apply-reviews review.csv

# Undo everything (restores tags from per-file .tags.bak.json backups):
python -m audio_tagger rollback /path/to/music
```

`tag` flags: `--config`, `--apply`, `--threshold`, `--limit N`,
`--lyrics`, `--no-fingerprint`, `--review-out`, `--json-out`,
`--agent`/`--no-agent` (default on when the LLM `/models` ping succeeds),
`--verify-audio {auto,always,off}`.

### Review TUI

`python -m audio_tagger review <path | run.json | review.json>` opens a two-pane
Textual reviewer: a **queue** of every `needs_review` track (status glyph,
confidence, short reason) and a **detail** pane showing your file's current tags
beside each candidate, the resolver's reasons log, and the audio-verify report.
Low-confidence and agent proposals are shown ranked but **never auto-applied** —
you pick. A per-track **steering box** ("this is the film version from Rockstar
(2011), singer is Mohit Chauhan") calls the agent with your hint and merges the
corrected (still confidence-capped) candidate back in. Keys: `a` accept, `e` edit
fields, `s` skip, `p` play a 25s clip, `w` write all accepted, `q` quit (session
saved to `review.json` to resume).

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
  config.py       HarnessConfig (YAML + env overrides)
  scan.py         read files + AcoustID fingerprint
  translit.py     roman-Hindi normalization + similarity (humein/hume/humey → hame)
  credits.py      composer-duo folding + cross-source role-tagged credit merge
  sources/        acoustid · musicbrainz · jiosaavn · itunes · deezer · spotify · discogs
  heuristics.py   release-preference scoring (the "greatest hits" fix)
  resolve.py      cross-source agreement + confidence gating + agent/verify hooks
  verify/         audio_clip (ffmpeg 25s clip) · omni (LLM language/vocalist check)
  agent/          llm · mcp_client · loop (bounded LLM judge) · schema (verdict validation)
  tagmap.py       role/album tag rules
  albumize.py     album reconciliation (multi-composer ALBUMARTIST, comp-flag fix)
  lyrics.py       LRCLIB synced lyrics
  writer.py       reversible multi-valued tag writing + lyrics tag (mutagen)
  review.py       CSV review queue (round-trips year/MBIDs/comp/film)
  tui/            state (session model) · app (Textual review UI)
  cli.py          `tag` / `review` / `apply-reviews` / `rollback`
tests/            offline unit tests for all decision logic (no network, fakes)
docs/SETUP.md     stack bring-up (LLM, AcoustID, MusicBrainz, JioSaavn, docker)
```

## Status

Decision logic (translit, credits, heuristics, tag mapping, album
reconciliation, resolver gating, agent loop, audio-verify consistency, config,
review round-trip) is implemented and unit-tested — **73 offline tests, no
network**, driven with fakes. The source adapters make real API calls but should
be pointed at a small sample in dry-run first.
