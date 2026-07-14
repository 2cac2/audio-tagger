# Setup — running the full audio-tagger stack

The harness works out of the box against the **keyless** sources (MusicBrainz +
iTunes + Deezer) with nothing running but Python. Everything below is additive
and only needed for the parts of the pipeline that improve accuracy on Hindi /
Punjabi / indipop libraries:

| Capability | What it needs |
|---|---|
| Hindi-film ground truth (album=film, composer, singers) | `jiosaavn-api` container |
| Acoustic fingerprint → recording identity | a free AcoustID API key |
| LLM judge + audio verification | a local OpenAI-compatible omni server (vLLM or llama.cpp) |
| Agent web-search / page-fetch fallback | SearXNG + Firecrawl containers (+ their MCP shims) |

All service endpoints are wired through [`config/config.example.yaml`](../config/config.example.yaml).
Copy it once and edit as needed:

```bash
cp config/config.example.yaml config.yaml
```

Secrets (API keys/tokens) are better supplied as environment variables — they
override the file. See the bottom of `config.example.yaml` for the full list.

---

## 1. Supporting services (Docker)

The compose file lives at [`docker/docker-compose.yml`](../docker/docker-compose.yml).

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml ps
```

### Services and ports

| Service | Host port | Purpose | config.yaml key |
|---|---|---|---|
| `searxng` | 8080 | web-search JSON API (behind the shim) | — |
| `searxng-mcp` | 8081 | SearXNG → MCP shim (SSE) | `mcp.servers.websearch` → `http://localhost:8081/sse` |
| `firecrawl-api` | 3002 | Firecrawl scrape/crawl HTTP API | — |
| `firecrawl-playwright` | — | headless browser renderer (internal) | — |
| `firecrawl-redis` | — | Firecrawl queue/cache (internal) | — |
| `firecrawl-worker` | — | Firecrawl job worker (internal) | — |
| `firecrawl-mcp` | 8082 | Firecrawl → MCP shim (SSE) | `mcp.servers.firecrawl` → `http://localhost:8082/sse` |
| `jiosaavn-api` | 3500 | unofficial JioSaavn API | `sources.jiosaavn.base_url` → `http://localhost:3500` |

The port numbers above are the defaults already written into
`config.example.yaml`, so a plain `cp` gets you a working config.

### SearXNG: enable the JSON API

The agent's search tool needs SearXNG's JSON output, which is **off by default**.
On first boot SearXNG writes `docker/searxng/settings.yml`. Edit it so the search
formats include `json`:

```yaml
search:
  formats:
    - html
    - json
```

Then restart just that service:

```bash
docker compose -f docker/docker-compose.yml restart searxng
```

Quick check that JSON works and the shim can reach it:

```bash
curl 'http://localhost:8080/search?q=lagaan+soundtrack&format=json' | head
```

### The MCP shims

SearXNG and Firecrawl speak plain HTTP, not MCP. The `searxng-mcp` and
`firecrawl-mcp` services are thin bridges that expose each as MCP tools over SSE
on 8081 / 8082 — the URLs the harness' agent connects to. If a shim image name
drifts, swap it in the compose file for any equivalent bridge; the only contract
that matters is "MCP over SSE at that port." If the MCP servers are unreachable
the agent **degrades gracefully to tool-less judging** — it just loses the
web-search/fetch fallback, the rest of the pipeline is unaffected.

### JioSaavn

`jiosaavn-api` (the `sumitkolhe/jiosaavn-api` image) listens on 3000 inside the
container; the compose maps it to host **3500** to match the config. Verify:

```bash
curl 'http://localhost:3500/api/search/songs?query=chaiyya+chaiyya' | head
```

---

## 2. Local LLM server (omni model)

The judge and the 25s audio-verification step both call one OpenAI-compatible
`/v1` endpoint. Point `llm.base_url` in `config.yaml` at it (default
`http://localhost:8000/v1`). An **omni** model (text **and** audio input) is
required for audio verification; a text-only model still works as the judge —
set `llm.supports_audio_input: false` to disable audio verify.

### Option A — vLLM (recommended)

```bash
pip install "vllm>=0.6"

vllm serve Qwen/Qwen3.5-Omni-7B \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

- `--enable-auto-tool-choice` + `--tool-call-parser hermes` are what let the
  agent loop use the MCP search/fetch tools. Match the parser to your model's
  chat template (`hermes` suits the Qwen family); a mismatched parser is the
  usual cause of the agent never calling a tool.
- Confirm it's up: `curl http://localhost:8000/v1/models`.

### Option B — llama.cpp

```bash
# Serves an OpenAI-compatible API with tool-calling via the Jinja chat template.
llama-server \
  --model /path/to/qwen3.5-omni.gguf \
  --port 8000 \
  --jinja
```

- `--jinja` enables the model's chat template, which is required for
  tool-calling to work.
- **Audio input caveat:** a stock `llama-server` build is text-only. Audio-in
  (needed for the verify step) requires an **omni-capable build** with the
  matching multimodal projector / mmproj file loaded. Without it, keep
  `supports_audio_input: false` and llama.cpp still serves as the text judge.

Whichever you run, set `llm.model` in `config.yaml` to the exact model id the
server reports at `/v1/models`.

---

## 3. AcoustID (free API key)

Fingerprinting maps the actual audio to a MusicBrainz recording, which catches
covers and remasters that filename-based matching gets wrong.

1. Sign up at <https://acoustid.org/> and register an application at
   <https://acoustid.org/new-application> to get a free **API key**.
2. Provide it via env (preferred) or config:

   ```bash
   export ACOUSTID_KEY=your_key_here
   ```

   or `sources.acoustid.api_key` in `config.yaml`.
3. Install the fingerprint tool `fpcalc` (Chromaprint), e.g.
   `apt install libchromaprint-tools` / `brew install chromaprint`.

The AcoustID source **self-disables** if no key is present — the rest of the
pipeline runs without it, just without fingerprint-verified identity.

---

## 4. MusicBrainz etiquette

MusicBrainz is keyless but rate-limited: **do not exceed ~1 request per second**
from a single IP (<https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting>).
The harness cooperates by:

- caching lookups in-memory and on disk (`~/.cache/audio-tagger/`) so re-runs
  don't re-hit the API, and
- limiting per-track detail (`get-by-id` with `includes`) lookups to the top few
  recordings.

Still, be a good citizen: send a descriptive User-Agent, run large libraries in
reasonable batches, and don't disable the cache. Repeated abuse gets your IP
throttled or blocked for everyone.

---

## 5. Risk note — JioSaavn is an unofficial API

`jiosaavn-api` (`sumitkolhe/jiosaavn-api`) is a community wrapper around
**private, undocumented** JioSaavn endpoints. There is **no official API and no
SLA**:

- The upstream shape can change without notice, breaking search/detail parsing.
  The adapter is deliberately thin and behind fixtures so breakage is isolated
  and offline tests still pass.
- Availability and terms are outside our control; use is at your own risk and
  subject to JioSaavn's terms. Prefer self-hosting the container (as configured
  here) over hammering any public instance.
- If JioSaavn is down or drifts, the harness keeps working on the other sources
  (MusicBrainz + iTunes + Deezer) — you just lose the strongest Hindi-film
  ground truth, so more tracks land in the review queue.

---

## 6. Putting it together

```bash
# 1. Services
docker compose -f docker/docker-compose.yml up -d
#    (enable SearXNG JSON, then: docker compose ... restart searxng)

# 2. LLM
vllm serve Qwen/Qwen3.5-Omni-7B --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes

# 3. Config + keys
cp config/config.example.yaml config.yaml
export ACOUSTID_KEY=your_key_here

# 4. Dry-run on a small sample first
python -m audio_tagger tag ~/music-sample \
  --lyrics --verify-audio auto --json-out run.json
```

See the repo `README.md` for the tagging model and the day-to-day `tag` /
`review` / `apply-reviews` / `rollback` commands.
