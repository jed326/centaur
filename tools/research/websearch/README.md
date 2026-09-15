# Websearch Plugin

Web search and deep research through a swappable backend, with optional Claude
synthesis on top of `search` results. Parallel is the only backend today.

| Backend (`WEBSEARCH_BACKEND`) | `search`, no key | `search`, keyed | `deep-research` (keyed only) |
| --- | --- | --- | --- |
| `parallel` (default) | Parallel's free Search MCP | Parallel Search REST (`PARALLEL_API_KEY`) | Parallel Task API (`PARALLEL_API_KEY`) |

`ANTHROPIC_API_KEY` adds the Claude reviewer → writer → citation-repair
pipeline over whatever the backend retrieved (`search` only). Without it,
`search` returns raw results and records the skipped synthesis in
`meta.partial_failures`.

## How the backend is chosen

Two facts decide a call:

- **Vendor** comes from the `WEBSEARCH_BACKEND` environment variable (set it
  in the chart's `sandbox.extraEnv`). Unset means `parallel`. Unknown values
  fail at startup with the accepted values in the message.
- **Keyed or anonymous** comes from the principal's grant. The tool sends the
  injected placeholder to the vendor's REST endpoint; a 401 means the key isn't
  granted, and the tool falls back to that vendor's anonymous path for the rest
  of the process.

`meta.backend` reports which path served the call: `parallel:api`,
`parallel:mcp`, or `parallel:task:<processor>`.

On the free-MCP path the response carries a `meta.attribution` string
("Search powered by the free Parallel Web Search MCP …") that `--pretty`
surfaces. Retain or display it when you redistribute free-tier results. See
<https://parallel.ai/customer-terms>.

## Quickstart

```python
from websearch.client import WebSearchClient

client = WebSearchClient()
result = await client.search("US GDP growth since 2020")
# meta.backend is 'parallel:mcp' with no key, 'parallel:api' with a granted PARALLEL_API_KEY
```

## Secrets

Set in root `.env` (preferred) or `tools/research/websearch/.env`.

- `PARALLEL_API_KEY` — optional. Unlocks Parallel Search REST (filters, `--effort`) and `deep-research` on the Task API. Get one at <https://platform.parallel.ai>.
- `ANTHROPIC_API_KEY` — optional. Enables the Claude synthesis pipeline on `search`.

Non-secret config (synthesis model, base URLs, the default Parallel processor) is set with `WebSearchClient(...)` kwargs. Defaults: `synthesis_model="claude-opus-4-6"`, `parallel_api_base_url="https://api.parallel.ai"`, `parallel_mcp_url="https://search.parallel.ai/mcp"`, `parallel_deep_research_processor="ultra-fast"`.

## Tools

### `search`

```python
await client.search("How should a fintech startup evaluate MPC vs HSM in 2026?", num_results=10)
```

- `query: str` — required.
- `effort: "instant" | "fast" | "deep"` — default `fast`. Parallel maps `instant` to `basic` and the others to `advanced`; `deep` is noted in `meta.partial_failures`. The anonymous path records it there too.
- `include_domains`, `exclude_domains: list[str]`, `max_age_hours: int` — keyed path only. `max_age_hours` rounds down to a UTC calendar date.
- `client_model`, `max_chars_total`, `session_id` — Parallel REST knobs.
- `synthesize: bool` — default `True`. Needs `ANTHROPIC_API_KEY`.

Results are `SourceDocument`s (`source_id`, `title`, `url`, `snippet`, `published_date`, `domain`).

### `deep-research`

```python
await client.deep_research(
    "How should a fintech startup evaluate MPC vs HSM in 2026?", effort="high"
)
```

- `effort: "medium" | "high"` — default `medium`. Parallel maps `medium` to `ultra-fast` and `high` to `ultra`.
- `timeout_seconds` — default per-processor. There is no cancel endpoint: a run that outlives the budget keeps running and keeps costing.

The run goes through the Task API with auto schema and is restricted to the `pro`/`ultra` processor family (`lite`/`base`/`core` raise a clear error pointing at the docs).

#### Processor cheatsheet

The hidden `--processor` flag overrides the `effort` mapping. Cost is per 1,000 runs.

| Processor       | Cost  | Latency band  | Use case |
| --------------- | -----:| -------------:| -------- |
| `pro-fast`      | $100  | 30s – 5min    | Quick research that still wants cross-source synthesis |
| `pro`           | $100  | 2min – 10min  | Same depth as `pro-fast`, less aggressive parallelism |
| `ultra-fast`    | $300  | 1min – 10min  | Default. Multi-source deep research with reasonable latency |
| `ultra`         | $300  | 5min – 25min  | Same depth, more time budget for harder questions |
| `ultra2x` … `ultra8x` | $600 – $2400 | 1min – 2hr | The most difficult deep research; rarely needed |

`-fast` variants are 2–5× faster than their non-fast siblings at the same price. See [Parallel pricing](https://docs.parallel.ai/getting-started/pricing) for the full table.

## CLI

```bash
# No credentials: Parallel's free Search MCP
websearch search "Recent funding for AI search startups"

# With a granted PARALLEL_API_KEY: REST path with filters
websearch search "Recent funding for AI search startups" \
  --include-domain techcrunch.com --include-domain reuters.com \
  --max-age-hours 720 --effort fast --pretty

# Deep research on the Parallel Task API (requires PARALLEL_API_KEY)
websearch deep-research "comparison of L2 rollup economics" --effort high --pretty
```

### Backward-compatibility notes

- `--mode basic|advanced` still parses as a hidden flag and maps to `--effort instant|fast` with a deprecation note. Passing both is an error.
- `--processor` still parses as a hidden flag and overrides `--effort`.
- Hidden flags `--search-type`, `--max-iterations`, `--num-queries-per-iteration`, `--num-results-per-query` are accepted, warn, and are ignored.
- `meta.exa_request_ids` mirrors `meta.request_ids`. `DeepResearchResponse.iterations` stays a single synthetic entry.

`meta.estimated_cost_usd` is an estimate from Parallel's list prices: `0.0` on the free MCP path, per-search on REST, and the per-run processor price for `deep-research`.
