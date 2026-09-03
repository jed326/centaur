"""Tako backend.

`search` calls `POST /api/v3/search` with the `TAKO_API_KEY` placeholder and,
when that returns 401, the anonymous `tako_search` tool on `mcp.tako.com`.
`deep_research` runs the Tako Answer Agent over SSE (keyed only). Transport is
`httpx`, which honors `HTTPS_PROXY` so iron-proxy can rewrite the header.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx
from httpx_sse import SSEError, aconnect_sse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ._common import (
    TOOL_VERSION,
    USER_AGENT,
    append_within_budget,
    decode_jsonrpc_response,
    render_sources_block,
)
from .models import (
    DeepResearchResult,
    DeepResearchSpec,
    RetrievalResult,
    SearchRequestSpec,
    SourceDocument,
)

API_BASE_URL = "https://tako.com"
SEARCH_PATH = "/api/v3/search"
AGENT_RUNS_PATH = "/api/v1/agent/answer/runs"
MCP_URL = "https://mcp.tako.com/mcp"
CALLER_HEADER = "X-Tako-Caller"
CALLER_VALUE = f'channel=centaur, client_version="{TOOL_VERSION}"'
SEARCH_PRICE_USD: dict[str, float] = {"instant": 0.007, "fast": 0.007, "deep": 0.012}
DEFAULT_SEARCH_EFFORT = "fast"
DEFAULT_RESEARCH_EFFORT = "medium"
DEFAULT_DEEP_RESEARCH_TIMEOUT_SECONDS = 600.0
STREAM_READ_TIMEOUT_SECONDS = 120.0
POLL_INTERVAL_SECONDS = 5.0
SNIPPET_CHAR_LIMIT = 7000
MAX_SOURCE_COUNT = 20
DATA_CARD_COUNT = 2
MAX_DOMAIN_FILTERS = 20
FALLBACK_CITATION_URL = "https://tako.com"
RATE_LIMIT_KINDS = {"rate_limited", "global_rate_limited"}
NOT_GRANTED_STATUSES = {401, 403}
TERMINAL_RUN_STATUSES = {"completed", "failed"}


class CardSource(BaseModel):
    source_name: str | None = None


class CardMethodology(BaseModel):
    methodology_name: str | None = None
    methodology_description: str | None = None


class MetricDefinition(BaseModel):
    name: str
    definition: str


class DataFreshness(BaseModel):
    last_updated: str | None = None


class TakoCard(BaseModel):
    """The fields of a v3 `TakoCard` this tool reads."""

    title: str | None = None
    description: str | None = None
    semantic_description: str | None = None
    webpage_url: str | None = None
    sources: list[CardSource] | None = None
    methodologies: list[CardMethodology] | None = None
    metric_definitions: list[MetricDefinition] | None = None
    data_freshness: DataFreshness | None = None


class TakoWebResult(BaseModel):
    title: str | None = None
    url: str
    snippet: str | None = None
    source_name: str | None = None
    publish_date: str | None = None


class Usage(BaseModel):
    """Tako's metered usage. Extra fields (`compute`, `data`) pass through `model_dump()`."""

    model_config = ConfigDict(extra="allow")

    total_cost_usd: float | None = None


class SearchApiResponse(BaseModel):
    """`POST /api/v3/search` response, the fields this tool reads."""

    cards: list[TakoCard] = Field(default_factory=list)
    web_results: list[TakoWebResult] = Field(default_factory=list)
    request_id: str | None = None
    usage: Usage | None = None


class ProjectedCard(BaseModel):
    """A card as the anonymous MCP worker projects it (slimmed, renamed fields)."""

    title: str | None = None
    description: str | None = None
    url: str | None = None
    source: str | None = None
    last_updated: str | None = None


class ProjectedWebResult(BaseModel):
    title: str | None = None
    url: str
    snippet: str | None = None
    source: str | None = None
    published: str | None = None


class AnonymousSearchOutput(BaseModel):
    """`tako_search` structuredContent from the anonymous MCP worker."""

    cards: list[ProjectedCard] = Field(default_factory=list)
    web_results: list[ProjectedWebResult] = Field(default_factory=list)
    usage: Usage | None = None
    metric_definitions: dict[str, str] = Field(default_factory=dict)
    source_notes: dict[str, str] = Field(default_factory=dict)


class Citation(BaseModel):
    index: int
    title: str
    url: str | None = None
    source_name: str | None = None
    excerpt: str | None = None
    publish_date: str | None = None


class Definition(BaseModel):
    term: str
    definition: str
    source_ref: int | None = None


class TitledNote(BaseModel):
    title: str
    description: str


class AnswerMetadata(BaseModel):
    definitions: list[Definition] | None = None
    assumptions: list[TitledNote] | None = None
    methodology: list[TitledNote] | None = None


class AnswerAgentResult(BaseModel):
    """The `agent_result` payload of an Answer Agent run."""

    answer: str | None = None
    cards: list[TakoCard] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    metadata: AnswerMetadata | None = None
    refusal_code: str | None = None
    request_id: str | None = None


class RunError(BaseModel):
    code: str
    message: str


class AnswerAgentRun(BaseModel):
    """The run resource from `GET /api/v1/agent/answer/runs/{run_id}`."""

    run_id: str
    status: str
    result: AnswerAgentResult | None = None
    error: RunError | None = None
    usage: Usage | None = None


class McpToolResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: list[dict[str, Any]] = Field(default_factory=list)
    structured_content: dict[str, Any] | None = Field(default=None, alias="structuredContent")
    is_error: bool = Field(default=False, alias="isError")
    meta: dict[str, Any] = Field(default_factory=dict, alias="_meta")

    def error_kind(self) -> str | None:
        tako_error = self.meta.get("tako/error")
        return tako_error.get("kind") if isinstance(tako_error, dict) else None

    def text(self) -> str:
        return " ".join(str(block.get("text", "")) for block in self.content).strip()


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    result: McpToolResult | None = None
    error: JsonRpcError | None = None


class StreamBlock(BaseModel):
    """One SSE block. Fields are the union of the kinds this tool reads; unknown kinds are ignored.

    `error` carries two shapes because two kinds send it: `run_summary` sends a
    `{code, message}` object, and `tool_retry` sends the tool's raw message as
    free text. A single-shape field rejects the whole frame.
    """

    kind: str
    message: str | None = None
    tool: str | None = None
    status_message: str | None = None
    done: bool = False
    subagent_type: str | None = None
    event: str | None = None
    data: dict[str, Any] | None = None
    status: str | None = None
    usage: Usage | None = None
    error: RunError | str | None = None


class StreamEnvelope(BaseModel):
    seq: int
    run_id: str
    block: StreamBlock


class _StreamState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str | None = None
    last_seq: int = -1
    status: str | None = None
    result: AnswerAgentResult | None = None
    usage: Usage | None = None
    error: RunError | None = None
    done: bool = False
    transport_error: Exception | None = None


class NotGranted(Exception):
    """The placeholder was not rewritten: this principal has no TAKO_API_KEY grant."""


def _skip_unreadable(progress: Callable[[str], None], what: str, exc: ValidationError) -> None:
    """Report a frame the tool cannot parse. The poll fallback still returns the run."""
    count = exc.error_count()
    progress(f"skipped an unreadable {what} ({count} field{'' if count == 1 else 's'})")


def _as_run_error(value: RunError | str | None) -> RunError | None:
    if value is None or isinstance(value, RunError):
        return value
    return RunError(code="unknown", message=value)


def _host(url: str) -> str | None:
    return urlparse(url).netloc or None


def _join_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line)[:SNIPPET_CHAR_LIMIT]


def _card_snippet(card: TakoCard) -> str:
    lines = [(card.description or card.semantic_description or "").strip()]
    for metric in card.metric_definitions or []:
        lines.append(f"{metric.name}: {metric.definition}")
    for method in card.methodologies or []:
        if method.methodology_name and method.methodology_description:
            lines.append(f"{method.methodology_name}: {method.methodology_description}")
    return _join_lines(lines)


def _card_domain(card: TakoCard, url: str) -> str | None:
    names = [source.source_name for source in card.sources or [] if source.source_name]
    return ", ".join(names) if names else _host(url)


def card_to_source(card: TakoCard, source_id: int) -> SourceDocument | None:
    if not card.webpage_url:
        return None
    return SourceDocument(
        source_id=source_id,
        title=card.title or card.webpage_url,
        url=card.webpage_url,
        snippet=_card_snippet(card),
        published_date=card.data_freshness.last_updated if card.data_freshness else None,
        domain=_card_domain(card, card.webpage_url),
    )


def _web_to_source(item: TakoWebResult, source_id: int) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        title=item.title or item.url,
        url=item.url,
        snippet=(item.snippet or "")[:SNIPPET_CHAR_LIMIT],
        published_date=item.publish_date,
        domain=item.source_name or _host(item.url),
    )


def normalize_search_response(payload: SearchApiResponse) -> list[SourceDocument]:
    """Cards first, then web results, deduplicated by URL and numbered by position."""
    sources: list[SourceDocument] = []
    seen: set[str] = set()
    for card in payload.cards:
        document = card_to_source(card, len(sources))
        if document is None or document.url in seen:
            continue
        seen.add(document.url)
        sources.append(document)
    for item in payload.web_results:
        if item.url in seen:
            continue
        seen.add(item.url)
        sources.append(_web_to_source(item, len(sources)))
    return sources


def normalize_anonymous_output(payload: AnonymousSearchOutput) -> list[SourceDocument]:
    """Same order as `normalize_search_response`, from the worker's projected shape."""
    definition_lines = [f"{name}: {text}" for name, text in payload.metric_definitions.items()]
    sources: list[SourceDocument] = []
    seen: set[str] = set()
    for card in payload.cards:
        if not card.url or card.url in seen:
            continue
        lines = [(card.description or "").strip(), *definition_lines]
        if card.source and card.source in payload.source_notes:
            lines.append(f"{card.source}: {payload.source_notes[card.source]}")
        seen.add(card.url)
        sources.append(
            SourceDocument(
                source_id=len(sources),
                title=card.title or card.url,
                url=card.url,
                snippet=_join_lines(lines),
                published_date=card.last_updated,
                domain=card.source or _host(card.url),
            )
        )
    for item in payload.web_results:
        if item.url in seen:
            continue
        seen.add(item.url)
        sources.append(
            SourceDocument(
                source_id=len(sources),
                title=item.title or item.url,
                url=item.url,
                snippet=(item.snippet or "")[:SNIPPET_CHAR_LIMIT],
                published_date=item.published,
                domain=item.source or _host(item.url),
            )
        )
    return sources


def _section(heading: str, lines: list[str]) -> str:
    if not lines:
        return ""
    return f"\n\n## {heading}\n" + "\n".join(lines)


def normalize_answer_result(
    result: AnswerAgentResult, *, max_report_chars: int
) -> tuple[list[SourceDocument], str]:
    """Citations keep their `[n]` indexes; uncited cards follow; the report gains
    Charts, Definitions, Assumptions, Methodology, and Sources sections."""
    card_url_by_title = {
        card.title: card.webpage_url for card in result.cards if card.title and card.webpage_url
    }
    sources: list[SourceDocument] = []
    seen_indexes: set[int] = set()
    for citation in result.citations:
        if citation.index in seen_indexes:
            continue
        seen_indexes.add(citation.index)
        url = citation.url or card_url_by_title.get(citation.title) or FALLBACK_CITATION_URL
        sources.append(
            SourceDocument(
                source_id=citation.index,
                title=citation.title,
                url=url,
                snippet=(citation.excerpt or "")[:SNIPPET_CHAR_LIMIT],
                published_date=citation.publish_date,
                domain=citation.source_name or _host(url),
            )
        )
    cited_urls = {source.url for source in sources}
    next_id = max((source.source_id for source in sources), default=0) + 1
    for card in result.cards:
        if not card.webpage_url or card.webpage_url in cited_urls:
            continue
        document = card_to_source(card, next_id)
        if document is None:
            continue
        sources.append(document)
        cited_urls.add(document.url)
        next_id += 1

    body = (result.answer or "").strip()
    body += _section(
        "Charts",
        [
            f"- {card.title or 'Chart'}: {card.webpage_url}"
            for card in result.cards
            if card.webpage_url
        ],
    )
    metadata = result.metadata
    if metadata is not None:
        body += _section(
            "Definitions",
            [
                f"- **{d.term}**: {d.definition}"
                + (f" [{d.source_ref}]" if d.source_ref is not None else "")
                for d in metadata.definitions or []
            ],
        )
        body += _section(
            "Assumptions", [f"- **{a.title}**: {a.description}" for a in metadata.assumptions or []]
        )
        body += _section(
            "Methodology", [f"- **{m.title}**: {m.description}" for m in metadata.methodology or []]
        )
    if sources:
        return sources, append_within_budget(body, render_sources_block(sources), max_report_chars)
    return sources, body[:max_report_chars].rstrip()


class TakoBackend:
    """Tako search (keyed REST or anonymous MCP) and the Tako Answer Agent."""

    def __init__(
        self,
        *,
        api_key: str | None,
        api_base_url: str = API_BASE_URL,
        mcp_url: str = MCP_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_base_url = api_base_url.rstrip("/")
        self._mcp_url = mcp_url
        self._transport = transport
        self._rest_auth_failed = False

    @property
    def search_mode(self) -> str:
        return "api" if self._api_key and not self._rest_auth_failed else "anonymous"

    def _http(self, timeout: float | httpx.Timeout) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    def _keyed_headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self._api_key or "",
            CALLER_HEADER: CALLER_VALUE,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _anonymous_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
        }

    async def search(self, request: SearchRequestSpec) -> RetrievalResult:
        query = request.query.strip()
        if not query:
            raise RuntimeError("query cannot be empty.")
        partial_failures: list[dict[str, str]] = []
        if self._api_key and not self._rest_auth_failed:
            try:
                return await self._search_api(request, query)
            except NotGranted:
                self._rest_auth_failed = True
        if self._api_key and self._rest_auth_failed:
            partial_failures.append(
                {
                    "query": query,
                    "error": (
                        "TAKO_API_KEY did not authenticate; fell back to anonymous Tako "
                        "search. Configure a granted key to use the REST API."
                    ),
                }
            )
        return await self._search_anonymous(request, query, partial_failures)

    def _search_body(self, request: SearchRequestSpec, query: str) -> dict[str, Any]:
        count = max(1, min(MAX_SOURCE_COUNT, request.num_results))
        data_count = min(count, DATA_CARD_COUNT)
        web: dict[str, Any] = {"count": count, "highlights": True}
        if request.include_domains:
            web["include_domains"] = request.include_domains[:MAX_DOMAIN_FILTERS]
        if request.exclude_domains:
            web["exclude_domains"] = request.exclude_domains[:MAX_DOMAIN_FILTERS]
        if request.max_age_hours is not None and request.max_age_hours > 0:
            cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(hours=request.max_age_hours)
            web["published_after"] = cutoff.date().isoformat()
        body: dict[str, Any] = {
            "query": query,
            "sources": {"data": {"count": data_count}, "web": web},
        }
        if request.effort:
            body["effort"] = request.effort
        return body

    async def _search_api(self, request: SearchRequestSpec, query: str) -> RetrievalResult:
        async with self._http(request.timeout_seconds) as client:
            response = await client.post(
                f"{self._api_base_url}{SEARCH_PATH}",
                headers=self._keyed_headers(),
                json=self._search_body(request, query),
            )
        if response.status_code in NOT_GRANTED_STATUSES:
            raise NotGranted
        response.raise_for_status()
        payload = SearchApiResponse.model_validate(response.json())
        partial_failures: list[dict[str, str]] = []
        if request.max_chars_total is not None:
            partial_failures.append(
                {
                    "query": query,
                    "error": "max_chars_total is not supported by Tako search (its cap is per result); ignored.",
                }
            )
        parallel_only = [
            name
            for name, value in (
                ("client_model", request.client_model),
                ("session_id", request.session_id),
            )
            if value is not None
        ]
        if parallel_only:
            partial_failures.append(
                {
                    "query": query,
                    "error": f"{', '.join(parallel_only)} is a Parallel REST knob; Tako search ignores it.",
                }
            )
        billed = payload.usage.total_cost_usd if payload.usage else None
        effort = request.effort or DEFAULT_SEARCH_EFFORT
        return RetrievalResult(
            sources=normalize_search_response(payload),
            backend="tako:api",
            request_ids=[payload.request_id] if payload.request_id else [],
            usage=[payload.usage.model_dump()] if payload.usage else [],
            partial_failures=partial_failures,
            estimated_cost_usd=billed if billed is not None else SEARCH_PRICE_USD[effort],
        )

    async def _search_anonymous(
        self, request: SearchRequestSpec, query: str, partial_failures: list[dict[str, str]]
    ) -> RetrievalResult:
        ignored: list[str] = []
        if request.include_domains or request.exclude_domains or request.max_age_hours is not None:
            ignored.append("include_domains/exclude_domains/max_age_hours")
        if request.effort:
            ignored.append(f"effort={request.effort!r}")
        if request.max_chars_total is not None:
            ignored.append("max_chars_total")
        if request.client_model is not None:
            ignored.append("client_model")
        if request.session_id is not None:
            ignored.append("session_id")
        if request.num_results != 10:
            ignored.append(
                f"num_results={request.num_results} (anonymous search serves a fixed count; client-side cap only)"
            )
        if ignored:
            partial_failures.append(
                {
                    "query": query,
                    "error": (
                        f"Anonymous Tako search does not honor: {', '.join(ignored)}. "
                        "Set TAKO_API_KEY to use the REST API."
                    ),
                }
            )
        request_id = str(uuid.uuid4())
        envelope = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": "tako_search",
                "arguments": {"query": query, "sources": ["data", "web"]},
            },
        }
        async with self._http(request.timeout_seconds) as client:
            response = await client.post(
                self._mcp_url, headers=self._anonymous_headers(), json=envelope
            )
        if response.status_code == 429:
            reply = JsonRpcResponse.model_validate(decode_jsonrpc_response(response, request_id))
            detail = reply.error.message if reply.error else response.text[:200]
            raise RuntimeError(f"Anonymous Tako search is rate limited: {detail}")
        response.raise_for_status()
        reply = JsonRpcResponse.model_validate(decode_jsonrpc_response(response, request_id))
        if reply.error is not None:
            raise RuntimeError(f"Tako MCP error: {reply.error.message[:500]}")
        result = reply.result
        if result is None:
            raise RuntimeError("Tako MCP returned no result.")
        if result.is_error:
            if result.error_kind() in RATE_LIMIT_KINDS:
                raise RuntimeError(f"Anonymous Tako search is rate limited: {result.text()}")
            raise RuntimeError(f"Tako MCP tool error: {result.text()[:500]}")
        if result.structured_content is None:
            raise RuntimeError("Tako MCP returned no structuredContent.")
        payload = AnonymousSearchOutput.model_validate(result.structured_content)
        return RetrievalResult(
            sources=normalize_anonymous_output(payload),
            backend="tako:anonymous",
            usage=[payload.usage.model_dump()] if payload.usage else [],
            partial_failures=partial_failures,
            estimated_cost_usd=0.0,
        )

    async def deep_research(
        self, request: DeepResearchSpec, progress: Callable[[str], None]
    ) -> DeepResearchResult:
        if not self._api_key:
            raise RuntimeError(
                "deep_research requires TAKO_API_KEY. Anonymous Tako access covers `search` only."
            )
        question = request.question.strip()
        if not question:
            raise RuntimeError("question cannot be empty.")
        effort = request.effort or DEFAULT_RESEARCH_EFFORT
        timeout_seconds = (
            DEFAULT_DEEP_RESEARCH_TIMEOUT_SECONDS
            if request.timeout_seconds is None
            else request.timeout_seconds
        )
        partial_failures: list[dict[str, str]] = []
        if request.processor:
            partial_failures.append(
                {
                    "query": question,
                    "error": (
                        f"--processor={request.processor!r} is Parallel-only; the Tako Answer Agent "
                        f"ran with effort={effort!r}."
                    ),
                }
            )
        progress(f"dispatching answer agent (effort={effort}, timeout={int(timeout_seconds)}s)")
        try:
            run = await asyncio.wait_for(
                self._run_answer_agent(question, effort, progress), timeout=timeout_seconds
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"Tako answer agent run did not finish within {int(timeout_seconds)}s. "
                "The run keeps going server-side; Tako has no cancel endpoint."
            ) from exc
        if run.result is None:
            detail = f": {run.error.message}" if run.error else ""
            raise RuntimeError(f"Tako answer agent run {run.run_id} {run.status}{detail}")
        if run.result.refusal_code:
            raise RuntimeError(
                f"Tako declined the question before running (refusal_code={run.result.refusal_code})."
            )
        sources, answer_markdown = normalize_answer_result(
            run.result, max_report_chars=request.max_report_chars
        )
        if not answer_markdown:
            raise RuntimeError(f"Tako answer agent run {run.run_id} returned no content.")
        billed = run.usage.total_cost_usd if run.usage else None
        return DeepResearchResult(
            sources=sources,
            answer_markdown=answer_markdown,
            backend="tako:agent",
            request_ids=[run.run_id],
            partial_failures=partial_failures,
            usage=[run.usage.model_dump()] if run.usage else [],
            estimated_cost_usd=billed,
        )

    async def _run_answer_agent(
        self, question: str, effort: str, progress: Callable[[str], None]
    ) -> AnswerAgentRun:
        runs_url = f"{self._api_base_url}{AGENT_RUNS_PATH}"
        timeout = httpx.Timeout(
            connect=30.0, read=STREAM_READ_TIMEOUT_SECONDS, write=30.0, pool=30.0
        )
        state = _StreamState()
        async with self._http(timeout) as client:
            await self._consume_stream(
                client,
                "POST",
                runs_url,
                state,
                progress,
                json={"query": question, "effort": effort},
            )
            if not state.done and state.run_id:
                progress("stream dropped; resuming")
                params = {"starting_after": str(state.last_seq)} if state.last_seq >= 0 else None
                await self._consume_stream(
                    client, "GET", f"{runs_url}/{state.run_id}", state, progress, params=params
                )
            if state.result is not None and state.run_id:
                return AnswerAgentRun(
                    run_id=state.run_id,
                    status=state.status or "completed",
                    result=state.result,
                    error=state.error,
                    usage=state.usage,
                )
            if state.status == "failed" and state.run_id:
                return AnswerAgentRun(run_id=state.run_id, status="failed", error=state.error)
            if not state.run_id:
                raise RuntimeError(
                    "Tako answer agent stream ended before a run_id arrived."
                ) from state.transport_error
            return await self._poll_run(client, state.run_id, progress)

    async def _consume_stream(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        state: _StreamState,
        progress: Callable[[str], None],
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> None:
        headers = {**self._keyed_headers(), "Accept": "text/event-stream"}
        try:
            async with aconnect_sse(
                client, method, url, headers=headers, json=json, params=params
            ) as source:
                status_code = source.response.status_code
                if status_code == 401:
                    raise RuntimeError("deep_research requires a valid, granted TAKO_API_KEY.")
                if status_code >= 400:
                    body = (await source.response.aread()).decode(errors="replace")
                    raise RuntimeError(
                        f"Tako answer agent request failed ({status_code}): {body[:500]}"
                    )
                async for sse in source.aiter_sse():
                    if not sse.data.strip():
                        continue
                    try:
                        envelope = StreamEnvelope.model_validate_json(sse.data)
                    except ValidationError as exc:
                        _skip_unreadable(progress, "stream frame", exc)
                        continue
                    state.run_id = envelope.run_id
                    state.last_seq = envelope.seq
                    self._apply_block(envelope.block, state, progress)
                    if state.done:
                        return
        except (
            httpx.NetworkError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            SSEError,
        ) as exc:
            progress(f"stream interrupted: {exc}")
            state.transport_error = exc

    @staticmethod
    def _apply_block(
        block: StreamBlock, state: _StreamState, progress: Callable[[str], None]
    ) -> None:
        if block.kind == "status" and block.message:
            progress(block.message)
        elif block.kind == "tool_call" and block.tool:
            suffix = f": {block.status_message}" if block.status_message else ""
            progress(f"{'finished' if block.done else 'calling'} {block.tool}{suffix}")
        elif block.kind == "tool_retry" and block.tool:
            detail = f": {block.error}" if isinstance(block.error, str) else ""
            progress(f"retrying {block.tool}{detail}")
        elif block.kind == "subagent" and block.subagent_type:
            progress(f"{block.event or 'subagent'} {block.subagent_type}")
        elif block.kind == "agent_result" and block.data is not None:
            try:
                state.result = AnswerAgentResult.model_validate(block.data)
            except ValidationError as exc:
                _skip_unreadable(progress, "agent_result", exc)
        elif block.kind == "run_summary":
            state.status = block.status
            state.usage = block.usage
            state.error = _as_run_error(block.error)
        elif block.kind == "stream_done":
            state.done = True

    async def _poll_run(
        self, client: httpx.AsyncClient, run_id: str, progress: Callable[[str], None]
    ) -> AnswerAgentRun:
        url = f"{self._api_base_url}{AGENT_RUNS_PATH}/{run_id}"
        while True:
            response = await client.get(url, headers=self._keyed_headers())
            if response.status_code == 401:
                raise RuntimeError("deep_research requires a valid, granted TAKO_API_KEY.")
            response.raise_for_status()
            run = AnswerAgentRun.model_validate(response.json())
            if run.status in TERMINAL_RUN_STATUSES:
                return run
            progress(f"state={run.status}")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
