from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from centaur_tool_websearch import _tako
from centaur_tool_websearch._tako import (
    AnonymousSearchOutput,
    AnswerAgentResult,
    SearchApiResponse,
    TakoBackend,
    normalize_anonymous_output,
    normalize_answer_result,
    normalize_search_response,
)
from centaur_tool_websearch.models import DeepResearchSpec, SearchRequestSpec

V3_RESPONSE = {
    "request_id": "req-v3-1",
    "usage": {"total_cost_usd": 0.007, "compute": {"cost_usd": 0.007}},
    "cards": [
        {
            "card_id": "abc",
            "title": "United States GDP Growth Rate",
            "description": "The real GDP growth rate of United States is 2.1% as of 2025.",
            "semantic_description": "Real GDP growth for the United States over time.",
            "webpage_url": "https://tako.com/card/abc/",
            "image_url": "https://tako.com/api/v1/image/abc/",
            "embed_url": "https://tako.com/embed/abc/",
            "sources": [
                {
                    "source_name": "International Monetary Fund",
                    "source_index": "data",
                    "url": "https://imf.org",
                },
                {"source_name": "World Bank", "source_index": "data"},
            ],
            "methodologies": [
                {
                    "methodology_name": "Where the Data Comes From - IMF",
                    "methodology_description": "World Economic Outlook estimates.",
                }
            ],
            "metric_definitions": [
                {
                    "name": "Real GDP Growth Rate",
                    "definition": "Percentage change in inflation-adjusted GDP.",
                }
            ],
            "data_freshness": {"coverage_end": "2025", "last_updated": "2026-09-03"},
            "exportable": True,
            "nodes": [{"id": "ent::us::1", "type": "entity", "name": "United States"}],
        },
        {"card_id": "nourl", "title": "No page", "description": "x", "webpage_url": None},
    ],
    "web_results": [
        {
            "title": "GDP (Second Estimate) | BEA",
            "url": "https://www.bea.gov/news/2026/gdp-q2",
            "snippet": "Real GDP increased at an annual rate of 1.5 percent … in the second quarter.",
            "source_name": "U.S. Bureau of Economic Analysis",
            "publish_date": "2026-08-28",
        },
        {"title": "No publisher", "url": "https://example.org/a", "snippet": None},
    ],
}

ANONYMOUS_OUTPUT = {
    "cards": [
        {
            "exportable": True,
            "title": "United States GDP Growth Rate",
            "description": "The real gdp growth rate of United States is 2.1% as of 2025.",
            "url": "https://tako.com/card/0qsNya-V-bgimmD4QFQB/",
            "source": "International Monetary Fund",
            "coverage_end": "2025-01-01",
            "last_updated": "2026-09-03",
            "relevance": "High",
        }
    ],
    "web_results": [
        {
            "url": "https://www.bea.gov/news/2026/gdp-q2",
            "title": "GDP (Second Estimate) | BEA",
            "snippet": "Real gross domestic product (GDP) increased at an annual rate of 1.5 percent",
            "source": "U.S. Bureau of Economic Analysis",
        },
        {"url": "https://example.org/b", "title": "B", "snippet": None, "published": "2026-01-02"},
    ],
    "usage": {"total_cost_usd": 0.007, "compute": {"cost_usd": 0.007}},
    "metric_definitions": {
        "Real GDP Growth Rate": "The percentage change in real GDP over a period."
    },
    "source_notes": {
        "International Monetary Fund": "An international organization that provides economic data."
    },
}

ANSWER_RESULT = {
    "answer": "GDP grew 2.1% in 2025 [1]. Q2 2026 came in at 1.5% [2].",
    "cards": [
        {
            "title": "United States GDP Growth Rate",
            "description": "The real GDP growth rate of United States is 2.1% as of 2025.",
            "webpage_url": "https://tako.com/card/abc/",
            "sources": [{"source_name": "International Monetary Fund", "source_index": "data"}],
            "metric_definitions": [{"name": "Real GDP Growth Rate", "definition": "Pct change."}],
            "data_freshness": {"last_updated": "2026-09-03"},
        }
    ],
    "citations": [
        {
            "index": 1,
            "title": "United States GDP Growth Rate",
            "url": None,
            "source_name": "International Monetary Fund",
        },
        {
            "index": 2,
            "title": "GDP (Second Estimate) | BEA",
            "url": "https://www.bea.gov/news/2026/gdp-q2",
            "source_name": "U.S. Bureau of Economic Analysis",
            "excerpt": "1.5 percent",
            "publish_date": "2026-08-28",
        },
        {"index": 3, "title": "Unlinked data source", "url": None},
    ],
    "metadata": {
        "definitions": [
            {"term": "Real GDP", "definition": "Inflation-adjusted output.", "source_ref": 1}
        ],
        "assumptions": [
            {"title": "Calendar years", "description": "Annual figures use calendar years."}
        ],
        "methodology": [
            {"title": "Growth rate", "description": "Year-over-year percentage change."}
        ],
    },
    "usage": None,
    "refusal_code": None,
    "request_id": "rq-1",
}


def test_v3_cards_come_first_and_carry_definitions_and_methodology() -> None:
    docs = normalize_search_response(SearchApiResponse.model_validate(V3_RESPONSE))

    assert [d.source_id for d in docs] == [0, 1, 2]
    card = docs[0]
    assert card.url == "https://tako.com/card/abc/"
    assert card.title == "United States GDP Growth Rate"
    assert card.published_date == "2026-09-03"
    assert card.domain == "International Monetary Fund, World Bank"
    assert card.snippet.splitlines() == [
        "The real GDP growth rate of United States is 2.1% as of 2025.",
        "Real GDP Growth Rate: Percentage change in inflation-adjusted GDP.",
        "Where the Data Comes From - IMF: World Economic Outlook estimates.",
    ]


def test_v3_card_falls_back_to_semantic_description() -> None:
    payload = SearchApiResponse.model_validate(V3_RESPONSE)
    payload.cards[0].description = ""
    docs = normalize_search_response(payload)
    assert docs[0].snippet.startswith("Real GDP growth for the United States over time.")


def test_v3_web_results_map_publisher_and_date() -> None:
    docs = normalize_search_response(SearchApiResponse.model_validate(V3_RESPONSE))
    web = docs[1]
    assert web.domain == "U.S. Bureau of Economic Analysis"
    assert web.published_date == "2026-08-28"
    assert "1.5 percent" in web.snippet
    assert docs[2].domain == "example.org"
    assert docs[2].snippet == ""


def test_anonymous_cards_use_projected_fields_and_merged_definitions() -> None:
    docs = normalize_anonymous_output(AnonymousSearchOutput.model_validate(ANONYMOUS_OUTPUT))

    card = docs[0]
    assert card.url == "https://tako.com/card/0qsNya-V-bgimmD4QFQB/"
    assert card.domain == "International Monetary Fund"
    assert card.published_date == "2026-09-03"
    assert card.snippet.splitlines() == [
        "The real gdp growth rate of United States is 2.1% as of 2025.",
        "Real GDP Growth Rate: The percentage change in real GDP over a period.",
        "International Monetary Fund: An international organization that provides economic data.",
    ]
    assert docs[1].domain == "U.S. Bureau of Economic Analysis"
    assert docs[1].published_date is None
    assert docs[2].published_date == "2026-01-02"
    assert docs[2].domain == "example.org"


def test_answer_result_sources_keep_citation_indexes_and_fill_missing_urls() -> None:
    docs, _report = normalize_answer_result(
        AnswerAgentResult.model_validate(ANSWER_RESULT), max_report_chars=50000
    )

    by_id = {d.source_id: d for d in docs}
    assert by_id[1].url == "https://tako.com/card/abc/"
    assert by_id[1].domain == "International Monetary Fund"
    assert by_id[2].url == "https://www.bea.gov/news/2026/gdp-q2"
    assert by_id[2].snippet == "1.5 percent"
    assert by_id[3].url == _tako.FALLBACK_CITATION_URL
    assert sorted(by_id) == [1, 2, 3]


def test_answer_result_report_sections_in_order() -> None:
    _docs, report = normalize_answer_result(
        AnswerAgentResult.model_validate(ANSWER_RESULT), max_report_chars=50000
    )

    assert report.startswith("GDP grew 2.1% in 2025 [1].")
    order = [
        report.index(h)
        for h in ("## Charts", "## Definitions", "## Assumptions", "## Methodology", "## Sources")
    ]
    assert order == sorted(order)
    assert "- United States GDP Growth Rate: https://tako.com/card/abc/" in report
    assert "- **Real GDP**: Inflation-adjusted output. [1]" in report
    assert "- **Calendar years**: Annual figures use calendar years." in report
    assert "- **Growth rate**: Year-over-year percentage change." in report
    assert "[2] GDP (Second Estimate) | BEA — https://www.bea.gov/news/2026/gdp-q2" in report


def test_answer_result_uncited_card_becomes_next_source() -> None:
    payload = AnswerAgentResult.model_validate(ANSWER_RESULT)
    payload.cards[0].webpage_url = "https://tako.com/card/other/"
    payload.cards[0].title = "Other chart"
    docs, _report = normalize_answer_result(payload, max_report_chars=50000)
    assert docs[-1].source_id == 4
    assert docs[-1].url == "https://tako.com/card/other/"


def test_answer_result_omits_empty_sections_and_protects_sources() -> None:
    payload = AnswerAgentResult.model_validate({**ANSWER_RESULT, "metadata": None, "cards": []})
    _docs, report = normalize_answer_result(payload, max_report_chars=120)
    assert "## Definitions" not in report
    assert "## Charts" not in report
    assert report.endswith("[3] Unlinked data source — https://tako.com")


def _rpc_ok(structured: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {"content": [], "structuredContent": structured},
    }


def _rpc_rate_limited() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "content": [
                {"type": "text", "text": "Anonymous search is limited to 10 calls per minute."}
            ],
            "_meta": {"tako/error": {"kind": "rate_limited"}},
            "isError": True,
        },
    }


class Recorder:
    def __init__(self, rest_status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.rest_status = rest_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "tako.com":
            if self.rest_status != 200:
                return httpx.Response(self.rest_status, json={"error_message": "Invalid API key"})
            return httpx.Response(200, json=V3_RESPONSE)
        if request.url.host == "mcp.tako.com":
            return httpx.Response(200, json=_rpc_ok(ANONYMOUS_OUTPUT))
        raise AssertionError(f"unexpected host {request.url.host}")

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _backend(recorder: Recorder, api_key: str | None = "TAKO_API_KEY") -> TakoBackend:
    return TakoBackend(api_key=api_key, transport=recorder.transport())


def test_keyed_search_hits_v3_with_placeholder_and_attribution_headers() -> None:
    recorder = Recorder()
    result = asyncio.run(
        _backend(recorder).search(
            SearchRequestSpec(
                query="US GDP",
                num_results=7,
                effort="deep",
                include_domains=["bea.gov"],
                max_age_hours=48,
            )
        )
    )

    assert [r.url.host for r in recorder.requests] == ["tako.com"]
    request = recorder.requests[0]
    assert request.url.path == "/api/v3/search"
    assert request.headers["x-api-key"] == "TAKO_API_KEY"
    assert request.headers["x-tako-caller"].startswith('channel=centaur, client_version="')
    assert request.headers["user-agent"].startswith("centaur-websearch/")
    body = json.loads(request.content)
    assert body["query"] == "US GDP"
    assert body["effort"] == "deep"
    assert body["sources"]["data"] == {"count": _tako.DATA_CARD_COUNT}
    assert body["sources"]["web"]["count"] == 7
    assert body["sources"]["web"]["highlights"] is True
    assert body["sources"]["web"]["include_domains"] == ["bea.gov"]
    assert len(body["sources"]["web"]["published_after"]) == 10
    assert result.backend == "tako:api"
    assert result.request_ids == ["req-v3-1"]
    assert result.estimated_cost_usd == 0.007
    assert result.usage == [V3_RESPONSE["usage"]]
    assert [d.source_id for d in result.sources] == [0, 1, 2]


def test_keyed_search_caps_counts_at_twenty_and_prices_deep_without_usage() -> None:
    class NoUsage(Recorder):
        def handler(self, request):
            self.requests.append(request)
            return httpx.Response(200, json={**V3_RESPONSE, "usage": None})

    recorder = NoUsage()
    result = asyncio.run(
        _backend(recorder).search(SearchRequestSpec(query="q", num_results=40, effort="deep"))
    )
    body = json.loads(recorder.requests[0].content)
    assert body["sources"]["data"]["count"] == _tako.DATA_CARD_COUNT
    assert body["sources"]["web"]["count"] == 20
    assert result.estimated_cost_usd == 0.012


def test_web_results_survive_the_client_cap_at_the_default_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "cards": [
                    {"title": f"c{i}", "webpage_url": f"https://tako.com/{i}", "description": "d"}
                    for i in range(body["sources"]["data"]["count"])
                ],
                "web_results": [
                    {"title": f"w{i}", "url": f"https://w.example/{i}", "snippet": "s"}
                    for i in range(body["sources"]["web"]["count"])
                ],
                "request_id": "r1",
            },
        )

    backend = TakoBackend(api_key="k", transport=httpx.MockTransport(handler))
    result = asyncio.run(backend.search(SearchRequestSpec(query="q", num_results=10)))

    hosts = [httpx.URL(d.url).host for d in result.sources[:10]]
    assert hosts.count("tako.com") == _tako.DATA_CARD_COUNT
    assert hosts.count("w.example") == 10 - _tako.DATA_CARD_COUNT


def test_keyed_search_notes_max_chars_total() -> None:
    recorder = Recorder()
    result = asyncio.run(
        _backend(recorder).search(SearchRequestSpec(query="q", max_chars_total=5000))
    )
    assert any("max_chars_total" in f["error"] for f in result.partial_failures)


def test_keyed_search_notes_the_parallel_only_knobs_it_drops() -> None:
    recorder = Recorder()
    result = asyncio.run(
        _backend(recorder).search(
            SearchRequestSpec(query="q", client_model="claude-opus-4-7", session_id="s-1")
        )
    )
    notes = " ".join(f["error"] for f in result.partial_failures)
    assert "client_model" in notes
    assert "session_id" in notes


def test_403_falls_back_to_anonymous_like_a_401() -> None:
    recorder = Recorder(rest_status=403)

    result = asyncio.run(_backend(recorder).search(SearchRequestSpec(query="q")))

    assert result.backend == "tako:anonymous"
    assert any("did not authenticate" in f["error"] for f in result.partial_failures)


def test_the_auth_note_repeats_on_every_later_search() -> None:
    recorder = Recorder(rest_status=401)
    backend = _backend(recorder)

    asyncio.run(backend.search(SearchRequestSpec(query="first")))
    second = asyncio.run(backend.search(SearchRequestSpec(query="second")))

    assert any("did not authenticate" in f["error"] for f in second.partial_failures)


def test_duplicate_citation_indexes_collapse_to_one_source() -> None:
    payload = {
        "answer": "a [1]",
        "cards": [],
        "citations": [
            {"index": 1, "title": "First", "url": "https://a.example"},
            {"index": 1, "title": "Second", "url": "https://b.example"},
        ],
    }
    sources, _ = normalize_answer_result(
        AnswerAgentResult.model_validate(payload), max_report_chars=5000
    )

    assert [d.source_id for d in sources] == [1]
    assert sources[0].title == "First"


def test_an_explicit_zero_timeout_is_not_coerced_to_the_default() -> None:
    stages: list[str] = []
    recorder = AgentRecorder([_sse(FULL_STREAM[0])])

    with pytest.raises(RuntimeError):
        asyncio.run(
            recorder.backend().deep_research(
                DeepResearchSpec(question="Why?", timeout_seconds=0), stages.append
            )
        )

    assert any("timeout=0s" in s for s in stages)


def test_401_falls_back_to_anonymous_with_no_auth_header() -> None:
    recorder = Recorder(rest_status=401)
    backend = _backend(recorder)

    result = asyncio.run(
        backend.search(SearchRequestSpec(query="US GDP", include_domains=["x.com"], num_results=3))
    )

    assert [r.url.host for r in recorder.requests] == ["tako.com", "mcp.tako.com"]
    anonymous = recorder.requests[1]
    assert "authorization" not in anonymous.headers
    assert "x-api-key" not in anonymous.headers
    assert anonymous.headers["user-agent"] == f"centaur-websearch/{_tako.TOOL_VERSION}"
    assert anonymous.headers["accept"] == "application/json, text/event-stream"
    envelope = json.loads(anonymous.content)
    assert envelope["method"] == "tools/call"
    assert envelope["params"] == {
        "name": "tako_search",
        "arguments": {"query": "US GDP", "sources": ["data", "web"]},
    }
    assert result.backend == "tako:anonymous"
    assert result.estimated_cost_usd == 0.0
    assert result.request_ids == []
    assert result.attribution is None
    errors = " ".join(f["error"] for f in result.partial_failures)
    assert "did not authenticate" in errors
    assert "include_domains" in errors
    assert "num_results=3" in errors
    assert backend.search_mode == "anonymous"


def test_second_search_after_401_skips_rest() -> None:
    recorder = Recorder(rest_status=401)
    backend = _backend(recorder)
    asyncio.run(backend.search(SearchRequestSpec(query="a")))
    asyncio.run(backend.search(SearchRequestSpec(query="b")))
    assert [r.url.host for r in recorder.requests] == ["tako.com", "mcp.tako.com", "mcp.tako.com"]


def test_non_auth_rest_error_raises_instead_of_falling_back() -> None:
    recorder = Recorder(rest_status=500)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_backend(recorder).search(SearchRequestSpec(query="a")))
    assert [r.url.host for r in recorder.requests] == ["tako.com"]


def test_anonymous_rate_limit_is_an_error_with_server_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_rpc_rate_limited())

    backend = TakoBackend(api_key=None, transport=httpx.MockTransport(handler))
    with pytest.raises(
        RuntimeError, match="rate limited: Anonymous search is limited to 10 calls per minute"
    ):
        asyncio.run(backend.search(SearchRequestSpec(query="a")))


def test_anonymous_429_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": "slow down", "data": {"kind": "rate_limited"}},
            },
        )

    backend = TakoBackend(api_key=None, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="rate limited: slow down"):
        asyncio.run(backend.search(SearchRequestSpec(query="a")))


def test_anonymous_429_with_an_html_body_reports_the_page_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"content-type": "text/html"},
            text="<html><body>429 Too Many Requests</body></html>",
        )

    backend = TakoBackend(api_key=None, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="rate limited: .*429 Too Many Requests"):
        asyncio.run(backend.search(SearchRequestSpec(query="a")))


def test_anonymous_sse_reply_is_decoded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.dumps(_rpc_ok(ANONYMOUS_OUTPUT))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {payload}\n\n".encode(),
        )

    backend = TakoBackend(api_key=None, transport=httpx.MockTransport(handler))
    result = asyncio.run(backend.search(SearchRequestSpec(query="a")))
    assert result.backend == "tako:anonymous"
    assert len(result.sources) == 3


def _frame(seq: int, block: dict) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "seq": seq,
                "run_id": "run-1",
                "thread_id": "thr-1",
                "category": "activity",
                "block": block,
            }
        )
        + "\n\n"
    )


def _sse(*frames: str) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content="".join(frames).encode()
    )


FULL_STREAM = (
    _frame(0, {"kind": "status", "message": "planning"}),
    _frame(
        1,
        {
            "kind": "tool_call",
            "id": "t1",
            "tool": "search_graph",
            "status_message": "looking up GDP",
        },
    ),
    _frame(
        2, {"kind": "subagent", "agent_id": "a1", "subagent_type": "retriever", "event": "dispatch"}
    ),
    _frame(3, {"kind": "heartbeat"}),
    _frame(4, {"kind": "agent_result", "id": "r1", "data": ANSWER_RESULT}),
    _frame(
        5,
        {
            "kind": "run_summary",
            "status": "completed",
            "created_at": "x",
            "usage": {"total_cost_usd": 0.49},
        },
    ),
    _frame(5, {"kind": "stream_done"}),
)


class AgentRecorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self.responses = list(responses)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def backend(self, api_key: str | None = "TAKO_API_KEY") -> TakoBackend:
        return TakoBackend(api_key=api_key, transport=httpx.MockTransport(self.handler))


def test_deep_research_streams_progress_and_returns_result() -> None:
    recorder = AgentRecorder([_sse(*FULL_STREAM)])
    stages: list[str] = []

    result = asyncio.run(
        recorder.backend().deep_research(DeepResearchSpec(question="Why?"), stages.append)
    )

    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/api/v1/agent/answer/runs"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["x-api-key"] == "TAKO_API_KEY"
    assert request.headers["x-tako-caller"] == _tako.CALLER_VALUE
    assert json.loads(request.content) == {"query": "Why?", "effort": "medium"}
    assert result.backend == "tako:agent"
    assert result.request_ids == ["run-1"]
    assert result.estimated_cost_usd == 0.49
    assert result.usage == [{"total_cost_usd": 0.49}]
    assert result.answer_markdown.startswith("GDP grew 2.1%")
    assert [d.source_id for d in result.sources] == [1, 2, 3]
    assert any("planning" in s for s in stages)
    assert any("search_graph" in s for s in stages)
    assert any("retriever" in s for s in stages)


def test_deep_research_effort_high_and_processor_note() -> None:
    recorder = AgentRecorder([_sse(*FULL_STREAM)])
    result = asyncio.run(
        recorder.backend().deep_research(
            DeepResearchSpec(question="Why?", effort="high", processor="pro-fast"), lambda _s: None
        )
    )
    assert json.loads(recorder.requests[0].content)["effort"] == "high"
    assert any(
        "--processor" in f["error"] and "pro-fast" in f["error"] for f in result.partial_failures
    )


def test_deep_research_resumes_once_after_a_drop() -> None:
    dropped = _sse(FULL_STREAM[0], FULL_STREAM[1])
    resumed = _sse(*FULL_STREAM[2:])
    recorder = AgentRecorder([dropped, resumed])

    result = asyncio.run(
        recorder.backend().deep_research(DeepResearchSpec(question="Why?"), lambda _s: None)
    )

    assert [r.method for r in recorder.requests] == ["POST", "GET"]
    resume = recorder.requests[1]
    assert resume.url.path == "/api/v1/agent/answer/runs/run-1"
    assert resume.url.params["starting_after"] == "1"
    assert result.request_ids == ["run-1"]


def test_non_sse_interstitial_on_resume_falls_through_to_polling() -> None:
    dropped = _sse(FULL_STREAM[0], FULL_STREAM[1])
    interstitial = httpx.Response(200, headers={"content-type": "text/html"}, text="<html>go away")
    completed = httpx.Response(
        200,
        json={
            "run_id": "run-1",
            "status": "completed",
            "created_at": "x",
            "result": ANSWER_RESULT,
            "usage": {"total_cost_usd": 0.11},
        },
    )
    recorder = AgentRecorder([dropped, interstitial, completed])
    stages: list[str] = []

    result = asyncio.run(
        recorder.backend().deep_research(DeepResearchSpec(question="Why?"), stages.append)
    )

    assert [r.method for r in recorder.requests] == ["POST", "GET", "GET"]
    assert result.answer_markdown.startswith("GDP grew 2.1%")
    assert any("text/event-stream" in s for s in stages)


def test_deep_research_polls_when_resume_has_no_result(monkeypatch: pytest.MonkeyPatch) -> None:
    dropped = _sse(FULL_STREAM[0])
    resumed_empty = _sse(_frame(1, {"kind": "status", "message": "still going"}))
    running = httpx.Response(200, json={"run_id": "run-1", "status": "running", "created_at": "x"})
    completed = httpx.Response(
        200,
        json={
            "run_id": "run-1",
            "status": "completed",
            "created_at": "x",
            "result": ANSWER_RESULT,
            "usage": {"total_cost_usd": 0.5},
        },
    )
    recorder = AgentRecorder([dropped, resumed_empty, running, completed])
    monkeypatch.setattr(_tako, "POLL_INTERVAL_SECONDS", 0.0)

    result = asyncio.run(
        recorder.backend().deep_research(DeepResearchSpec(question="Why?"), lambda _s: None)
    )

    assert [r.method for r in recorder.requests] == ["POST", "GET", "GET", "GET"]
    assert recorder.requests[2].headers["accept"] == "application/json"
    assert result.estimated_cost_usd == 0.5


def test_deep_research_401_names_the_missing_grant() -> None:
    recorder = AgentRecorder([httpx.Response(401, json={"error_message": "Invalid API key"})])
    with pytest.raises(RuntimeError, match="requires a valid, granted TAKO_API_KEY"):
        asyncio.run(
            recorder.backend().deep_research(DeepResearchSpec(question="Why?"), lambda _s: None)
        )


def test_deep_research_without_key_is_an_error() -> None:
    recorder = AgentRecorder([])
    with pytest.raises(RuntimeError, match="requires TAKO_API_KEY"):
        asyncio.run(
            recorder.backend(api_key=None).deep_research(
                DeepResearchSpec(question="Why?"), lambda _s: None
            )
        )


def test_deep_research_refusal_is_an_error() -> None:
    refused = {**ANSWER_RESULT, "answer": None, "refusal_code": "rejected_input_classifier"}
    recorder = AgentRecorder(
        [
            _sse(
                _frame(0, {"kind": "agent_result", "id": "r", "data": refused}),
                _frame(1, {"kind": "stream_done"}),
            )
        ]
    )
    with pytest.raises(RuntimeError, match="refusal_code=rejected_input_classifier"):
        asyncio.run(
            recorder.backend().deep_research(DeepResearchSpec(question="Why?"), lambda _s: None)
        )


def test_deep_research_failed_run_is_an_error() -> None:
    frames = (
        _frame(
            0,
            {
                "kind": "run_summary",
                "status": "failed",
                "created_at": "x",
                "error": {"code": "boom", "message": "agent crashed"},
            },
        ),
        _frame(0, {"kind": "stream_done"}),
    )
    recorder = AgentRecorder([_sse(*frames)])
    with pytest.raises(RuntimeError, match="agent crashed"):
        asyncio.run(
            recorder.backend().deep_research(DeepResearchSpec(question="Why?"), lambda _s: None)
        )


def test_deep_research_timeout_message_mentions_no_cancel() -> None:
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return _sse(*FULL_STREAM)

    backend = TakoBackend(api_key="TAKO_API_KEY", transport=httpx.MockTransport(slow_handler))
    with pytest.raises(RuntimeError, match="did not finish within 0s"):
        asyncio.run(
            backend.deep_research(
                DeepResearchSpec(question="Why?", timeout_seconds=0.01), lambda _s: None
            )
        )


TOOL_RETRY_FRAME = _frame(
    2,
    {
        "kind": "tool_retry",
        "id": "t2",
        "tool": "execute_dataframe_code",
        "parent_id": None,
        "elapsed_ms": 1200,
        "error": "Your output DataFrame is missing the year column.",
    },
)
IGNORED_KIND_FRAMES = (
    _frame(1, {"kind": "reasoning", "id": "rs1", "delta": "thinking", "done": False}),
    _frame(
        1,
        {
            "kind": "tool_result",
            "id": "t1",
            "tool": "search",
            "parent_id": None,
            "elapsed_ms": 900,
            "link": None,
        },
    ),
    _frame(1, {"kind": "text", "id": "x1", "delta": "GDP grew", "done": False}),
)


def test_tool_retry_carries_a_string_error_and_does_not_break_the_stream() -> None:
    frames = (FULL_STREAM[0], TOOL_RETRY_FRAME, *IGNORED_KIND_FRAMES, *FULL_STREAM[4:])
    recorder = AgentRecorder([_sse(*frames)])
    stages: list[str] = []

    result = asyncio.run(
        recorder.backend().deep_research(DeepResearchSpec(question="Why?"), stages.append)
    )

    assert [r.method for r in recorder.requests] == ["POST"]
    assert result.backend == "tako:agent"
    assert any("retrying execute_dataframe_code" in s for s in stages)
    assert any("missing the year column" in s for s in stages)


def test_run_summary_string_error_still_names_the_failure() -> None:
    frames = (
        _frame(
            0,
            {
                "kind": "run_summary",
                "status": "failed",
                "created_at": "x",
                "error": "agent crashed hard",
            },
        ),
        _frame(1, {"kind": "stream_done"}),
    )
    recorder = AgentRecorder([_sse(*frames)])
    with pytest.raises(RuntimeError, match="agent crashed hard"):
        asyncio.run(
            recorder.backend().deep_research(DeepResearchSpec(question="Why?"), lambda _s: None)
        )


def test_unreadable_frame_is_skipped_not_fatal() -> None:
    garbled = 'data: {"seq": 1, "run_id": "run-1", "block": {"kind": 42}}\n\n'
    frames = (FULL_STREAM[0], garbled, *FULL_STREAM[4:])
    recorder = AgentRecorder([_sse(*frames)])
    stages: list[str] = []

    result = asyncio.run(
        recorder.backend().deep_research(DeepResearchSpec(question="Why?"), stages.append)
    )

    assert result.answer_markdown.startswith("GDP grew 2.1%")
    assert any("unreadable stream frame" in s for s in stages)


def test_lost_agent_result_frame_falls_through_to_polling() -> None:
    garbled_result = (
        'data: {"seq": 4, "run_id": "run-1", "block": {"kind": "agent_result", "data": 7}}\n\n'
    )
    stream = _sse(FULL_STREAM[0], garbled_result, FULL_STREAM[5], FULL_STREAM[6])
    completed = httpx.Response(
        200,
        json={
            "run_id": "run-1",
            "status": "completed",
            "created_at": "x",
            "result": ANSWER_RESULT,
            "usage": {"total_cost_usd": 0.11},
        },
    )
    recorder = AgentRecorder([stream, completed])

    result = asyncio.run(
        recorder.backend().deep_research(DeepResearchSpec(question="Why?"), lambda _s: None)
    )

    assert [r.method for r in recorder.requests] == ["POST", "GET"]
    assert result.estimated_cost_usd == 0.11
    assert result.answer_markdown.startswith("GDP grew 2.1%")


def test_undecodable_agent_result_payload_falls_through_to_polling() -> None:
    drifted_result = (
        'data: {"seq": 4, "run_id": "run-1", "block": {"kind": "agent_result", '
        '"data": {"answer": "a", "cards": [], "citations": [{"index": 1}]}}}\n\n'
    )
    stream = _sse(FULL_STREAM[0], drifted_result, FULL_STREAM[5], FULL_STREAM[6])
    completed = httpx.Response(
        200,
        json={
            "run_id": "run-1",
            "status": "completed",
            "created_at": "x",
            "result": ANSWER_RESULT,
            "usage": {"total_cost_usd": 0.11},
        },
    )
    recorder = AgentRecorder([stream, completed])
    stages: list[str] = []

    result = asyncio.run(
        recorder.backend().deep_research(DeepResearchSpec(question="Why?"), stages.append)
    )

    assert [r.method for r in recorder.requests] == ["POST", "GET"]
    assert result.answer_markdown.startswith("GDP grew 2.1%")
    assert any("unreadable agent_result" in s for s in stages)


def test_sparse_citation_indexes_survive_and_uncited_card_follows_the_highest() -> None:
    payload = AnswerAgentResult.model_validate(
        {
            **ANSWER_RESULT,
            "citations": [
                {"index": 1, "title": "St. Louis Fed", "url": "https://www.stlouisfed.org/"},
                {
                    "index": 15,
                    "title": "FRED series",
                    "url": "https://fred.stlouisfed.org/series/A191RL1A225NBEA",
                },
                {
                    "index": 22,
                    "title": "Eurostat",
                    "url": "https://ec.europa.eu/eurostat/web/products-euro-indicators",
                },
            ],
            "cards": [
                {
                    "title": "Real GDP Growth Comparison",
                    "webpage_url": "https://tako.com/card/DjDkgFurektE8dRo4JiI/",
                    "data_freshness": {"last_updated": "2026-09-03"},
                }
            ],
        }
    )

    docs, report = normalize_answer_result(payload, max_report_chars=50000)

    assert [d.source_id for d in docs] == [1, 15, 22, 23]
    assert docs[0].domain == "www.stlouisfed.org"
    assert docs[3].url == "https://tako.com/card/DjDkgFurektE8dRo4JiI/"
    assert "[15] FRED series — https://fred.stlouisfed.org/series/A191RL1A225NBEA" in report
