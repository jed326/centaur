from __future__ import annotations

import tomllib
from pathlib import Path

import httpx
import pytest
from centaur_tool_websearch import _common
from centaur_tool_websearch.models import (
    DeepResearchResult,
    DeepResearchSpec,
    RetrievalResult,
    SearchRequestSpec,
    SourceDocument,
)


def test_tool_version_matches_pyproject() -> None:
    manifest = tomllib.loads(Path(__file__).with_name("pyproject.toml").read_text())
    assert manifest["project"]["version"] == _common.FALLBACK_VERSION
    assert f"centaur-websearch/{_common.TOOL_VERSION}" == _common.USER_AGENT


def test_search_request_spec_defaults() -> None:
    spec = SearchRequestSpec(query="q")
    assert spec.num_results == 10
    assert spec.timeout_seconds == 60.0
    assert spec.effort is None
    assert spec.include_domains is None


def test_search_request_spec_rejects_unknown_effort() -> None:
    with pytest.raises(ValueError):
        SearchRequestSpec(query="q", effort="ultra")


def test_retrieval_result_defaults_are_independent() -> None:
    first = RetrievalResult(sources=[], backend="x")
    second = RetrievalResult(sources=[], backend="y")
    first.partial_failures.append({"query": "q", "error": "e"})
    assert second.partial_failures == []
    assert first.attribution is None
    assert first.estimated_cost_usd is None


def test_deep_research_spec_defaults() -> None:
    spec = DeepResearchSpec(question="why")
    assert spec.effort is None
    assert spec.processor is None
    assert spec.timeout_seconds is None
    assert spec.max_report_chars == 50000


def test_deep_research_result_carries_usage() -> None:
    result = DeepResearchResult(sources=[], answer_markdown="a", backend="tako:agent")
    assert result.usage == []
    assert result.request_ids == []


def test_append_within_budget_protects_trailer() -> None:
    trailer = "\n\n## Sources\n[1] t — u"
    out = _common.append_within_budget("x" * 100, trailer, 40)
    assert out == "x" * (40 - len(trailer)) + trailer
    assert len(out) == 40
    assert _common.append_within_budget("short", trailer, 400) == "short" + trailer


def test_render_sources_block() -> None:
    docs = [SourceDocument(source_id=1, title="T", url="https://a.example")]
    assert _common.render_sources_block(docs) == "\n\n## Sources\n[1] T — https://a.example"
    assert _common.render_sources_block([]) == ""


def test_decode_jsonrpc_response_handles_json_and_sse() -> None:
    plain = httpx.Response(200, json={"jsonrpc": "2.0", "result": {"a": 1}})
    assert _common.decode_jsonrpc_response(plain) == {"jsonrpc": "2.0", "result": {"a": 1}}
    sse = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=b'event: message\ndata: {"jsonrpc": "2.0", "result": {"b": 2}}\n\n',
    )
    assert _common.decode_jsonrpc_response(sse) == {"jsonrpc": "2.0", "result": {"b": 2}}
