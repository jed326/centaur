from __future__ import annotations

from typing import get_args

import pytest
from centaur_tool_websearch import _parallel, _tako
from centaur_tool_websearch import client as client_module
from centaur_tool_websearch.models import (
    DeepResearchResult,
    DeepResearchSpec,
    ResearchEffort,
    RetrievalResult,
    SearchEffort,
    SearchRequestSpec,
)


def test_every_effort_has_a_price_and_a_vendor_mapping() -> None:
    search_efforts = set(get_args(SearchEffort))
    research_efforts = set(get_args(ResearchEffort))

    assert set(client_module.SEARCH_EFFORTS) == search_efforts
    assert set(client_module.RESEARCH_EFFORTS) == research_efforts
    assert search_efforts <= set(_tako.SEARCH_PRICE_USD)
    assert search_efforts <= set(_parallel.EFFORT_TO_SEARCH_MODE)
    assert research_efforts <= set(_parallel.EFFORT_TO_PROCESSOR)


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
