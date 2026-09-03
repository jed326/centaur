"""Constants and text helpers shared by every websearch backend."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

from .models import SourceDocument

TOOL_NAME = "centaur-websearch"
DISTRIBUTION_NAME = "websearch"
FALLBACK_VERSION = "0.3.0"


def _installed_version() -> str:
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return FALLBACK_VERSION


TOOL_VERSION = _installed_version()
USER_AGENT = f"{TOOL_NAME}/{TOOL_VERSION}"


def append_within_budget(body: str, trailer: str, max_chars: int) -> str:
    """Append `trailer` to `body` within `max_chars`, truncating only the body.

    The trailer carries citation integrity (a `## Sources` block or an
    attribution footer), so it is never sliced. If the trailer alone exceeds
    the budget, the cap is exceeded rather than the citation map corrupted.
    """
    body = body.rstrip()
    if len(body) + len(trailer) <= max_chars:
        return body + trailer
    body_budget = max(0, max_chars - len(trailer))
    return body[:body_budget].rstrip() + trailer


def render_sources_block(sources: list[SourceDocument]) -> str:
    if not sources:
        return ""
    lines = [f"[{source.source_id}] {source.title} — {source.url}" for source in sources]
    return "\n\n## Sources\n" + "\n".join(lines)


def decode_jsonrpc_response(response: httpx.Response) -> dict[str, Any]:
    """Decode a JSON-RPC reply that arrived as JSON or as an SSE body.

    For SSE, the last event whose `data:` lines parse as a JSON object wins.
    """
    content_type = response.headers.get("content-type", "")
    text = response.text
    if "text/event-stream" not in content_type:
        return json.loads(text) if text.strip() else {}
    latest: dict[str, Any] | None = None
    for event_block in text.split("\n\n"):
        data_lines = [
            line[len("data:") :].lstrip(" ")
            for line in event_block.splitlines()
            if line.startswith("data:")
        ]
        payload_text = "\n".join(data_lines).strip()
        if not payload_text:
            continue
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            latest = parsed
    if latest is None:
        raise RuntimeError("JSON-RPC endpoint returned an empty SSE stream.")
    return latest
