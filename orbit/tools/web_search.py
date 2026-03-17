"""Web search — ddgs/duckduckgo-search (real results) with DuckDuckGo API + Wikipedia fallback."""

from __future__ import annotations

import httpx
import warnings

from orbit.tools.base import Tool, ToolResult


def _search_ddgs(query: str, max_results: int) -> list[str]:
    """Use ddgs or duckduckgo-search for real web results. Returns list of formatted strings."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

        results = []
        ddgs = DDGS()
        for r in ddgs.text(query, max_results=max_results, region="us-en"):
            title = r.get("title") or ""
            body = r.get("body") or ""
            href = r.get("href") or ""
            if title or body:
                line = f"- {title}: {body}" if body else f"- {title}"
                if href:
                    line += f" ({href})"
                results.append(line)
        return results
    except ImportError:
        return []
    except Exception:
        return []


def _search_ddgo_api(query: str, max_results: int) -> list[str]:
    """DuckDuckGo Instant Answer API — only returns Wikipedia-style abstracts, often empty."""
    results = []
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json"},
            timeout=10,
        )
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict):
            return []
        if data.get("Abstract"):
            results.append(f"- {data.get('Abstract', '')} (Source: {data.get('AbstractURL', '')})")
        if data.get("Answer"):
            results.append(f"- {data.get('Answer', '')}")
        for r in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(r, dict) and r.get("Text"):
                results.append(f"- {r.get('Text', '')}")
            elif isinstance(r, dict) and r.get("FirstURL"):
                results.append(f"- {r.get('Text', r.get('FirstURL', ''))}")
        return results[:max_results]
    except Exception:
        return []


def _search_wikipedia(query: str, max_results: int) -> list[str]:
    """Wikipedia OpenSearch fallback."""
    try:
        wiki = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": max_results, "format": "json"},
            timeout=10,
        )
        wiki_data = wiki.json()
        results = []
        if len(wiki_data) >= 2 and wiki_data[1]:
            for i, title in enumerate(wiki_data[1]):
                desc = wiki_data[2][i] if len(wiki_data) > 2 and i < len(wiki_data[2]) else ""
                url = wiki_data[3][i] if len(wiki_data) > 3 and i < len(wiki_data[3]) else ""
                results.append(f"- {title}: {desc} {url}")
        return results
    except Exception:
        return []


class WebSearchTool(Tool):
    """Search the web. Returns snippets and URLs."""

    name = "web_search"
    description = "Search the internet. Returns top results with titles and snippets."

    def run(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            # 1. Try duckduckgo-search (real web results) — best for any query
            results = _search_ddgs(query, max_results)
            # 2. Fallback: DuckDuckGo API (instant answers, often empty for general queries)
            if not results:
                results = _search_ddgo_api(query, max_results)
            # 3. Fallback: Wikipedia
            if not results:
                results = _search_wikipedia(query, max_results)
            if not results:
                results = [f"No results for: {query}"]
            return ToolResult(success=True, output="\n".join(results))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
