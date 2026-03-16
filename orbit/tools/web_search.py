"""Web search via DuckDuckGo + Wikipedia fallback."""

from __future__ import annotations

import httpx

from orbit.tools.base import Tool, ToolResult


def _extract_ddgo(results: list) -> list[str]:
    out = []
    for r in results:
        if isinstance(r, dict):
            if "Text" in r:
                out.append(f"- {r.get('Text', '')}")
            elif "FirstURL" in r:
                out.append(f"- {r.get('Text', r.get('FirstURL', ''))}")
        elif isinstance(r, dict) and "Topics" in r:
            out.extend(_extract_ddgo(r.get("Topics", [])))
    return out


class WebSearchTool(Tool):
    """Search the web. Returns snippets and URLs."""

    name = "web_search"
    description = "Search the internet. Returns top results with titles and snippets."

    def run(self, query: str, max_results: int = 5) -> ToolResult:
        results = []
        try:
            resp = httpx.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json"},
                timeout=10,
            )
            try:
                data = resp.json() if resp.content else {}
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            results = []
            for r in data.get("RelatedTopics", [])[: max_results * 2]:
                if isinstance(r, dict) and "Text" in r:
                    results.append(f"- {r.get('Text', '')}")
                elif isinstance(r, dict) and "FirstURL" in r:
                    results.append(f"- {r.get('Text', r.get('FirstURL', ''))}")
                elif isinstance(r, dict) and "Topics" in r:
                    results.extend(_extract_ddgo(r.get("Topics", []))[:max_results])
            if data.get("Abstract"):
                results.insert(0, f"- {data.get('Abstract', '')} (Source: {data.get('AbstractURL', '')})")
            if data.get("Answer"):
                results.insert(0, f"- {data.get('Answer', '')}")
            results = results[:max_results]
            if not results:
                # Fallback: Wikipedia API
                try:
                    wiki = httpx.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={"action": "opensearch", "search": query, "limit": max_results, "format": "json"},
                        timeout=10,
                    )
                    wiki_data = wiki.json()
                    if len(wiki_data) >= 2 and wiki_data[1]:
                        for i, title in enumerate(wiki_data[1]):
                            desc = wiki_data[2][i] if len(wiki_data) > 2 and i < len(wiki_data[2]) else ""
                            url = wiki_data[3][i] if len(wiki_data) > 3 and i < len(wiki_data[3]) else ""
                            results.append(f"- {title}: {desc} {url}")
                except Exception:
                    pass
            if not results:
                results = [f"No results for: {query}"]
            return ToolResult(success=True, output="\n".join(results))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
