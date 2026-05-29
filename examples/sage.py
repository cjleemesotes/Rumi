"""Sage — an Dervish proficient in the web.

Wraps `botasaurus` for anti-bot-detection scraping. The Sage visits pages,
extracts text, clicks buttons, types into fields, follows links, and searches
Google — without being flagged as a bot by Cloudflare, Datadome, or similar.

`botasaurus` is an OPTIONAL dependency. Without it installed the Sage class
still imports, and its tools raise an informative ImportError on first call
telling you what to install. With `pip install botasaurus`, the tools work.

Default tools registered for agent_type="Sage":

| Tool                | What it does                              |
|---------------------|-------------------------------------------|
| `visit_page`        | Navigate to a URL; return title + lead.    |
| `extract_text`      | Get text content matching a CSS selector. |
| `extract_all_text`  | Return visible text of the whole page.    |
| `extract_links`     | Return all `<a>` anchors as {text, href}. |
| `fill_field`        | Type text into an input by selector.       |
| `click_element`     | Click an element by selector.             |
| `search_google`     | Search and return the top-N result URLs.  |

The Sage is patient and observant. It moves through the web as a guest,
never a thief. It leaves no trace.
"""
from __future__ import annotations

from typing import Any

from rumi import Dervish, HeartConfig, Voice, tool


# ---- The Heart speaks ----

_SAGE_INSTRUCTIONS = """
You are a Sage — patient, observant, leaves no trace. You travel through
the web as a guest, not a thief.

How you work:
- For an unknown entity, search_google first to find the right page.
- For a known URL, visit_page — it returns title + lead text.
- For specific text, extract_text with a CSS selector you can name.
- Use extract_all_text only when no selector fits.

When you answer, you cite the URL you read it on. When the page said it
in a memorable line, you quote that line directly instead of paraphrasing.

You never invent facts. If the page didn't say it, you say so.

Available tools: visit_page, extract_text, extract_all_text, extract_links,
fill_field, click_element, search_google.
""".strip()


class Sage(Dervish):
    """A preset Dervish for web exploration via botasaurus.

    `pip install botasaurus` for the optional botasaurus dependency.
    """

    heart_config = HeartConfig(voices=[Voice(
        model_name="~openai/gpt-latest",
        instructions=_SAGE_INSTRUCTIONS,
        context_window=24,
    )])


# ---- Helpers ----

def _require_botasaurus():
    """Raise an informative ImportError if botasaurus isn't available."""
    try:
        from botasaurus.browser import browser, Driver  # noqa: F401
        return True
    except ImportError as exc:
        raise ImportError(
            "Sage's web tools need botasaurus. Install it with "
            "`pip install botasaurus`. "
            f"(Underlying error: {exc})"
        ) from exc


def _run_in_browser(fn):
    """Wrap a function-of-driver in a botasaurus @browser(cache=False) call.

    Used so each tool starts a fresh-but-pooled headless Chrome, runs `fn`,
    and returns the result.
    """
    _require_botasaurus()
    from botasaurus.browser import browser

    @browser(cache=False, headless=True)
    def _job(driver, data):
        return fn(driver, data)

    return _job


def _run_scalar(fn, data: dict) -> Any:
    """Run a driver job whose result is a scalar/dict, unwrapping botasaurus's
    single-element batch list (`[value]` → `value`)."""
    result = _run_in_browser(fn)(data)
    if isinstance(result, list) and result:
        result = result[0]
    return result


def _run_list(fn, data: dict) -> list:
    """Run a driver job whose result is a list. botasaurus wraps it as `[[...]]`;
    unwrap one layer — but only when the inner element is itself a list, so an
    already-unwrapped list isn't mistaken for the batch. Defaults to `[]`."""
    result = _run_in_browser(fn)(data)
    if isinstance(result, list) and result and isinstance(result[0], list):
        result = result[0]
    return result if isinstance(result, list) else []


# ---- Tools ----

@tool(["Sage"])
def visit_page(url: str) -> str:
    """
    Navigate to a URL and return a short summary: title + first 500 chars of
    visible body text.
    ---
    url: an http(s) URL to visit
    """
    def _job(driver, data):
        driver.get(data["url"])
        title = driver.title or ""
        body = driver.get_text("body") or ""
        return {"title": title, "lead": body[:500]}

    result = _run_scalar(_job, {"url": url})
    return f"{result.get('title', '')} — {result.get('lead', '')}".strip(" —")


@tool(["Sage"])
def extract_text(url: str, css_selector: str) -> str:
    """
    Visit a URL and return the visible text of the first element matching the
    given CSS selector.
    ---
    url: an http(s) URL to visit
    css_selector: a CSS selector (e.g. "h1", ".article-body", "#main p")
    """
    def _job(driver, data):
        driver.get(data["url"])
        return driver.get_text(data["css_selector"]) or ""

    result = _run_scalar(_job, {"url": url, "css_selector": css_selector})
    return str(result) if result is not None else ""


@tool(["Sage"])
def extract_all_text(url: str) -> str:
    """
    Visit a URL and return the visible text of the whole page (body).
    Use this when you want everything readable, no CSS selector needed.
    ---
    url: an http(s) URL to visit
    """
    def _job(driver, data):
        driver.get(data["url"])
        return driver.get_text("body") or ""

    result = _run_scalar(_job, {"url": url})
    return str(result) if result is not None else ""


@tool(["Sage"])
def extract_links(url: str) -> list[dict]:
    """
    Visit a URL and return all anchor tags as `[{text, href}, ...]`.
    ---
    url: an http(s) URL to visit
    """
    def _job(driver, data):
        driver.get(data["url"])
        anchors = driver.select_all("a") or []
        out = []
        for a in anchors:
            try:
                href = a.get_attribute("href") or ""
                text = (a.text or "").strip()
                if href:
                    out.append({"text": text, "href": href})
            except Exception:
                continue
        return out

    return _run_list(_job, {"url": url})


@tool(["Sage"])
def fill_field(url: str, css_selector: str, text: str) -> str:
    """
    Visit a URL and type `text` into the element matching `css_selector`.
    Returns "ok" on success.
    ---
    url: an http(s) URL to visit
    css_selector: a CSS selector for the input/textarea
    text: text to type
    """
    def _job(driver, data):
        driver.get(data["url"])
        driver.type(data["css_selector"], data["text"])
        return "ok"

    result = _run_scalar(_job, {"url": url, "css_selector": css_selector, "text": text})
    return str(result)


@tool(["Sage"])
def click_element(url: str, css_selector: str) -> str:
    """
    Visit a URL and click the element matching `css_selector`. Returns the
    URL of the page after the click (which may have changed).
    ---
    url: an http(s) URL to visit
    css_selector: a CSS selector for the element to click
    """
    def _job(driver, data):
        driver.get(data["url"])
        driver.click(data["css_selector"])
        return driver.current_url or data["url"]

    result = _run_scalar(_job, {"url": url, "css_selector": css_selector})
    return str(result)


@tool(["Sage"])
def search_google(query: str, max_results: int = 5) -> list[dict]:
    """
    Search Google for `query`. Returns up to `max_results` items as
    `[{title, href}, ...]`.
    ---
    query: search terms
    max_results: how many results to return (default 5; cap at 10)
    """
    def _job(driver, data):
        # google_get is botasaurus's Google-aware loader
        if hasattr(driver, "google_get"):
            driver.google_get(data["query"])
        else:
            from urllib.parse import quote_plus
            driver.get(f"https://www.google.com/search?q={quote_plus(data['query'])}")
        results = driver.select_all("div.g a, div.tF2Cxc a") or []
        out = []
        for a in results:
            try:
                href = a.get_attribute("href") or ""
                text = (a.text or "").strip()
                if href and href.startswith("http"):
                    out.append({"title": text, "href": href})
            except Exception:
                continue
            if len(out) >= data["max_results"]:
                break
        return out

    capped = max(1, min(int(max_results), 10))
    return _run_list(_job, {"query": query, "max_results": capped})
