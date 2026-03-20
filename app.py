"""
Web Intelligence API — Extract structured data from any URL.

Built for AI agents that need to consume web pages as clean, structured JSON —
not raw HTML. Handles JS rendering, content extraction, table parsing, metadata,
and full Markdown output suitable for feeding directly into language models.

NEW in v1.1.0:
  - POST /api/feed — Parse RSS 2.0 and Atom 1.0 feeds into structured JSON items.
    Returns per-item: title, link, pub_date, summary, content, author, guid, media_url.
    autodiscover=true: if given a web page URL, finds and follows feed links automatically.
  - POST /api/tech — Detect 60+ technologies from headers + HTML analysis.
    Categories: CMS, JS/CSS frameworks, analytics, CRM, CDN, hosting, server, runtime, libraries.
    Uses: HTTP headers, script/CSS src URLs, HTML attributes, meta generator, cookies.

Version: 1.1.0
"""
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field
from typing import Any, Dict, List, Literal, Optional, Union
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import urllib.parse
from datetime import datetime, timezone

# ──────────────────────────────────────────────
# Optional heavy deps — graceful degradation
# ──────────────────────────────────────────────
HTTPX_AVAILABLE = False
BS4_AVAILABLE = False
TRAFILATURA_AVAILABLE = False
PLAYWRIGHT_AVAILABLE = False
DATEUTIL_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    pass

try:
    from bs4 import BeautifulSoup
    import bs4
    BS4_AVAILABLE = True
except ImportError:
    pass

try:
    import trafilatura
    from trafilatura.settings import use_config
    TRAFILATURA_AVAILABLE = True
except ImportError:
    pass

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

try:
    from dateutil import parser as dateutil_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VERSION = "1.1.0"

app = FastAPI(
    title="Web Intelligence API",
    version=VERSION,
    description=(
        "Extract structured data from any URL. Built for AI agents that need "
        "clean JSON from web pages — not raw HTML."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "20"))
MAX_TIMEOUT = int(os.getenv("MAX_TIMEOUT", "60"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "10"))
MAX_CONTENT_SIZE = int(os.getenv("MAX_CONTENT_SIZE", str(5 * 1024 * 1024)))  # 5 MB

USER_AGENTS = [
    "Mozilla/5.0 (compatible; WebIntelBot/1.0; +https://rebaselabs.online)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
]

# ──────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────

class ExtractRequest(BaseModel):
    url: str = Field(..., description="URL to extract content from")
    render_js: bool = Field(False, description="Use headless browser for JS-heavy pages (slower, ~3-5s extra)")
    include_links: bool = Field(True, description="Include hyperlinks in response")
    include_images: bool = Field(True, description="Include image metadata in response")
    include_tables: bool = Field(True, description="Include extracted tables in response")
    include_metadata: bool = Field(True, description="Include OpenGraph / Twitter / schema.org metadata")
    include_markdown: bool = Field(True, description="Include full Markdown rendering of body content")
    include_raw_html: bool = Field(False, description="Include raw HTML in response (large, use sparingly)")
    timeout: int = Field(DEFAULT_TIMEOUT, ge=5, le=MAX_TIMEOUT, description="Request timeout in seconds")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers (e.g. auth cookies)")
    user_agent: Optional[str] = Field(None, description="Custom User-Agent string")
    wait_for: Optional[str] = Field(None, description="CSS selector to wait for before extracting (render_js only)")
    proxy: Optional[str] = Field(None, description="Proxy URL (e.g. http://user:pass@host:port)")


class BatchExtractRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, max_length=MAX_BATCH, description="List of URLs (max 10)")
    render_js: bool = Field(False)
    include_links: bool = Field(True)
    include_images: bool = Field(False)
    include_tables: bool = Field(True)
    include_metadata: bool = Field(True)
    include_markdown: bool = Field(False)
    timeout: int = Field(DEFAULT_TIMEOUT, ge=5, le=MAX_TIMEOUT)
    concurrency: int = Field(3, ge=1, le=5, description="Parallel fetch limit (max 5)")


class MarkdownRequest(BaseModel):
    url: str = Field(..., description="URL to convert to Markdown")
    render_js: bool = Field(False)
    timeout: int = Field(DEFAULT_TIMEOUT, ge=5, le=MAX_TIMEOUT)
    headers: Optional[Dict[str, str]] = Field(None)
    include_links: bool = Field(True, description="Preserve hyperlinks in Markdown output")
    include_images: bool = Field(True, description="Preserve image tags in Markdown output")


class SchemaExtractRequest(BaseModel):
    url: str = Field(..., description="URL to extract from")
    schema: Dict[str, str] = Field(
        ...,
        description=(
            "JSON object mapping field names to plain-English descriptions. "
            "Example: {\"price\": \"product price including currency\", \"rating\": \"star rating out of 5\"}"
        ),
    )
    render_js: bool = Field(False)
    timeout: int = Field(DEFAULT_TIMEOUT, ge=5, le=MAX_TIMEOUT)


# ──────────────────────────────────────────────
# Fetch layer
# ──────────────────────────────────────────────

async def fetch_url_httpx(
    url: str,
    timeout: int,
    headers: Optional[Dict[str, str]] = None,
    user_agent: Optional[str] = None,
    proxy: Optional[str] = None,
) -> tuple[int, str, Dict[str, str], float]:
    """Fetch URL via httpx. Returns (status_code, html, response_headers, elapsed_ms)."""
    if not HTTPX_AVAILABLE:
        raise HTTPException(503, "httpx not available")

    ua = user_agent or USER_AGENTS[0]
    default_headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
    }
    if headers:
        default_headers.update(headers)

    proxy_mounts = None
    if proxy:
        proxy_mounts = {"http://": httpx.AsyncHTTPTransport(proxy=proxy), "https://": httpx.AsyncHTTPTransport(proxy=proxy)}

    t0 = time.monotonic()
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout),
        headers=default_headers,
        mounts=proxy_mounts,
        max_redirects=5,
    ) as client:
        resp = await client.get(url)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Guard against massive responses
        content_length = int(resp.headers.get("content-length", 0))
        if content_length > MAX_CONTENT_SIZE:
            raise HTTPException(413, f"Response too large ({content_length:,} bytes > {MAX_CONTENT_SIZE:,} limit)")

        html = resp.text
        if len(html.encode()) > MAX_CONTENT_SIZE:
            html = html[:MAX_CONTENT_SIZE]  # truncate

        return resp.status_code, html, dict(resp.headers), elapsed_ms


async def fetch_url_playwright(
    url: str,
    timeout: int,
    headers: Optional[Dict[str, str]] = None,
    user_agent: Optional[str] = None,
    wait_for: Optional[str] = None,
) -> tuple[int, str, float]:
    """Fetch URL via headless Chromium. Returns (status_code, html, elapsed_ms)."""
    if not PLAYWRIGHT_AVAILABLE:
        raise HTTPException(503, "Playwright not installed — set render_js=false")

    t0 = time.monotonic()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        context_opts: Dict[str, Any] = {"java_script_enabled": True}
        if user_agent:
            context_opts["user_agent"] = user_agent
        if headers:
            context_opts["extra_http_headers"] = headers

        context = await browser.new_context(**context_opts)
        page = await context.new_page()

        status_code = 200
        try:
            resp = await page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            if resp:
                status_code = resp.status

            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=5000)
                except Exception:
                    pass  # Best-effort selector wait
            else:
                # Brief settle time for dynamic content
                await asyncio.sleep(1.5)

            html = await page.content()
        finally:
            await browser.close()

    elapsed_ms = (time.monotonic() - t0) * 1000
    return status_code, html, elapsed_ms


# ──────────────────────────────────────────────
# Extraction helpers
# ──────────────────────────────────────────────

def _resolve_url(href: str, base_url: str) -> str:
    if not href:
        return ""
    if href.startswith(("http://", "https://", "//")):
        if href.startswith("//"):
            parsed = urllib.parse.urlparse(base_url)
            return f"{parsed.scheme}:{href}"
        return href
    if href.startswith("/"):
        parsed = urllib.parse.urlparse(base_url)
        return f"{parsed.scheme}://{parsed.netloc}{href}"
    return urllib.parse.urljoin(base_url, href)


def _is_external(href: str, base_url: str) -> bool:
    try:
        base_netloc = urllib.parse.urlparse(base_url).netloc.lstrip("www.")
        href_netloc = urllib.parse.urlparse(href).netloc.lstrip("www.")
        return href_netloc != "" and href_netloc != base_netloc
    except Exception:
        return False


def extract_metadata(soup: "BeautifulSoup", url: str) -> Dict[str, Any]:
    """Extract OpenGraph, Twitter Card, and schema.org metadata."""
    meta: Dict[str, Any] = {}

    # Standard meta tags
    for tag in soup.find_all("meta"):
        name = tag.get("name", "") or tag.get("property", "")
        content = tag.get("content", "")
        if not name or not content:
            continue
        key = name.lower().replace(":", "_")
        # OpenGraph
        if name.startswith("og:"):
            meta[key] = content
        # Twitter
        elif name.startswith("twitter:"):
            meta[key] = content
        # Standard
        elif name in ("description", "keywords", "author", "robots", "viewport", "theme-color"):
            meta[name] = content

    # Canonical URL
    canonical_tag = soup.find("link", rel="canonical")
    if canonical_tag:
        meta["canonical_url"] = canonical_tag.get("href", url)

    # Schema.org JSON-LD
    schema_org = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            schema_org.append(data)
        except (json.JSONDecodeError, TypeError):
            pass
    if schema_org:
        meta["schema_org"] = schema_org

    # Favicon
    for rel in ("shortcut icon", "icon"):
        icon_tag = soup.find("link", rel=re.compile(rel, re.I))
        if icon_tag:
            meta["favicon"] = _resolve_url(icon_tag.get("href", ""), url)
            break

    return meta


def extract_headings(soup: "BeautifulSoup") -> List[Dict[str, Any]]:
    """Extract heading hierarchy."""
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = tag.get_text(strip=True)
        if text:
            headings.append({
                "level": int(tag.name[1]),
                "text": text,
                "id": tag.get("id"),
            })
    return headings


def extract_tables(soup: "BeautifulSoup") -> List[Dict[str, Any]]:
    """Extract HTML tables as structured JSON arrays."""
    tables_data = []
    for table in soup.find_all("table"):
        headers: List[str] = []
        rows: List[List[str]] = []

        # Try <thead> first
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

        # Body rows
        tbody = table.find("tbody") or table
        for row in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells or all(c == "" for c in cells):
                continue
            # If no explicit headers, use first data row
            if not headers and row.find("th"):
                headers = cells
                continue
            rows.append(cells)

        if rows or headers:
            tables_data.append({
                "headers": headers,
                "rows": rows[:200],  # Cap at 200 rows
                "row_count": len(rows),
            })

    return tables_data


def extract_links(soup: "BeautifulSoup", base_url: str) -> List[Dict[str, Any]]:
    """Extract all hyperlinks with type classification."""
    seen: set = set()
    links = []

    for a in soup.find_all("a", href=True):
        raw_href = a.get("href", "").strip()
        if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        resolved = _resolve_url(raw_href, base_url)
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)

        link_type = "external" if _is_external(resolved, base_url) else "internal"
        link = {
            "text": a.get_text(strip=True) or a.get("title", "") or a.get("aria-label", ""),
            "href": resolved,
            "type": link_type,
        }
        title = a.get("title")
        if title:
            link["title"] = title
        rel = a.get("rel")
        if rel:
            link["rel"] = " ".join(rel) if isinstance(rel, list) else rel

        links.append(link)

    return links[:500]  # Cap at 500 links


def extract_images(soup: "BeautifulSoup", base_url: str) -> List[Dict[str, Any]]:
    """Extract image metadata."""
    images = []
    seen: set = set()

    for img in soup.find_all("img"):
        # Support both src and data-src (lazy loading)
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
        if not src or src in seen:
            continue
        resolved = _resolve_url(src, base_url)
        if not resolved:
            continue
        seen.add(resolved)

        images.append({
            "src": resolved,
            "alt": img.get("alt", ""),
            "title": img.get("title"),
            "width": img.get("width"),
            "height": img.get("height"),
            "loading": img.get("loading"),
        })

    return images[:100]  # Cap at 100 images


def html_to_markdown(soup: "BeautifulSoup", include_links: bool = True, include_images: bool = True, base_url: str = "") -> str:
    """Convert main content HTML to clean Markdown."""
    # Use trafilatura for content extraction if available, then convert
    lines: List[str] = []

    def process_node(node: Any, depth: int = 0) -> None:
        if isinstance(node, bs4.NavigableString):
            text = str(node)
            # Only add non-whitespace text at leaf level
            if text.strip():
                lines.append(text.strip())
            return

        if not hasattr(node, "name") or node.name is None:
            return

        tag = node.name.lower()

        # Skip non-content tags
        if tag in ("script", "style", "noscript", "nav", "footer", "header",
                   "aside", "form", "button", "select", "option"):
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            text = node.get_text(strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")

        elif tag in ("p", "div", "section", "article", "main"):
            lines.append("\n")
            for child in node.children:
                process_node(child, depth + 1)
            lines.append("\n")

        elif tag == "br":
            lines.append("\n")

        elif tag in ("strong", "b"):
            text = node.get_text(strip=True)
            if text:
                lines.append(f"**{text}**")

        elif tag in ("em", "i"):
            text = node.get_text(strip=True)
            if text:
                lines.append(f"*{text}*")

        elif tag == "code":
            text = node.get_text(strip=True)
            if text:
                lines.append(f"`{text}`")

        elif tag == "pre":
            code_tag = node.find("code")
            code_text = (code_tag or node).get_text()
            lang = ""
            if code_tag:
                classes = code_tag.get("class", [])
                for cls in classes:
                    if cls.startswith("language-"):
                        lang = cls.replace("language-", "")
                        break
            lines.append(f"\n```{lang}\n{code_text.strip()}\n```\n")

        elif tag == "blockquote":
            text = node.get_text(strip=True)
            if text:
                quoted = "\n".join(f"> {line}" for line in text.split("\n") if line.strip())
                lines.append(f"\n{quoted}\n")

        elif tag in ("ul", "ol"):
            lines.append("\n")
            for i, li in enumerate(node.find_all("li", recursive=False)):
                prefix = f"{i + 1}." if tag == "ol" else "-"
                li_text = li.get_text(strip=True)
                if li_text:
                    lines.append(f"{prefix} {li_text}")
            lines.append("\n")

        elif tag == "a" and include_links:
            href = node.get("href", "")
            text = node.get_text(strip=True)
            if href and text:
                resolved = _resolve_url(href, base_url)
                lines.append(f"[{text}]({resolved})")
            else:
                for child in node.children:
                    process_node(child, depth + 1)

        elif tag == "img" and include_images:
            src = node.get("src") or node.get("data-src", "")
            alt = node.get("alt", "")
            if src:
                resolved = _resolve_url(src, base_url)
                lines.append(f"![{alt}]({resolved})")

        elif tag == "hr":
            lines.append("\n---\n")

        elif tag == "table":
            # Simple table → Markdown table
            headers = []
            thead = node.find("thead")
            if thead:
                header_row = thead.find("tr")
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

            if headers:
                lines.append("\n| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                tbody = node.find("tbody") or node
                for row in tbody.find_all("tr")[:20]:
                    cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                    if cells:
                        lines.append("| " + " | ".join(cells) + " |")
                lines.append("\n")
            else:
                for child in node.children:
                    process_node(child, depth + 1)

        else:
            for child in node.children:
                process_node(child, depth + 1)

    # Try to find main content area
    main_content = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"content|main|article|post|entry", re.I)) or
        soup.find(class_=re.compile(r"content|main|article|post|entry|body", re.I)) or
        soup.find("body") or
        soup
    )

    process_node(main_content)

    # Clean up excessive blank lines
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def parse_date(date_str: Optional[str]) -> Optional[str]:
    """Parse various date formats to ISO 8601."""
    if not date_str:
        return None
    if DATEUTIL_AVAILABLE:
        try:
            dt = dateutil_parser.parse(date_str, fuzzy=True)
            return dt.isoformat()
        except Exception:
            pass
    # Fallback: return as-is if it looks like a date
    if re.search(r"\d{4}", date_str):
        return date_str.strip()
    return None


def extract_dates(soup: "BeautifulSoup", metadata: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Extract published and modified dates."""
    published = None
    modified = None

    # Check metadata first
    published = published or parse_date(metadata.get("article_published_time"))
    published = published or parse_date(metadata.get("og_article_published_time"))
    modified = modified or parse_date(metadata.get("article_modified_time"))
    modified = modified or parse_date(metadata.get("og_article_modified_time"))

    # Schema.org dates
    for schema in metadata.get("schema_org", []):
        if isinstance(schema, dict):
            published = published or parse_date(schema.get("datePublished"))
            modified = modified or parse_date(schema.get("dateModified"))

    # time tags in HTML
    for time_tag in soup.find_all("time"):
        dt_str = time_tag.get("datetime") or time_tag.get_text(strip=True)
        parsed = parse_date(dt_str)
        if parsed and not published:
            published = parsed

    # Meta name="date" / pubdate
    for meta in soup.find_all("meta", attrs={"name": re.compile(r"date|pubdate|publish", re.I)}):
        if not published:
            published = parse_date(meta.get("content"))

    return published, modified


def extract_author(soup: "BeautifulSoup", metadata: Dict[str, Any]) -> Optional[str]:
    """Extract article author."""
    # Metadata
    author = metadata.get("author")

    # Schema.org
    for schema in metadata.get("schema_org", []):
        if isinstance(schema, dict):
            author_data = schema.get("author")
            if isinstance(author_data, dict):
                author = author or author_data.get("name")
            elif isinstance(author_data, str):
                author = author or author_data
            elif isinstance(author_data, list) and author_data:
                first = author_data[0]
                if isinstance(first, dict):
                    author = author or first.get("name")
                elif isinstance(first, str):
                    author = author or first

    # Common author markup patterns
    if not author:
        for selector in [
            {"rel": "author"}, {"itemprop": "author"}, {"class": re.compile(r"author|byline", re.I)},
        ]:
            tag = soup.find(attrs=selector)
            if tag:
                author = tag.get_text(strip=True)
                # Trim long strings (probably not just an author name)
                if author and len(author) > 100:
                    author = None
                else:
                    break

    return author


def extract_content_trafilatura(html: str, url: str) -> Optional[str]:
    """Use trafilatura to extract main body content."""
    if not TRAFILATURA_AVAILABLE:
        return None
    try:
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "5")
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            config=config,
        )
        return text
    except Exception as e:
        logger.warning(f"trafilatura extraction failed: {e}")
        return None


def soup_main_text(soup: "BeautifulSoup") -> str:
    """Fallback: extract visible text from main content area."""
    # Remove noise elements
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer",
                               "header", "aside", "form", "iframe", "advertisement"]):
        tag.decompose()

    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"content|main|article|post", re.I)) or
        soup.find(class_=re.compile(r"content|main|article|post|entry", re.I)) or
        soup.find("body") or
        soup
    )
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text)) if text else 0


def reading_time(word_count: int) -> float:
    """Estimate reading time in minutes at 200 WPM."""
    return round(word_count / 200, 1)


# ──────────────────────────────────────────────
# Core extraction pipeline
# ──────────────────────────────────────────────

async def extract_from_url(req: ExtractRequest) -> Dict[str, Any]:
    """Full extraction pipeline. Returns structured result dict."""
    t0 = time.monotonic()
    url = str(req.url)
    status_code = 200
    fetch_time_ms = 0.0

    # Fetch HTML
    if req.render_js:
        status_code, html, fetch_time_ms = await fetch_url_playwright(
            url, req.timeout, req.headers, req.user_agent, req.wait_for
        )
    else:
        status_code, html, response_headers, fetch_time_ms = await fetch_url_httpx(
            url, req.timeout, req.headers, req.user_agent, req.proxy
        )

    if not BS4_AVAILABLE:
        raise HTTPException(503, "beautifulsoup4 not available")

    soup = BeautifulSoup(html, "lxml" if _has_lxml() else "html.parser")

    # ── Core fields ──
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # ── Metadata ──
    metadata: Dict[str, Any] = {}
    if req.include_metadata:
        metadata = extract_metadata(soup, url)

    # OG title overrides <title> if richer
    og_title = metadata.get("og_title", "")
    if og_title and og_title != title:
        title = og_title or title

    # Description
    description = (
        metadata.get("og_description") or
        metadata.get("description") or
        metadata.get("twitter_description") or
        ""
    )

    # Author
    author = extract_author(soup, metadata)

    # Dates
    published_date, modified_date = extract_dates(soup, metadata)

    # Language
    language = soup.find("html").get("lang", "") if soup.find("html") else ""
    if not language:
        lang_meta = soup.find("meta", attrs={"http-equiv": "Content-Language"})
        if lang_meta:
            language = lang_meta.get("content", "")
    language = language.split("-")[0].lower() if language else None

    # ── Content extraction ──
    body_text = extract_content_trafilatura(html, url) or soup_main_text(soup)
    word_count = count_words(body_text)

    # ── Structural data ──
    headings = extract_headings(soup)
    tables = extract_tables(soup) if req.include_tables else []
    links = extract_links(soup, url) if req.include_links else []
    images = extract_images(soup, url) if req.include_images else []
    markdown = html_to_markdown(soup, req.include_links, req.include_images, url) if req.include_markdown else None

    canonical_url = metadata.pop("canonical_url", url)

    total_ms = (time.monotonic() - t0) * 1000

    result: Dict[str, Any] = {
        "url": url,
        "canonical_url": canonical_url,
        "title": title,
        "description": description,
        "author": author,
        "published_date": published_date,
        "modified_date": modified_date,
        "language": language,
        "content": {
            "text": body_text,
            "headings": headings,
        },
        "stats": {
            "word_count": word_count,
            "reading_time_minutes": reading_time(word_count),
            "fetch_time_ms": round(fetch_time_ms, 1),
            "total_time_ms": round(total_ms, 1),
            "status_code": status_code,
            "rendered_js": req.render_js,
        },
    }

    if req.include_tables:
        result["content"]["tables"] = tables
    if req.include_links:
        result["links"] = links
    if req.include_images:
        result["images"] = images
    if req.include_metadata:
        result["metadata"] = metadata
    if req.include_markdown:
        result["content"]["markdown"] = markdown
    if req.include_raw_html:
        result["raw_html"] = html

    return result


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────
# API routes
# ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Web Intelligence API",
        "version": VERSION,
        "status": "online",
        "endpoints": {
            "extract":          "POST /api/extract — structured content extraction (text, tables, links, metadata)",
            "extract_batch":    "POST /api/extract/batch — extract from up to 10 URLs concurrently",
            "extract_markdown": "POST /api/extract/markdown — clean Markdown output for LLM ingestion",
            "extract_schema":   "POST /api/extract/schema — heuristic field extraction from natural-language schema",
            "feed":             "POST /api/feed [v1.1] — parse RSS/Atom feeds, with HTML autodiscovery",
            "tech":             "POST /api/tech [v1.1] — detect tech stack (CMS, frameworks, analytics, CDN, server)",
            "health":           "GET /health",
        },
        "new_in_v1_1_0": [
            "POST /api/feed — RSS 2.0 + Atom 1.0 feed parsing. Returns items with title/link/pub_date/summary/author.",
            "  autodiscover=true: follow <link rel=alternate> to find feeds on a regular web page.",
            "POST /api/tech — Detect 60+ technologies: CMS, JS/CSS frameworks, analytics, CDN, hosting, server.",
            "  Uses: HTTP headers, script/CSS URLs, HTML attributes, meta generator, cookies.",
            "  Returns: technologies[] with name/category/evidence, categories dict, server_header, meta_generator.",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "capabilities": {
            "http_fetch": HTTPX_AVAILABLE,
            "html_parsing": BS4_AVAILABLE,
            "content_extraction": TRAFILATURA_AVAILABLE,
            "js_rendering": PLAYWRIGHT_AVAILABLE,
            "date_parsing": DATEUTIL_AVAILABLE,
        },
    }


@app.post("/api/extract")
async def extract(req: ExtractRequest):
    """
    Extract structured content from a URL.

    Returns title, description, author, dates, clean body text,
    headings, tables, links, images, metadata, and optional Markdown.

    Set `render_js=true` for JavaScript-heavy pages (React, Vue, Angular).
    """
    try:
        result = await extract_from_url(req)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction failed for {req.url}: {e}", exc_info=True)
        raise HTTPException(500, f"Extraction failed: {str(e)}")


@app.post("/api/extract/batch")
async def extract_batch(req: BatchExtractRequest):
    """
    Extract structured content from multiple URLs concurrently.

    Max 10 URLs per request. Use `concurrency` (1-5) to control parallel fetches.
    """
    results = []
    errors = []
    semaphore = asyncio.Semaphore(req.concurrency)

    async def fetch_one(url: str) -> Dict[str, Any]:
        async with semaphore:
            single_req = ExtractRequest(
                url=url,
                render_js=req.render_js,
                include_links=req.include_links,
                include_images=req.include_images,
                include_tables=req.include_tables,
                include_metadata=req.include_metadata,
                include_markdown=req.include_markdown,
                timeout=req.timeout,
            )
            try:
                return await extract_from_url(single_req)
            except HTTPException as e:
                return {"url": url, "error": e.detail, "status_code": e.status_code}
            except Exception as e:
                return {"url": url, "error": str(e)}

    tasks = [fetch_one(url) for url in req.urls]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for url, result in zip(req.urls, raw_results):
        if isinstance(result, Exception):
            errors.append({"url": url, "error": str(result)})
        elif "error" in result:
            errors.append(result)
        else:
            results.append(result)

    return {
        "success_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors,
    }


@app.post("/api/extract/markdown")
async def extract_markdown(req: MarkdownRequest):
    """
    Fetch a URL and return its content as clean Markdown.

    Ideal for feeding web content directly into language models.
    Strips all HTML, preserves structure (headings, lists, tables, code blocks).
    """
    try:
        extract_req = ExtractRequest(
            url=req.url,
            render_js=req.render_js,
            include_links=req.include_links,
            include_images=req.include_images,
            include_tables=True,
            include_metadata=True,
            include_markdown=True,
            timeout=req.timeout,
            headers=req.headers,
        )
        result = await extract_from_url(extract_req)
        markdown = result.get("content", {}).get("markdown", "")

        return {
            "url": req.url,
            "title": result.get("title", ""),
            "markdown": markdown,
            "word_count": result.get("stats", {}).get("word_count", 0),
            "fetch_time_ms": result.get("stats", {}).get("fetch_time_ms", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Markdown extraction failed: {str(e)}")


@app.post("/api/extract/schema")
async def extract_schema(req: SchemaExtractRequest):
    """
    Extract specific fields from a URL using a plain-English schema.

    Pass a JSON object mapping field names to descriptions of what to extract.
    The API returns values for each field found on the page.

    Example schema: {"price": "product price with currency symbol", "rating": "star rating out of 5"}
    """
    if not BS4_AVAILABLE:
        raise HTTPException(503, "beautifulsoup4 not available")

    try:
        extract_req = ExtractRequest(
            url=req.url,
            render_js=req.render_js,
            include_links=False,
            include_images=False,
            include_tables=True,
            include_metadata=True,
            include_markdown=True,
            timeout=req.timeout,
        )
        result = await extract_from_url(extract_req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Fetch failed: {str(e)}")

    # Build extraction context
    context = result.get("content", {}).get("text", "") or ""
    markdown = result.get("content", {}).get("markdown", "") or ""
    metadata = result.get("metadata", {})
    tables = result.get("content", {}).get("tables", [])

    extracted: Dict[str, Any] = {}

    for field_name, description in req.schema.items():
        value = _schema_extract_field(
            field_name=field_name,
            description=description,
            text=context,
            markdown=markdown,
            metadata=metadata,
            tables=tables,
            full_result=result,
        )
        extracted[field_name] = value

    return {
        "url": req.url,
        "title": result.get("title", ""),
        "extracted": extracted,
        "fetch_time_ms": result.get("stats", {}).get("fetch_time_ms", 0),
    }


def _schema_extract_field(
    field_name: str,
    description: str,
    text: str,
    markdown: str,
    metadata: Dict[str, Any],
    tables: List[Dict[str, Any]],
    full_result: Dict[str, Any],
) -> Any:
    """
    Heuristic field extraction based on field name + description.
    Uses regex patterns calibrated to common web content structures.
    """
    desc_lower = description.lower()
    name_lower = field_name.lower()
    combined = f"{name_lower} {desc_lower}"

    # ── Price extraction ──
    if any(w in combined for w in ("price", "cost", "amount", "fee", "rate")):
        patterns = [
            r"[\$£€¥₹]\s*[\d,]+(?:\.\d{2})?",
            r"[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|INR)",
            r"(?i)price[:\s]+[\$£€]?\s*([\d,]+(?:\.\d{2})?)",
        ]
        for pat in patterns:
            m = re.search(pat, text or markdown)
            if m:
                return m.group(0).strip()

    # ── Rating / score extraction ──
    if any(w in combined for w in ("rating", "score", "stars", "review")):
        patterns = [
            r"(\d(?:\.\d)?)\s*(?:out of|\/)\s*5",
            r"(\d(?:\.\d)?)\s*(?:out of|\/)\s*10",
            r"(\d{1,2}(?:\.\d)?)\s*(?:stars?|points?|\/10|\/5)",
            r"(?i)rating[:\s]+(\d(?:\.\d)?)",
        ]
        for pat in patterns:
            m = re.search(pat, text or markdown)
            if m:
                return m.group(0).strip()

    # ── Author / byline ──
    if any(w in combined for w in ("author", "writer", "byline", "by")):
        if full_result.get("author"):
            return full_result["author"]

    # ── Date fields ──
    if any(w in combined for w in ("date", "publish", "written", "posted", "updated", "modified")):
        if "modified" in combined or "updated" in combined:
            return full_result.get("modified_date") or full_result.get("published_date")
        return full_result.get("published_date") or full_result.get("modified_date")

    # ── Title ──
    if any(w in combined for w in ("title", "headline", "heading", "name")):
        return full_result.get("title", "")

    # ── Description / summary ──
    if any(w in combined for w in ("description", "summary", "excerpt", "abstract")):
        return full_result.get("description", "")

    # ── Contact / email ──
    if any(w in combined for w in ("email", "contact", "e-mail")):
        m = re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b", text or "")
        if m:
            return m.group(0)

    # ── Phone ──
    if any(w in combined for w in ("phone", "tel", "telephone", "mobile", "number")):
        m = re.search(r"(?:\+?\d[\d\s\-\(\)\.]{7,}\d)", text or "")
        if m:
            return m.group(0).strip()

    # ── Table column extraction ──
    # If description mentions "table" or "list", look in tables
    if any(w in combined for w in ("table", "list", "column", "row")):
        for table in tables:
            headers = [h.lower() for h in (table.get("headers") or [])]
            for i, h in enumerate(headers):
                if name_lower in h or any(word in h for word in name_lower.split()):
                    col_values = [row[i] for row in (table.get("rows") or []) if len(row) > i]
                    if col_values:
                        return col_values[:20]

    # ── Generic keyword search in text ──
    # Look for "field_name: value" or "field_name - value" patterns
    patterns = [
        rf"(?i)\b{re.escape(field_name)}\s*[:\-–]\s*([^\n\r|]+)",
        rf"(?i)\b{re.escape(name_lower.replace('_', ' '))}\s*[:\-–]\s*([^\n\r|]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text or "")
        if m:
            return m.group(1).strip()[:200]

    # ── OG / schema.org metadata fields ──
    for key, val in metadata.items():
        if name_lower in key.lower():
            return val

    return None


# ──────────────────────────────────────────────
# v1.1.0: RSS/Atom feed parsing
# ──────────────────────────────────────────────

import xml.etree.ElementTree as ET

_RSS_NS = {
    "dc":      "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "media":   "http://search.yahoo.com/mrss/",
    "atom10":  "http://www.w3.org/2005/Atom",
}


def _xml_text(el: Optional[ET.Element], fallback: str = "") -> str:
    if el is None:
        return fallback
    return (el.text or "").strip()


def _parse_rss2(root: ET.Element) -> Dict[str, Any]:
    """Parse RSS 2.0 feed."""
    channel = root.find("channel")
    if channel is None:
        raise ValueError("No <channel> element found in RSS feed")

    feed_title = _xml_text(channel.find("title"))
    feed_link  = _xml_text(channel.find("link"))
    feed_desc  = _xml_text(channel.find("description"))

    items = []
    for item in channel.findall("item"):
        title   = _xml_text(item.find("title"))
        link    = _xml_text(item.find("link"))
        desc_el = item.find("description")
        summary = ""
        if desc_el is not None and desc_el.text:
            # Strip HTML tags from description
            summary = re.sub(r"<[^>]+>", "", desc_el.text).strip()[:1000]

        # content:encoded
        content_el = item.find("content:encoded", _RSS_NS)
        content = ""
        if content_el is not None and content_el.text:
            content = re.sub(r"<[^>]+>", "", content_el.text).strip()[:2000]

        pub_date = _xml_text(item.find("pubDate"))
        guid     = _xml_text(item.find("guid"))
        author   = _xml_text(item.find("author"))
        if not author:
            dc_creator = item.find("dc:creator", _RSS_NS)
            author = _xml_text(dc_creator)

        # Enclosure (podcast audio, image)
        enclosure = item.find("enclosure")
        media_url = None
        if enclosure is not None:
            media_url = enclosure.get("url")

        # media:thumbnail
        media_thumb = item.find("media:thumbnail", _RSS_NS)
        if media_thumb is not None and not media_url:
            media_url = media_thumb.get("url")

        items.append({
            "title":      title,
            "link":       link,
            "pub_date":   pub_date,
            "summary":    summary or content[:500],
            "content":    content,
            "author":     author,
            "guid":       guid,
            "media_url":  media_url,
        })

    return {
        "format":     "rss2",
        "title":      feed_title,
        "link":       feed_link,
        "description": feed_desc,
        "items":      items,
        "item_count": len(items),
    }


def _parse_atom(root: ET.Element) -> Dict[str, Any]:
    """Parse Atom 1.0 feed."""
    ns = "http://www.w3.org/2005/Atom"

    def _t(el: Optional[ET.Element]) -> str:
        return (el.text or "").strip() if el is not None else ""

    feed_title = _t(root.find(f"{{{ns}}}title"))
    link_el    = root.find(f".//{{{ns}}}link[@rel='alternate']")
    if link_el is None:
        link_el = root.find(f"{{{ns}}}link")
    feed_link  = link_el.get("href", "") if link_el is not None else ""
    feed_desc_el = root.find(f"{{{ns}}}subtitle")
    feed_desc  = _t(feed_desc_el)

    items = []
    for entry in root.findall(f"{{{ns}}}entry"):
        title   = _t(entry.find(f"{{{ns}}}title"))

        entry_link_el = entry.find(f"{{{ns}}}link[@rel='alternate']")
        if entry_link_el is None:
            entry_link_el = entry.find(f"{{{ns}}}link")
        link = entry_link_el.get("href", "") if entry_link_el is not None else ""

        # Summary or content
        summary_el = entry.find(f"{{{ns}}}summary")
        content_el = entry.find(f"{{{ns}}}content")
        raw_text = ""
        if summary_el is not None and summary_el.text:
            raw_text = re.sub(r"<[^>]+>", "", summary_el.text).strip()
        elif content_el is not None and content_el.text:
            raw_text = re.sub(r"<[^>]+>", "", content_el.text).strip()

        published = _t(entry.find(f"{{{ns}}}published")) or _t(entry.find(f"{{{ns}}}updated"))
        entry_id  = _t(entry.find(f"{{{ns}}}id"))

        author_el   = entry.find(f"{{{ns}}}author")
        author_name = ""
        if author_el is not None:
            author_name = _t(author_el.find(f"{{{ns}}}name"))

        items.append({
            "title":    title,
            "link":     link,
            "pub_date": published,
            "summary":  raw_text[:1000],
            "content":  raw_text[:2000],
            "author":   author_name,
            "guid":     entry_id,
            "media_url": None,
        })

    return {
        "format":     "atom",
        "title":      feed_title,
        "link":       feed_link,
        "description": feed_desc,
        "items":      items,
        "item_count": len(items),
    }


def _autodiscover_feeds(html: str, base_url: str) -> List[Dict[str, str]]:
    """Find feed URLs in a page's <head> via <link> autodiscovery."""
    if not BS4_AVAILABLE:
        return []
    soup = BeautifulSoup(html, "html.parser")
    feeds = []
    for link in soup.find_all("link", rel=True):
        rel = " ".join(link.get("rel", [])).lower()
        if "alternate" in rel:
            type_ = link.get("type", "").lower()
            if any(t in type_ for t in ("rss", "atom", "feed", "xml")):
                href = link.get("href", "")
                if href:
                    # Resolve relative URL
                    href = urllib.parse.urljoin(base_url, href)
                    feeds.append({
                        "url":   href,
                        "type":  type_,
                        "title": link.get("title", ""),
                    })
    return feeds


class FeedRequest(BaseModel):
    url: str = Field(..., description="Feed URL (RSS or Atom) or a web page with feed autodiscovery")
    limit: int = Field(50, ge=1, le=200, description="Max items to return (default 50)")
    autodiscover: bool = Field(
        False,
        description=(
            "If true and URL is a web page (not a feed), follow feed autodiscovery links. "
            "Fetches the first discovered RSS/Atom feed URL."
        ),
    )
    timeout: int = Field(DEFAULT_TIMEOUT, ge=1, le=MAX_TIMEOUT)


@app.post("/api/feed")
async def parse_feed(req: FeedRequest):
    """
    Parse an RSS 2.0 or Atom 1.0 feed.

    **v1.1.0** — Returns structured feed data with per-item fields.

    Pass any RSS or Atom feed URL and receive:
    - Feed-level metadata: title, link, description, format
    - `items[]`: title, link, pub_date, summary, content, author, guid, media_url

    **Autodiscovery mode** (`autodiscover: true`): If you pass a regular web page URL
    (not a feed), the API will find and follow any `<link rel="alternate" type="application/rss+xml">`
    tags to locate the feed automatically.

    **Example:**
    ```json
    {"url": "https://hnrss.org/frontpage", "limit": 20}
    ```

    **Autodiscovery example:**
    ```json
    {"url": "https://news.ycombinator.com", "autodiscover": true, "limit": 10}
    ```
    """
    if not HTTPX_AVAILABLE:
        raise HTTPException(503, "httpx not available")

    try:
        status, body, resp_headers, elapsed_ms = await fetch_url_httpx(req.url, req.timeout)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch {req.url}: {e}")

    if status >= 400:
        raise HTTPException(status, f"HTTP {status} from {req.url}")

    content_type = resp_headers.get("content-type", "").lower()

    # Check if we got HTML instead of a feed
    is_html = "text/html" in content_type or body.strip().lower().startswith("<!doctype") or body.strip().lower().startswith("<html")
    is_xml  = any(t in content_type for t in ("xml", "rss", "atom", "feed")) or (
        body.strip().startswith("<?xml") or body.strip().startswith("<rss") or body.strip().startswith("<feed")
    )

    if is_html and not is_xml:
        if req.autodiscover:
            discovered = _autodiscover_feeds(body, req.url)
            if not discovered:
                raise HTTPException(
                    422,
                    {
                        "error": "No RSS/Atom feeds discovered on this page",
                        "url": req.url,
                        "tip": "Pass a direct feed URL, or enable autodiscover on a page that links to a feed",
                    },
                )
            # Fetch the first discovered feed
            feed_url = discovered[0]["url"]
            try:
                status, body, resp_headers, elapsed_ms2 = await fetch_url_httpx(feed_url, req.timeout)
                elapsed_ms += elapsed_ms2
            except Exception as e:
                raise HTTPException(502, f"Failed to fetch discovered feed {feed_url}: {e}")
        else:
            raise HTTPException(
                422,
                {
                    "error": "URL returned HTML, not a feed",
                    "url": req.url,
                    "tip": "Pass a direct RSS/Atom URL, or set autodiscover=true to find feeds on a web page",
                },
            )

    # Parse the feed XML
    try:
        # Strip BOM/encoding declaration issues
        xml_body = body.strip()
        if xml_body.startswith("\ufeff"):
            xml_body = xml_body[1:]
        root = ET.fromstring(xml_body)
    except ET.ParseError as e:
        raise HTTPException(422, f"XML parse error: {e}")

    # Detect format
    tag_lower = root.tag.lower()
    if "feed" in tag_lower or "{http://www.w3.org/2005/atom}" in root.tag:
        try:
            parsed = _parse_atom(root)
        except Exception as e:
            raise HTTPException(422, f"Atom parse error: {e}")
    elif "rss" in tag_lower or root.find("channel") is not None:
        try:
            parsed = _parse_rss2(root)
        except Exception as e:
            raise HTTPException(422, f"RSS parse error: {e}")
    else:
        raise HTTPException(422, f"Unrecognised feed format (root tag: {root.tag})")

    # Apply limit
    parsed["items"] = parsed["items"][:req.limit]
    parsed["item_count"] = len(parsed["items"])
    parsed["url"] = req.url
    parsed["fetch_time_ms"] = round(elapsed_ms)

    return parsed


# ──────────────────────────────────────────────
# v1.1.0: Tech stack detection
# ──────────────────────────────────────────────

# Technology fingerprints: (name, category, signal_type, pattern)
# signal_type: "header", "script_src", "html_comment", "meta_generator",
#              "css_class", "html_attr", "json_key", "cookie"
_TECH_FINGERPRINTS: List[tuple] = [
    # CMS / Platforms
    ("WordPress",    "cms",       "script_src",     r"wp-content|wp-includes"),
    ("WordPress",    "cms",       "html_comment",   r"WordPress"),
    ("WordPress",    "cms",       "meta_generator", r"WordPress"),
    ("Ghost",        "cms",       "meta_generator", r"Ghost"),
    ("Ghost",        "cms",       "html_attr",      r"ghost"),
    ("Drupal",       "cms",       "meta_generator", r"Drupal"),
    ("Drupal",       "cms",       "script_src",     r"/sites/default/files|drupal\.js"),
    ("Joomla",       "cms",       "meta_generator", r"Joomla"),
    ("Joomla",       "cms",       "script_src",     r"/media/jui/"),
    ("Shopify",      "ecommerce", "script_src",     r"cdn\.shopify\.com"),
    ("Shopify",      "ecommerce", "cookie",         r"^cart$|^_shopify"),
    ("WooCommerce",  "ecommerce", "css_class",      r"woocommerce"),
    ("Magento",      "ecommerce", "script_src",     r"mage/"),
    ("Squarespace",  "website_builder", "script_src", r"static\.squarespace\.com"),
    ("Wix",          "website_builder", "html_attr",  r"data-wix"),
    ("Wix",          "website_builder", "script_src", r"static\.wixstatic\.com"),
    ("Webflow",      "website_builder", "html_attr",  r"data-wf-site"),
    ("Webflow",      "website_builder", "script_src", r"webflow\.com"),
    # JS Frameworks
    ("React",        "frontend",  "html_attr",      r"data-reactroot|data-reactid"),
    ("React",        "frontend",  "script_src",     r"react(?:\.min)?\.js"),
    ("Next.js",      "frontend",  "html_attr",      r"__NEXT_DATA__|data-next-page"),
    ("Next.js",      "frontend",  "script_src",     r"/_next/static"),
    ("Vue.js",       "frontend",  "html_attr",      r"data-v-\w+"),
    ("Vue.js",       "frontend",  "script_src",     r"vue(?:\.min)?\.js"),
    ("Angular",      "frontend",  "html_attr",      r"ng-version|_nghost|_ngcontent"),
    ("Angular",      "frontend",  "script_src",     r"angular(?:\.min)?\.js"),
    ("Nuxt.js",      "frontend",  "html_attr",      r"data-n-head|nuxt__"),
    ("Svelte",       "frontend",  "html_attr",      r"class=\"s-\w+\""),
    ("Ember.js",     "frontend",  "html_attr",      r"data-ember-action|ember-view"),
    # CSS Frameworks
    ("Bootstrap",    "css_framework", "script_src", r"bootstrap(?:\.min)?\.(?:js|css)"),
    ("Bootstrap",    "css_framework", "css_link",   r"bootstrap(?:\.min)?\.css"),
    ("Tailwind CSS", "css_framework", "css_class",  r"class=\"[^\"]*(?:flex|grid|px-|py-|text-|bg-)[^\"]*\""),
    ("Foundation",   "css_framework", "css_link",   r"foundation(?:\.min)?\.css"),
    # Analytics / Marketing
    ("Google Analytics", "analytics",  "script_src", r"google-analytics\.com|googletagmanager\.com"),
    ("Google Tag Manager", "analytics", "script_src", r"googletagmanager\.com"),
    ("Segment",      "analytics",  "script_src",   r"segment\.io|segment\.com"),
    ("Mixpanel",     "analytics",  "script_src",   r"mixpanel\.com"),
    ("Hotjar",       "analytics",  "script_src",   r"hotjar\.com"),
    ("Amplitude",    "analytics",  "script_src",   r"amplitude\.com"),
    ("PostHog",      "analytics",  "script_src",   r"posthog\.com|posthog\.js"),
    ("Intercom",     "crm",        "script_src",   r"intercom\.io|intercom-cdn"),
    ("HubSpot",      "crm",        "script_src",   r"hs-scripts\.com|hubspot\.com"),
    ("Zendesk",      "crm",        "script_src",   r"zendesk\.com|zdassets\.com"),
    # CDN / Infrastructure
    ("Cloudflare",   "cdn",        "header",        r"cf-ray|cloudflare"),
    ("Fastly",       "cdn",        "header",        r"x-served-by.*cache"),
    ("Amazon CloudFront", "cdn",   "header",        r"x-amz-cf-id|cloudfront"),
    ("Vercel",       "hosting",    "header",        r"x-vercel-id"),
    ("Netlify",      "hosting",    "header",        r"x-nf-request-id"),
    ("GitHub Pages", "hosting",    "header",        r"x-github-request-id"),
    ("AWS",          "hosting",    "header",        r"x-amzn-requestid|x-amz-request-id"),
    # Server / Runtime
    ("Nginx",        "server",     "header",        r"nginx"),
    ("Apache",       "server",     "header",        r"apache"),
    ("Caddy",        "server",     "header",        r"caddy"),
    ("Node.js",      "runtime",    "header",        r"node\.js"),
    ("PHP",          "runtime",    "header",        r"php"),
    ("Python",       "runtime",    "header",        r"python|uvicorn|gunicorn|django|flask|fastapi"),
    ("Ruby on Rails","runtime",    "header",        r"phusion|puma|thin"),
    # Libraries
    ("jQuery",       "library",    "script_src",   r"jquery(?:\.min)?\.js"),
    ("lodash",       "library",    "script_src",   r"lodash(?:\.min)?\.js"),
    ("moment.js",    "library",    "script_src",   r"moment(?:\.min)?\.js"),
    ("D3.js",        "library",    "script_src",   r"d3(?:\.v\d+)?(?:\.min)?\.js"),
    # Fonts
    ("Google Fonts", "font",       "css_link",     r"fonts\.googleapis\.com"),
    ("Adobe Fonts",  "font",       "script_src",   r"use\.typekit\.net|adobe\.fonts"),
]


def _detect_tech_stack(
    html: str,
    resp_headers: Dict[str, str],
) -> Dict[str, Any]:
    """
    Detect technology stack from HTML content and response headers.
    Returns categorized findings with evidence strings.
    """
    if not BS4_AVAILABLE:
        # Fallback: header-only detection
        found: Dict[str, Dict] = {}
        headers_lower = {k.lower(): v.lower() for k, v in resp_headers.items()}
        server = headers_lower.get("server", "")
        powered_by = headers_lower.get("x-powered-by", "")
        generator = resp_headers.get("x-generator", resp_headers.get("X-Generator", ""))
        for name, category, sig_type, pattern in _TECH_FINGERPRINTS:
            if sig_type != "header":
                continue
            combined_hdrs = " ".join([server, powered_by, generator])
            if re.search(pattern, combined_hdrs, re.I):
                found[name] = {"name": name, "category": category, "evidence": [sig_type]}
        return {
            "technologies": list(found.values()),
            "tech_count": len(found),
            "categories": _group_by_category(list(found.values())),
        }

    soup = BeautifulSoup(html, "html.parser")
    headers_str = " ".join(f"{k}: {v}" for k, v in resp_headers.items()).lower()

    # Build signal strings for each type
    script_srcs = " ".join(
        (s.get("src") or "") for s in soup.find_all("script", src=True)
    ).lower()
    css_links = " ".join(
        (l.get("href") or "") for l in soup.find_all("link", rel=True)
        if "stylesheet" in " ".join(l.get("rel", []))
    ).lower()
    html_attrs = html  # check full HTML for attribute patterns
    html_comments = " ".join(str(c) for c in soup.find_all(string=lambda t: isinstance(t, str) and "<!--" in str(t)))

    meta_gen_el = soup.find("meta", attrs={"name": re.compile(r"generator", re.I)})
    meta_gen = (meta_gen_el.get("content", "") if meta_gen_el else "").lower()

    # Cookies from Set-Cookie headers
    cookie_header = " ".join(
        v for k, v in resp_headers.items() if k.lower() == "set-cookie"
    ).lower()

    signals = {
        "header":       headers_str,
        "script_src":   script_srcs,
        "css_link":     css_links,
        "html_attr":    html_attrs,
        "html_comment": html_comments,
        "meta_generator": meta_gen,
        "css_class":    html_attrs,
        "cookie":       cookie_header,
        "json_key":     html,
    }

    found: Dict[str, Dict] = {}
    for name, category, sig_type, pattern in _TECH_FINGERPRINTS:
        signal_text = signals.get(sig_type, "")
        if signal_text and re.search(pattern, signal_text, re.I):
            if name not in found:
                found[name] = {"name": name, "category": category, "evidence": [sig_type]}
            else:
                if sig_type not in found[name]["evidence"]:
                    found[name]["evidence"].append(sig_type)

    tech_list = list(found.values())
    return {
        "technologies": tech_list,
        "tech_count": len(tech_list),
        "categories": _group_by_category(tech_list),
        "meta_generator": meta_gen or None,
        "server_header": resp_headers.get("server", resp_headers.get("Server")),
        "powered_by": resp_headers.get("x-powered-by", resp_headers.get("X-Powered-By")),
    }


def _group_by_category(tech_list: List[Dict]) -> Dict[str, List[str]]:
    """Group technology names by category."""
    groups: Dict[str, List[str]] = {}
    for tech in tech_list:
        cat = tech.get("category", "other")
        groups.setdefault(cat, []).append(tech["name"])
    return groups


class TechRequest(BaseModel):
    url: str = Field(..., description="URL to analyse")
    timeout: int = Field(DEFAULT_TIMEOUT, ge=1, le=MAX_TIMEOUT)
    render_js: bool = Field(
        False,
        description="Use headless browser for JS-rendered content detection (slower but more thorough for SPAs)",
    )


@app.post("/api/tech")
async def detect_tech(req: TechRequest):
    """
    Detect the technology stack of a website.

    **v1.1.0** — Returns a categorised list of detected technologies with evidence.

    Detects 60+ technologies across:
    - **CMS**: WordPress, Ghost, Drupal, Joomla, Shopify, WooCommerce, Magento
    - **Website builders**: Squarespace, Wix, Webflow
    - **JS frameworks**: React, Next.js, Vue.js, Angular, Nuxt.js, Svelte, Ember
    - **CSS frameworks**: Bootstrap, Tailwind CSS, Foundation
    - **Analytics**: Google Analytics/GTM, Segment, Mixpanel, Hotjar, Amplitude, PostHog
    - **CRM/Support**: Intercom, HubSpot, Zendesk
    - **CDN/Hosting**: Cloudflare, Fastly, CloudFront, Vercel, Netlify, GitHub Pages, AWS
    - **Server/Runtime**: Nginx, Apache, Node.js, PHP, Python, Ruby on Rails
    - **Libraries**: jQuery, lodash, D3.js
    - **Fonts**: Google Fonts, Adobe Fonts

    Detection uses HTTP headers, `<script src>`, `<link href>`, HTML attributes,
    `<meta name="generator">`, cookies, and comment patterns.

    **Response:**
    ```json
    {
      "url": "https://example.com",
      "technologies": [
        {"name": "WordPress", "category": "cms", "evidence": ["script_src", "meta_generator"]},
        {"name": "jQuery",    "category": "library", "evidence": ["script_src"]},
        {"name": "Cloudflare","category": "cdn", "evidence": ["header"]}
      ],
      "tech_count": 3,
      "categories": {"cms": ["WordPress"], "library": ["jQuery"], "cdn": ["Cloudflare"]},
      "meta_generator": "WordPress 6.4.3",
      "server_header": "nginx/1.18.0"
    }
    ```
    """
    if not HTTPX_AVAILABLE:
        raise HTTPException(503, "httpx not available")

    try:
        if req.render_js and PLAYWRIGHT_AVAILABLE:
            status, html, elapsed_ms = await fetch_url_playwright(req.url, req.timeout)
            resp_headers: Dict[str, str] = {}
        else:
            status, html, resp_headers, elapsed_ms = await fetch_url_httpx(req.url, req.timeout)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch {req.url}: {e}")

    if status >= 400:
        raise HTTPException(status, f"HTTP {status} from {req.url}")

    result = _detect_tech_stack(html, resp_headers)
    result["url"] = req.url
    result["status_code"] = status
    result["fetch_time_ms"] = round(elapsed_ms)

    return result


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
