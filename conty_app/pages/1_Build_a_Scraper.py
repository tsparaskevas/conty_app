from __future__ import annotations

# --- Make project root importable -------------------------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (../.. from pages/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Standard library imports -----------------------------------------------
import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
import os

# --- Third-party imports ----------------------------------------------------
import yaml
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
import pandas as pd

# --- conty_core imports ---
from conty_core.extract import extract_article
from conty_core.utils import to_text, preview_value, absolutize_url
from conty_core.postprocess import postprocess_article_dict
from conty_core.fetcher import FetchResult, fetch_with_fallback, close_shared_driver
from conty_core.consent import KNOWN_CONSENT_XPATHS

# --- Streamlit page config -------------
st.set_page_config(page_title="Build/Edit a Scraper (conty)", layout="wide")

# --- UI CSS tweaks ---
st.markdown(
    """
    <style>
      textarea{
        resize: vertical !important;
      }
      .conty-edit-banner{
        margin:10px 0 0 0;
        padding:10px 12px;
        border:1px solid #e5e7eb;
        border-radius:10px;
        background:#fafafa;
        color:#0f172a;
      }
      .conty-edit-banner .label{
        font-weight:700;
        font-size:1.10rem;
      }
      .conty-edit-banner .name{
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        font-weight:700;
        font-size:1.15rem;
        padding:0 .25rem;
      }
      .conty-edit-banner .hint{
        opacity:.7;
        font-size:.95rem;
      }
      @media (prefers-color-scheme: dark){
        .conty-edit-banner{
          background:#0b1220;
          border-color:#334155;
          color:#e5e7eb;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Paths & directories -----------------------------------------------------
DATA_DIR      = ROOT / "data"
SCRAPERS_DIR  = DATA_DIR / "scrapers"
INPUTS_DIR    = DATA_DIR / "inputs"
OUTPUTS_DIR   = DATA_DIR / "outputs"
RUN_LOGS_DIR  = OUTPUTS_DIR / "run_logs"  
LOGS_DIR      = DATA_DIR / "logs"
TMP_DIR       = DATA_DIR / "tmp"

SCRAPERS_DIR.mkdir(parents=True, exist_ok=True)
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Demo mode ---
DEMO_DIR        = DATA_DIR / "demo"
DEMO_HTML_DIR   = DEMO_DIR / "html"
DEMO_TEASERS_DIR = DEMO_DIR / "teasers"
DEMO_OUTPUTS_DIR = DEMO_DIR / "outputs"

# --- Demo mode flag --------------------------------------
DEMO_MODE = bool(os.environ.get("CONTY_DEMO", "").strip())

def show_demo_banner():
    if DEMO_MODE:
        st.info(
            "🔒 **Demo mode** is ON. This instance runs in read-only mode with "
            "demo teaser CSVs and HTML snapshots under `data/demo/`."
        )

# --- Defaults for body container / exclusions / boilerplate ---
DEFAULT_CONTAINER_EXCLUDES = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    ".ad",
    ".ads",
    ".advert",
    ".advertisement",
    ".social",
    ".share",
    ".related",
    ".newsletter",
]

DEFAULT_CONTAINER_BOILERPLATE = [
    r"^Διαβάστε επίσης:.*",
    r"^Διαβάστε ακόμη:.*",
    r"^Διαβάστε ακόμα:.*",
    r"^Δείτε επίσης:.*",
    r"^Δείτε ακόμη:.*",
    r"^Δείτε ακόμα:.*",
]

# --- A minimal generic template skeleton (used for new templates) -----------
DEFAULT_GENERIC_TEMPLATE: Dict[str, Any] = {
    "name": "generic",
    "priority": 10,
    "enabled": True,
    "url_regex": "",
    "fetch": {
        "requires_js": False,
        "wait_css": "",
        "wait_timeout": 5,
        "consent": {
            "enabled": True,
            "xpaths": [],
        },
    },
    "extract": {
        "body_config": {
            "main_container_css": "",
            "body_container_css": "",
        },
        "container_text": {
            "exclude": list(DEFAULT_CONTAINER_EXCLUDES),
            "boilerplate_regex": list(DEFAULT_CONTAINER_BOILERPLATE),
        },
        "accept": {
            "min_body_chars": 400,
        },
    },
}

# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def _st_rerun():
    """Safe wrapper around st.rerun() so old calls don't crash."""
    try:
        st.rerun()
    except Exception:
        pass


def _g(key: str, default: Any = "") -> Any:
    """Shorthand for st.session_state.get with a default."""
    return st.session_state.get(key, default)


def _editing_banner() -> None:
    """Show 'currently editing' banner for the active template."""
    name = st.session_state.get("editing_template_name") or ""
    if not name:
        return
    st.markdown(
        f"""
        <div class="conty-edit-banner">
          <div class="label">Currently editing template:</div>
          <div class="name">{name}</div>
          <div class="hint">
            Any changes you make below apply to this template.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


class ui_card:
    """
    Simple context manager for small bordered blocks:

    with ui_card("Title", "optional help"):
        ...
    """

    def __init__(self, title: str, help_text: str = ""):
        self.title = title
        self.help_text = help_text

    def __enter__(self):
        st.markdown(
            """
            <div style="
              border:1px solid #e5e7eb;
              border-radius:10px;
              padding:10px 12px;
              margin:6px 0;
            ">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{self.title}**")
        if self.help_text:
            st.caption(self.help_text)

    def __exit__(self, exc_type, exc, tb):
        st.markdown("</div>", unsafe_allow_html=True)


def _site_key_from_url(url: str) -> str:
    """Return a normalized 'site key' from URL hostname (e.g. news.example.com → example.com)."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return "unknown"
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host or "unknown"


def _looks_like_image(url: str) -> bool:
    """Quick heuristic to detect image URLs."""
    url = url.lower()
    return any(url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"])


# --- Article object to plain dict ---

def _art_to_dict(art: Any) -> Dict[str, Any]:
    """
    Normalize extractor output (Article, Pydantic, dataclass, dict) into a plain dict.
    """
    if isinstance(art, dict):
        return art
    # Pydantic v2
    try:
        return art.model_dump()  # type: ignore[attr-defined]
    except Exception:
        pass
    # Pydantic v1
    try:
        return art.dict()  # type: ignore[attr-defined]
    except Exception:
        pass
    # Dataclass / generic object
    try:
        return {
            k: getattr(art, k)
            for k in dir(art)
            if not k.startswith("_") and not callable(getattr(art, k))
        }
    except Exception:
        return getattr(art, "__dict__", {}) or {}


# --- Matrix test helpers ------------------------------------------------------
def _url_section_at(u: str, seg_idx: int = 1) -> str:
    """
    Return the Nth path segment (1-based).
    E.g. /news/ellada -> seg_idx=1:'news', seg_idx=2:'ellada'.
    """
    try:
        p = urlparse(u)
        parts = [x for x in (p.path or "/").split("/") if x]
        i = max(1, int(seg_idx or 1))
        return parts[i - 1].lower() if len(parts) >= i else ""
    except Exception:
        return ""

def _url_section(u: str) -> str:
    """Back-compat: first segment = section."""
    return _url_section_at(u, 1)


def _sample_by_section(
    urls: list[str],
    per_section: int = 5,
    seed: int = 0,
    seg_idx: int = 1,
) -> list[str]:
    """
    Bucket URLs by the Nth path segment and take up to per_section from each
    bucket (shuffled within each bucket).
    """
    random.seed(seed or 0)
    buckets: dict[str, list[str]] = {}
    for u in urls:
        s = _url_section_at(u, seg_idx)
        buckets.setdefault(s, []).append(u)

    out: list[str] = []
    for s, arr in buckets.items():
        random.shuffle(arr)
        out.extend(arr[:per_section])
    return out

# ---------------------------------------------------------------------------
# Selector helpers (used by the form + template builder)
# ---------------------------------------------------------------------------

def _split_sel(token: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse a selector token like:
      'a::text'          → ('a', 'text', None)
      'img::attr(src)'   → ('img', 'attr', 'src')
      '.foo .bar'        → ('.foo .bar', 'text', None)
    """
    if not isinstance(token, str):
        return "", "text", None
    if "::attr(" in token:
        m = re.search(r"::attr\(([^)]+)\)", token)
        return token.split("::attr(")[0], "attr", (m.group(1) if m else None)
    if token.endswith("::text"):
        return token[:-6], "text", None
    return token, "text", None

def _rule_to_css_attr(rule: Any) -> Tuple[str, Optional[str]]:
    """
    Accepts rule in any of these forms and returns (css, attr):

      - "css::attr(foo)"
      - {"any": ["css::attr(foo)", "other::text", ...]}
      - {"join_texts": {"selector": "css::attr(foo)", "sep": ", "}}

    Used both to prefill form fields from YAML and by the Matrix extractor.
    """
    # Simple string rule
    if isinstance(rule, str):
        css, mode, attr = _split_sel(rule)
        return css, (attr if mode == "attr" else None)

    # Dict-based legacy rules
    if isinstance(rule, dict):
        # 1) legacy {"any": [...]} rule
        any_val = rule.get("any")
        if isinstance(any_val, list) and any_val:
            first = any_val[0]
            if isinstance(first, str):
                css, mode, attr = _split_sel(first)
                return css, (attr if mode == "attr" else None)

        # 2) legacy {"join_texts": {"selector": "...", "sep": ", "}} rule
        jt = rule.get("join_texts")
        if isinstance(jt, dict):
            sel = jt.get("selector", "") or ""
            if sel:
                css, mode, attr = _split_sel(sel)
                return css, (attr if mode == "attr" else None)

    # Fallback: nothing usable
    return "", None


def _prefill_join_opts(rule: Any, default_sep: str) -> Tuple[str, str]:
    """
    For rules saved as
        {'join_texts': {'selector': '...', 'sep': ', '}}
    return (selector, sep) for the form.
    """
    if isinstance(rule, dict) and "join_texts" in rule:
        cfg = rule.get("join_texts") or {}
        return cfg.get("selector", ""), cfg.get("sep", default_sep)
    if isinstance(rule, str):
        css, _, _ = _split_sel(rule)
        return css, default_sep
    return "", default_sep


def _ensure_list(v: Any) -> List[str]:
    """Coerce a value to a list[str] (used for templates' url_patterns, etc.)."""
    if not v:
        return []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    if isinstance(v, str):
        return [v]
    return []

def _rule_first_css(rule: Any) -> str:
    """
    Convert an extract rule into a single CSS token for lightweight extraction:

      - "h1::text" -> "h1::text"
      - {"any": ["a::text", "b::text"]} -> first non-empty string
      - {"join_texts": {"selector": ".tag", "sep": ", "}} -> ".tag"
    """
    if isinstance(rule, str):
        return rule

    if isinstance(rule, dict):
        # legacy: { any: ['css1', 'css2', ...] }
        any_val = rule.get("any")
        if isinstance(any_val, list):
            for v in any_val:
                if isinstance(v, str) and v.strip():
                    return v

        # legacy: { join_texts: { selector: '...', sep: '...' } }
        jt = rule.get("join_texts")
        if isinstance(jt, dict):
            sel = jt.get("selector")
            if isinstance(sel, str) and sel.strip():
                return sel

    return ""

# ---------------------------------------------------------------------------
# YAML I/O for site configs
# ---------------------------------------------------------------------------

def _load_site_yaml(site_key: str) -> Dict[str, Any]:
    """
    Load <site>.yml from SCRAPERS_DIR.
    Returns at least: {'site': ..., 'templates': []}
    """
    ypath = SCRAPERS_DIR / f"{site_key}.yml"
    if not ypath.exists():
        return {"site": site_key, "templates": []}
    try:
        with ypath.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {"site": site_key, "templates": []}
    except Exception:
        return {"site": site_key, "templates": []}


def _save_site_yaml(site_key: str, cfg: Dict[str, Any]) -> None:
    """Persist site config YAML back to disk.

    In demo mode **skip** writing and just show a notice so that the UI feels normal but the filesystem stays read-only.
    """
    if DEMO_MODE:
        st.info(
            "💾 Saving of scrapers/templates is disabled in demo mode.\n\n"
            f"(Pretending to save `{site_key}.yml` only in memory.)"
        )
        return

    ypath = SCRAPERS_DIR / f"{site_key}.yml"
    with ypath.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _upsert_template(cfg: Dict[str, Any], new_tpl: Dict[str, Any]) -> None:
    """
    Insert or replace a template (by name) into site config.
    """
    name = (new_tpl.get("name") or "").strip()
    if not name:
        return
    arr = cfg.setdefault("templates", [])
    for i, t in enumerate(arr):
        if t.get("name") == name:
            arr[i] = new_tpl
            break
    else:
        arr.append(new_tpl)

# ---------------------------------------------------------------------------
# Session defaults & small helpers
# ---------------------------------------------------------------------------

def _load_template_into_form_state(tpl: Dict[str, Any]) -> None:
    """
    Populate Streamlit form fields from a template dict.
    ONLY handles selector-related fields (no tpl_* meta fields).
    """
    extract = tpl.get("extract") or {}
    body_cfg = extract.get("body_config") or {}
    ct_cfg = extract.get("container_text") or {}
    accept_cfg = extract.get("accept") or {}

    # Containers
    st.session_state["fld_main_container_css"] = body_cfg.get("main_container_css", "") or ""
    st.session_state["fld_body_container_css"] = body_cfg.get("body_container_css", "") or ""

    # Simple one-field rules (we only keep CSS selector part)
    for state_key, tpl_key in [
        ("fld_title_css", "title"),
        ("fld_subtitle_css", "subtitle"),
        ("fld_lead_css", "lead"),
        ("fld_section_css", "section"),
        ("fld_author_css", "author"),
        ("fld_tags_css", "tags"),
        ("fld_published_css", "published_time"),
        ("fld_updated_css", "updated_time"),
        ("fld_main_image_css", "main_image"),
    ]:
        rule = extract.get(tpl_key)
        css, _attr = _rule_to_css_attr(rule)
        st.session_state[state_key] = css or ""

    # Body clean-up and acceptance
    st.session_state["fld_body_exclude"] = "\n".join(ct_cfg.get("exclude") or [])
    st.session_state["fld_body_boilerplate"] = "\n".join(ct_cfg.get("boilerplate_regex") or [])
    st.session_state["fld_min_body_chars"] = int(accept_cfg.get("min_body_chars", 400) or 400)

def _build_extract_from_form(current_tpl: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the `extract` dict for a template from the selector/body widgets.
    Starts from the existing `extract` block so unknown keys are preserved.
    """
    extract = (current_tpl.get("extract") or {}).copy()

    body_cfg = (extract.get("body_config") or {}).copy()
    ct_cfg = (extract.get("container_text") or {}).copy()
    accept_cfg = (extract.get("accept") or {}).copy()

    # Containers
    body_cfg["main_container_css"] = (st.session_state.get("fld_main_container_css") or "").strip()
    body_cfg["body_container_css"] = (st.session_state.get("fld_body_container_css") or "").strip()

    def _rule_from_css(state_key: str):
        css = (st.session_state.get(state_key) or "").strip()
        return css or None

    # Simple text-based fields
    extract["title"] = _rule_from_css("fld_title_css")
    extract["subtitle"] = _rule_from_css("fld_subtitle_css")
    extract["lead"] = _rule_from_css("fld_lead_css")
    extract["section"] = _rule_from_css("fld_section_css")
    extract["author"] = _rule_from_css("fld_author_css")
    extract["tags"] = _rule_from_css("fld_tags_css")
    extract["published_time"] = _rule_from_css("fld_published_css")
    extract["updated_time"] = _rule_from_css("fld_updated_css")
    extract["main_image"] = _rule_from_css("fld_main_image_css")

    # Container text clean-up
    exclude_lines = [
        ln.strip()
        for ln in (st.session_state.get("fld_body_exclude") or "").splitlines()
        if ln.strip()
    ]
    boilerplate_lines = [
        ln.strip()
        for ln in (st.session_state.get("fld_body_boilerplate") or "").splitlines()
        if ln.strip()
    ]
    ct_cfg["exclude"] = exclude_lines
    ct_cfg["boilerplate_regex"] = boilerplate_lines

    # Acceptance
    try:
        min_chars = int(st.session_state.get("fld_min_body_chars", 400) or 400)
    except Exception:
        min_chars = 400
    accept_cfg["min_body_chars"] = min_chars

    extract["body_config"] = body_cfg
    extract["container_text"] = ct_cfg
    extract["accept"] = accept_cfg

    return extract

def _preview_field(html: str, container_css: str, field_css: str, label: str) -> None:
    """
    Show a more prominent preview for a CSS selector inside an optional container.
    """
    field_css = (field_css or "").strip()
    if not field_css:
        st.markdown(f"🧪 **{label}** — _no selector set_")
        return

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    root = soup
    container_css = (container_css or "").strip()
    if container_css:
        node = soup.select_one(container_css)
        if node:
            root = node

    try:
        nodes = root.select(field_css)
    except Exception as e:
        st.markdown(f"🧪 **{label}** — selector error:")
        st.code(str(e))
        return

    count = len(nodes)
    if count == 0:
        st.markdown(f"🧪 **{label}** — 0 matches.")
        return

    texts = []
    for n in nodes[:3]:
        txt = n.get_text(" ", strip=True)
        if txt:
            texts.append(txt)

    st.markdown(f"🧪 **{label}** — {count} match(es)")

    if texts:
        preview = " | ".join(texts)
        if len(preview) > 400:
            preview = preview[:400] + "…"
        # Show text in a code-style block so it stands out
        st.code(preview)
    else:
        st.markdown("_Matches have no text content._")

def _preview_image_field(html: str, container_css: str, field_css: str, label: str) -> None:
    """
    Preview for image selectors: show image URL(s) and the first actual image.
    Supports plain CSS or CSS with ::attr(src).
    """
    field_css = (field_css or "").strip()
    if not field_css:
        st.markdown(f"🧪 **{label}** — _no selector set_")
        return

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    root = soup
    container_css = (container_css or "").strip()
    if container_css:
        node = soup.select_one(container_css)
        if node:
            root = node

    css, mode, attr = _split_sel(field_css)
    css = css.strip() or field_css

    try:
        nodes = root.select(css)
    except Exception as e:
        st.markdown(f"🧪 **{label}** — selector error:")
        st.code(str(e))
        return

    if not nodes:
        st.markdown(f"🧪 **{label}** — 0 matches.")
        return

    # Collect candidate URLs
    urls: List[str] = []
    if mode == "attr" and attr:
        for n in nodes:
            v = n.get(attr)
            if isinstance(v, str) and v.strip():
                v = v.strip()
                # Special handling for srcset: "url1 320w, url2 640w, ..."
                if attr.lower() == "srcset":
                    first_part = v.split(",", 1)[0].strip()
                    # usually "url 320w" → take the first token as URL
                    first_url = first_part.split()[0]
                    urls.append(first_url)
                else:
                    urls.append(v)
    else:
        for n in nodes:
            # direct <img>
            if n.name == "img":
                src = n.get("src") or n.get("data-src") or n.get("data-lazy-src")
                if src:
                    urls.append(src.strip())
                    continue
            # or first <img> inside
            img = n.find("img")
            if img:
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                if src:
                    urls.append(src.strip())

    if not urls:
        st.markdown(f"🧪 **{label}** — matches found, but no image URLs.")
        return

    # Make URLs absolute using final_url as base (if known)
    base_url = st.session_state.get("final_url") or st.session_state.get("test_url") or ""
    abs_urls = [absolutize_url(u, base_url) for u in urls]

    # Consider only real remote URLs (or data URIs) for st.image
    def _looks_like_remote(u: str) -> bool:
        u = (u or "").strip().lower()
        return u.startswith("http://") or u.startswith("https://") or u.startswith("data:")

    first_remote = next((u for u in abs_urls if _looks_like_remote(u)), None)

    st.markdown(f"🧪 **{label}** — {len(abs_urls)} image URL(s) found. First URL:")
    st.code(first_remote or abs_urls[0])

    # Only try to render if it looks like a real remote URL
    if first_remote and _looks_like_remote(first_remote):
        try:
            st.image(first_remote, caption=f"{label} preview", width='stretch')
        except Exception:
            st.markdown("_Could not render image; the URL above may still be correct._")
    else:
        st.markdown(
            "_Could not render image preview (URL is not an absolute web URL). "
            "The URL above may still be correct on the site._"
        )

def _preview_body_text(html: str, main_css: str, body_css: str, label: str = "Body text") -> None:
    """
    Preview body text based on main/body container + exclude selectors + boilerplate regex.
    Shows length and a snippet.
    """
    main_css = (main_css or "").strip()
    body_css = (body_css or "").strip()

    if not main_css and not body_css:
        st.markdown(f"🧪 **{label}** — _set main/body containers first_")
        return

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    root = soup
    if main_css:
        node = soup.select_one(main_css)
        if node:
            root = node

    body_node = root
    if body_css:
        node = root.select_one(body_css)
        if node:
            body_node = node

    if body_node is None:
        st.markdown(f"🧪 **{label}** — container not found.")
        return

    # Apply exclude selectors from form
    exclude_lines = [
        ln.strip()
        for ln in (st.session_state.get("fld_body_exclude") or "").splitlines()
        if ln.strip()
    ]
    for sel in exclude_lines:
        try:
            for n in body_node.select(sel):
                n.decompose()
        except Exception:
            # Ignore bad selectors here; they will be obvious in the preview
            pass

    # Strip script/style
    for bad in body_node.find_all(["script", "style"]):
        bad.decompose()

    text = body_node.get_text(" ", strip=True)
    length = len(text)

    if not text:
        st.markdown(f"🧪 **{label}** — 0 characters after cleanup.")
        return

    snippet = text if length <= 1200 else text[:1200] + "…"

    st.markdown(f"🧪 **{label}** — {length} characters")
    st.text_area(f"{label} preview", snippet, height=220)

# Helpers to capture the extraction previews
def _first_text_for_selector(root, css: str) -> str:
    css = (css or "").strip()
    if not css:
        return ""
    try:
        nodes = root.select(css)
    except Exception:
        return ""
    for n in nodes:
        txt = n.get_text(" ", strip=True)
        if txt:
            return txt
    return ""


def _all_texts_for_selector(root, css: str, max_items: int = 20) -> List[str]:
    css = (css or "").strip()
    if not css:
        return []
    try:
        nodes = root.select(css)
    except Exception:
        return []
    out: List[str] = []
    for n in nodes[:max_items]:
        txt = n.get_text(" ", strip=True)
        if txt:
            out.append(txt)
    return out


def _first_image_url_from_selector(root, field_css: str, base_url: str) -> str:
    """
    Try to resolve a main image URL from a selector.
    Supports either CSS or CSS::attr(src).
    """
    field_css = (field_css or "").strip()
    if not field_css:
        return ""

    css, mode, attr = _split_sel(field_css)
    css = css.strip() or field_css

    try:
        nodes = root.select(css)
    except Exception:
        return ""

    if not nodes:
        return ""

    urls: List[str] = []
    if mode == "attr" and attr:
        for n in nodes:
            v = n.get(attr)
            if isinstance(v, str) and v.strip():
                v = v.strip()
                if attr.lower() == "srcset":
                    first_part = v.split(",", 1)[0].strip()
                    first_url = first_part.split()[0]
                    urls.append(first_url)
                else:
                    urls.append(v)
    else:
        for n in nodes:
            # direct <img>
            if n.name == "img":
                src = n.get("src") or n.get("data-src") or n.get("data-lazy-src")
                if src:
                    urls.append(src.strip())
                    continue
            # or first <img> inside
            img = n.find("img")
            if img:
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                if src:
                    urls.append(src.strip())

    if not urls:
        return ""

    base = base_url or ""
    return absolutize_url(urls[0], base) if base else urls[0]


def _extract_preview_from_form(html: str, url: str) -> Dict[str, Any]:
    """
    Run a lightweight extraction using the CURRENT selector fields (not necessarily saved),
    and return a dict of extracted article fields suitable for YAML preview.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Containers
    main_css = (st.session_state.get("fld_main_container_css") or "").strip()
    body_css = (st.session_state.get("fld_body_container_css") or "").strip()

    root = soup
    if main_css:
        node = soup.select_one(main_css)
        if node:
            root = node

    body_node = root
    if body_css:
        node = root.select_one(body_css)
        if node:
            body_node = node

    # Body cleanup: excludes
    body_node = body_node or root
    exclude_lines = [
        ln.strip()
        for ln in (st.session_state.get("fld_body_exclude") or "").splitlines()
        if ln.strip()
    ]
    for sel in exclude_lines:
        try:
            for n in body_node.select(sel):
                n.decompose()
        except Exception:
            # ignore bad selectors
            pass

    # Strip script/style
    for bad in body_node.find_all(["script", "style"]):
        bad.decompose()

    # Body text
    body_text = body_node.get_text(" ", strip=True)
    # Boilerplate regex cleanup
    boiler_lines = [
        ln.strip()
        for ln in (st.session_state.get("fld_body_boilerplate") or "").splitlines()
        if ln.strip()
    ]
    for pattern in boiler_lines:
        try:
            rx = re.compile(pattern, flags=re.MULTILINE | re.UNICODE)
            body_text = rx.sub("", body_text)
        except Exception:
            # If regex is invalid, just skip it
            continue

    body_text = re.sub(r"\s+", " ", body_text).strip()
    body_len = len(body_text)

    try:
        min_chars = int(st.session_state.get("fld_min_body_chars", 400) or 400)
    except Exception:
        min_chars = 400

    # Simple fields (first matching text)
    title = _first_text_for_selector(root, st.session_state.get("fld_title_css", ""))
    subtitle = _first_text_for_selector(root, st.session_state.get("fld_subtitle_css", ""))
    lead = _first_text_for_selector(root, st.session_state.get("fld_lead_css", ""))
    section = _first_text_for_selector(root, st.session_state.get("fld_section_css", ""))
    author = _first_text_for_selector(root, st.session_state.get("fld_author_css", ""))
    published_raw = _first_text_for_selector(root, st.session_state.get("fld_published_css", ""))
    updated_raw = _first_text_for_selector(root, st.session_state.get("fld_updated_css", ""))

    # Tags: list of texts
    tags = _all_texts_for_selector(root, st.session_state.get("fld_tags_css", ""))

    # Main image
    base_url = url or st.session_state.get("final_url") or st.session_state.get("test_url") or ""
    main_image = _first_image_url_from_selector(
        root,
        st.session_state.get("fld_main_image_css", ""),
        base_url,
    )

    # ---- Postprocess with ISO dates for preview YAML -----------------------------
    site_key = st.session_state.get("site_key") or None

    preview_article = {
        "url": url or st.session_state.get("final_url") or st.session_state.get("test_url") or "",
        "title": title,
        "subtitle": subtitle,
        "lead": lead,
        "section": section,
        "author": author,
        "tags": tags,
        "published_time": published_raw,
        "updated_time": updated_raw,
        "body": body_text,
    }

    preview_clean = postprocess_article_dict(
        preview_article,
        html_for_fallback=str(root),
        site_hint=site_key,
        url_for_log=preview_article["url"],
    )

    return {
        "url": preview_clean.get("url", preview_article["url"]),
        "engine": st.session_state.get("engine") or "",
        "title": preview_clean.get("title", title),
        "subtitle": preview_clean.get("subtitle", subtitle),
        "lead": preview_clean.get("lead", lead),
        "section": preview_clean.get("section", section),
        "author": preview_clean.get("author", author),
        "tags": preview_clean.get("tags", tags),
        "published_time_raw": published_raw,
        "updated_time_raw": updated_raw,
        "published_time": preview_clean.get("published_time"),
        "updated_time": preview_clean.get("updated_time"),
        "main_image": main_image,
        "body_chars": body_len,
        "min_body_chars": min_chars,
        "accepted_full": bool(body_len >= min_chars),
        "body": preview_clean.get("body", body_text),
    }

# Basic session defaults used across the page
for k, v in {
    "editor_rev": 0,             # this forces widget keys to change when we reload a template
    "editing_template_name": "",
    "editing_tpl_loaded": False,
    "site_key": "",
    "site_cfg": {},
    "final_url": "",
    "html_cache": "",
    "engine": "",
}.items():
    st.session_state.setdefault(k, v)

# URL currently under test (used in multiple sections later)
st.session_state.setdefault("test_url", "")

# Remember CSV folder for teasers (used later in B/Matrix)
if DEMO_MODE:
    st.session_state.setdefault("csv_folder_dir_persist", str(DEMO_TEASERS_DIR))
    st.session_state.setdefault("csv_folder_dir_input", str(DEMO_TEASERS_DIR))
else:
    st.session_state.setdefault("csv_folder_dir_persist", str(INPUTS_DIR))
    st.session_state.setdefault("csv_folder_dir_input", str(INPUTS_DIR))

# For “new site” bootstrap logic (only once per site)
st.session_state.setdefault("_bootstrapped_default_tpl_for_site", {})
st.session_state.setdefault("last_new_site_key", "")

# If another page (like View Data) staged a URL for test/edit, promote it early
if "__next_test_url" in st.session_state:
    st.session_state["test_url"] = st.session_state.pop("__next_test_url")

# ---------------------------------------------------------------------------
# Page header & quick guide
# ---------------------------------------------------------------------------

st.title("Build / Edit / Test scrapers")

if DEMO_MODE:
    st.info(
        "🔒 **Demo mode** is ON. Network fetching is disabled. "
        "Use the demo HTML samples and teaser CSVs under `data/demo/` to try the workflow."
    )

with st.expander("How this page works (quick guide)", expanded=False):
    st.markdown(
        """
**Goal:** Build and validate scraper templates that extract articles reliably.

### Choose site
Pick or create a site key (`data/scrapers/<site>.yml`).  
View existing templates and YAML. Configure site-level fetch defaults (engine, waits, consent XPaths).

### Article URL(s)
Paste a single URL for testing, or choose URLs from teasers CSVs.  
Optionally build a section-balanced sample set for Matrix tests.  
You can view and clear the active sample set anytime.

### Fetch page
Fetch HTML using Requests/Selenium (auto-selected based on site/template settings).  
See final URL, engine, size, raw HTML, and a rendered preview for inspecting CSS selectors.

### Templates
Create, select, clone, rename, delete templates.  
Edit template meta: name, enabled, priority, and URL regex used to match article pages.

### Selectors & extraction rules
Set main/body containers, title/subtitle/lead, section, author, tags, dates, main image.  
Configure body cleanup (exclude selectors, boilerplate regex) and minimum body length.  
Each field shows a live preview (text/image/body snippet).

### Preview extraction
Run a temporary extraction using current selector values.  
Shows processed title/metadata, body length and acceptance, main image, and cleaned dates.

### Matrix test
Test the scraper across many URLs.  
See success rates, missing fields, short bodies, and per-URL extraction results.  
Use this to validate that your template generalizes across the site.

**Typical workflow:**  
Pick site → load/create template → paste URL → fetch → tune selectors → preview → matrix test → save YAML.
        """
    )

# ---------------------------------------------------------------------------
# A) Choose site
# ---------------------------------------------------------------------------

st.subheader("A) Choose site")

col_left, col_right = st.columns([2, 3])

with col_left:
#    mode = st.radio("Mode", ["Load existing site", "Create new site"], horizontal=True)
    mode_options = ["Load existing site", "Create new site"]
    if DEMO_MODE:
        mode_options = ["Load existing site"]

    mode = st.radio("Mode", mode_options, horizontal=True)

    if DEMO_MODE and mode == "Create new site":
        st.warning("Creating brand-new sites is disabled in demo mode.")

    if mode == "Load existing site":
        sites = sorted([p.stem for p in SCRAPERS_DIR.glob("*.yml")])
        if sites:
            site_key = st.selectbox(
                "Site key (YAML under data/scrapers)",
                sites,
                index=sites.index(st.session_state.get("site_key")) if st.session_state.get("site_key") in sites else 0,
            )
            if site_key:
                st.session_state["site_key"] = site_key
                st.session_state["site_cfg"] = _load_site_yaml(site_key)
        else:
            st.info("No scraper YAMLs found in data/scrapers yet.")
            site_key = ""
    else:
        new_key = st.text_input(
            "New site key (example: athensvoice)",
            placeholder="example: athensvoice",
            value=st.session_state.get("site_key", ""),
        ).strip()

        site_key = new_key
        if new_key:
            yml_path = SCRAPERS_DIR / f"{new_key}.yml"
            if yml_path.exists():
                st.warning("This site already exists; loading it for editing.")
                st.session_state["site_cfg"] = _load_site_yaml(new_key)
            else:
                # Initialize a fresh, minimal config for this site
                st.session_state["site_cfg"] = {"site": new_key, "templates": []}

            st.session_state["site_key"] = new_key

            # If this is a *brand new* site with no templates on disk, bootstrap a generic template once
            cfg = st.session_state["site_cfg"] or {}
            booted = st.session_state["_bootstrapped_default_tpl_for_site"]
            if not (cfg.get("templates") or []) and not booted.get(new_key):
                cfg["templates"] = [DEFAULT_GENERIC_TEMPLATE]
                st.session_state["site_cfg"] = cfg
                st.session_state["editing_template_name"] = DEFAULT_GENERIC_TEMPLATE.get("name", "generic")
                booted[new_key] = True

with col_right:
    site_key = st.session_state.get("site_key", "")
    site_cfg = st.session_state.get("site_cfg") or {}
    if site_key:
        st.success(f"Active site: **{site_key}**")
        st.caption(f"YAML path: `data/scrapers/{site_key}.yml`")

        with st.expander("Current templates (names only)", expanded=False):
            tpls = [t.get("name", "<unnamed>") for t in site_cfg.get("templates", [])]
            if tpls:
                st.write(", ".join(tpls))
            else:
                st.write("_No templates yet — we’ll create one in section C._")

        # Full YAML preview for this site
        with st.expander("Full site YAML (read-only preview)", expanded=False):
            cfg = st.session_state.get("site_cfg") or {"site": site_key, "templates": []}
            yaml_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
            st.code(yaml_text, language="yaml")

        # Site-level fetch & consent defaults (Requests vs Selenium)
        with st.expander("Fetch defaults (engine / JS / consent / timeouts)", expanded=False):
            # Always read the latest site config from session
            site_cfg = st.session_state.get("site_cfg") or {}

            # Current fetch config
            fetch_cfg = (site_cfg.get("fetch") or {}).copy()
            consent_cfg = (fetch_cfg.get("consent") or {}).copy()

            # ---- Fetch defaults (engine / wait) ----
            requires_js_val = bool(fetch_cfg.get("requires_js", False))
            wait_css_val = fetch_cfg.get("wait_css") or ""
            try:
                wait_timeout_val = float(fetch_cfg.get("wait_timeout") or 20)
            except Exception:
                wait_timeout_val = 20.0

            st.caption(
                "These defaults are used to decide Requests vs Selenium. "
                "Templates can override them via their own `fetch` block."
            )

            requires_js_new = st.checkbox(
                "Prefer JS engine (Selenium) for this site",
                value=requires_js_val,
                help=(
                    "If checked, `_prefers_selenium` will default to Selenium "
                    "(`force_js=True`) unless a matching template overrides "
                    "`fetch.requires_js`."
                ),
            )

            wait_css_new = st.text_input(
                "Wait CSS (optional, Selenium only)",
                value=wait_css_val,
                help=(
                    "Optional CSS selector Selenium should wait for before "
                    "reading the HTML (e.g. the main article container)."
                ),
            )

            wait_timeout_new = st.number_input(
                "Wait timeout (seconds, Selenium only)",
                min_value=1.0,
                max_value=120.0,
                step=1.0,
                value=wait_timeout_val,
            )

            st.markdown("---")

            # ---- Consent click config (cookie walls) ----
            existing_xpaths = consent_cfg.get("xpaths") or []
            existing_xpaths = [
                x for x in existing_xpaths
                if isinstance(x, str) and x.strip()
            ]

            consent_enabled_val = bool(
                consent_cfg.get("enabled", False) or bool(existing_xpaths)
            )

            st.caption(
                "Consent XPaths are clicked (in order) before reading the HTML, "
                "to close cookie / consent dialogs."
            )

            consent_enabled_new = st.checkbox(
                "Enable consent / cookie clicks for this site",
                value=consent_enabled_val,
            )

            xpaths_text_default = "\n".join(existing_xpaths)
            xpaths_text_new = st.text_area(
                "Consent XPaths (one per line)",
                value=xpaths_text_default,
                height=120,
                help=(
                    "Example:\n"
                    "//*[@id='CybotCookiebotDialogBodyButtonDecline']\n"
                    "//button[contains(@class,'accept')]"
                ),
            )

            if st.button("Save fetch + consent defaults", key="btn_save_site_fetch"):
                # Write back fetch defaults
                fetch_cfg["requires_js"] = bool(requires_js_new)
                fetch_cfg["wait_css"] = wait_css_new.strip()
                fetch_cfg["wait_timeout"] = float(wait_timeout_new)

                # Normalize XPaths from textarea
                lines = [
                    ln.strip()
                    for ln in xpaths_text_new.splitlines()
                    if ln.strip()
                ]

                if lines:
                    consent_cfg["enabled"] = bool(consent_enabled_new)
                    consent_cfg["xpaths"] = lines
                else:
                    # No XPaths → disable consent block
                    consent_cfg["enabled"] = False
                    consent_cfg["xpaths"] = []

                fetch_cfg["consent"] = consent_cfg

                # Ensure site id present
                site_cfg.setdefault("site", site_key)
                site_cfg["fetch"] = fetch_cfg

                # Persist in session and on disk
                st.session_state["site_cfg"] = site_cfg
                _save_site_yaml(site_key, site_cfg)

                st.success("Site fetch + consent defaults saved.")

    else:
        st.info("Pick or create a site to continue.")

# ---------------------------------------------------------------------------
# Fetch helpers & template matching
# ---------------------------------------------------------------------------

def _effective_consent_xpaths(site_cfg: Dict[str, Any], tpl_name: str | None) -> List[str]:
    """
    Combine site-level and template-level consent xpaths.
    """
    site_xp = (
        site_cfg.get("fetch", {})
        .get("consent", {})
        .get("xpaths", [])
    ) or []

    tpl_xp: List[str] = []
    for t in site_cfg.get("templates", []):
        if t.get("name") == tpl_name:
            tpl_xp = (
                t.get("fetch", {})
                .get("consent", {})
                .get("xpaths", [])
            ) or []
            break

    # normalize to strings
    return [x for x in site_xp + tpl_xp if isinstance(x, str) and x.strip()]


def _prefers_selenium(site_cfg: Dict[str, Any], url: str) -> Tuple[bool, str, float]:
    """
    Decide if template prefers JS (Selenium) engine.
    Returns: (force_js, wait_css, wait_timeout)
    """
    # Default site-wide preferences
    site_fetch = site_cfg.get("fetch", {}) or {}
    site_requires_js = bool(site_fetch.get("requires_js", False))
    site_wait_css = site_fetch.get("wait_css") or ""
    site_wait_timeout = float(site_fetch.get("wait_timeout") or 20)

    # Try template-specific override (if applicable)
    tpl_wait_css = ""
    tpl_wait_timeout = None
    tpl_requires_js = None

    for t in site_cfg.get("templates", []):
        regex = (t.get("url_regex") or "").strip()
        if regex and re.search(regex, url):
            tf = t.get("fetch", {}) or {}
            tpl_requires_js = tf.get("requires_js")
            tpl_wait_css = tf.get("wait_css") or ""
            tpl_wait_timeout = tf.get("wait_timeout")
            break

    # Merge with template-level overrides:
    requires_js = tpl_requires_js if tpl_requires_js is not None else site_requires_js
    wait_css = tpl_wait_css or site_wait_css
    wait_timeout = float(tpl_wait_timeout or site_wait_timeout or 20)

    return bool(requires_js), wait_css, wait_timeout


def _templates_matching_url(site_cfg: Dict[str, Any], url: str) -> List[Dict[str, Any]]:
    """Return all templates whose url_regex matches this URL."""
    matched = []
    for t in site_cfg.get("templates", []):
        regex = (t.get("url_regex") or "").strip()
        if not regex:
            continue
        try:
            if re.search(regex, url):
                matched.append(t)
        except Exception:
            pass
    return matched

def _candidate_templates_for_url(site_cfg: Dict[str, Any], url: str) -> List[Dict[str, Any]]:
    """
    Return enabled templates to try for this URL, ordered by priority (high→low).
    Prefer templates whose url_regex matches; if none match, fall back to all enabled templates.
    """
    all_tpls = site_cfg.get("templates") or []
    enabled = [t for t in all_tpls if t.get("enabled", True)]
    if not enabled:
        return []

    matched: list[Dict[str, Any]] = []
    for t in enabled:
        pattern = (t.get("url_regex") or "").strip()
        if not pattern:
            continue
        try:
            if re.search(pattern, url):
                matched.append(t)
        except Exception:
            continue

    candidates = matched or enabled

    def _prio(t: Dict[str, Any]) -> int:
        try:
            return int(t.get("priority", 10) or 10)
        except Exception:
            return 10

    # Higher priority first
    return sorted(candidates, key=_prio, reverse=True)


# -----------------------------------------------------------------------------
# B) Article URL(s)
# -----------------------------------------------------------------------------
st.subheader("B) Article URL(s)")

col_url, col_clear = st.columns([3, 1])

with col_url:
    # -------------------------------------------------------------------------
    # 1) Handle handoff from View Data / other pages
    # -------------------------------------------------------------------------
    HANDOFF_SINGLE = TMP_DIR / "builder_single_url.json"
    HANDOFF_MATRIX = TMP_DIR / "builder_matrix_urls.json"

    # Incoming single URL (from View Data or elsewhere)
    if HANDOFF_SINGLE.exists():
        try:
            payload = json.loads(HANDOFF_SINGLE.read_text(encoding="utf-8"))
            incoming_url = (payload.get("url") or "").strip()
            if incoming_url.startswith("http"):
                # Stage URL so that it will be in the text_input on this run
                st.session_state["__next_test_url"] = incoming_url
                # Clear fetch-related caches so we actually refetch
                st.session_state["final_url"] = ""
                st.session_state["html_cache"] = ""
                st.session_state["engine"] = ""
                # Keep regex test state in sync with this URL (later sections)
                rev = st.session_state.get("editor_rev", 0)
                st.session_state[f"tpl_regex_test_{rev}"] = incoming_url
                st.session_state["__regex_test_last_url"] = incoming_url
        finally:
            HANDOFF_SINGLE.unlink(missing_ok=True)

    # Incoming MATRIX URLs (from View Data page or run logs)
    if HANDOFF_MATRIX.exists():
        try:
            payload = json.loads(HANDOFF_MATRIX.read_text(encoding="utf-8"))
            raw_urls = payload.get("urls") or []
            urls = [
                u.strip()
                for u in raw_urls
                if isinstance(u, str) and u.strip().startswith("http")
            ]
            # De-duplicate while preserving order
            seen = set()
            urls = [u for u in urls if not (u in seen or seen.add(u))]

            if urls:
                st.session_state["matrix_urls"] = urls
                st.session_state["matrix_urls_rev"] = (
                    st.session_state.get("matrix_urls_rev", 0) + 1
                )
                st.session_state["matrix_source"] = payload.get("source", "run_logs")
        finally:
            HANDOFF_MATRIX.unlink(missing_ok=True)

    # Promote staged URL (from handoff or CSV picker) into the text_input
    if "__next_test_url" in st.session_state:
        st.session_state["test_url"] = st.session_state.pop("__next_test_url")

    # -------------------------------------------------------------------------
    # 2) Main "Article URL" text input
    # -------------------------------------------------------------------------
    test_url = st.text_input(
        "Article URL",
        key="test_url",
        placeholder="Paste a full article URL…",
        help="Example: https://example.com/news/world/story-123",
    )

with col_clear:
    if st.button("Clear URL"):
        # stage an empty value for next run
        st.session_state["__next_test_url"] = ""
        # clear fetch-related caches so previews reset
        st.session_state["final_url"] = ""
        st.session_state["html_cache"] = ""
        st.session_state["engine"] = ""
        st.session_state["__regex_test_dirty"] = False
        _st_rerun()

# -------------------------------------------------------------------------
# 3) Pick a URL from teasers CSV
# -------------------------------------------------------------------------

st.markdown("**Or pick from a teasers CSV**")

def _list_csvs(folder: Path, site_key: str) -> list[Path]:
    if not folder.exists():
        return []
    # only CSVs beginning with site_key (e.g., athensvoice_*)
    return sorted(
        [p for p in folder.glob("*.csv") if p.name.startswith(f"{site_key}")]
    )

def _guess_url_column(df: pd.DataFrame) -> str | None:
    # common names first
    candidates = ["url", "link", "href", "article_url", "article"]
    for c in candidates:
        if c in df.columns:
            return c
    # heuristic: first column whose first non-null value looks like an http(s) URL
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if not s.empty and s.iloc[0].startswith(("http://", "https://")):
            return c
    return None

with st.expander("Pick URL from teasers CSV", expanded=False):
    if not st.session_state.get("site_key"):
        st.warning("Choose a **site** first in A) so I can filter CSVs by site key.")
    else:
        def _csv_folder_sync():
            # copy widget value into persistent storage on every change
            val = st.session_state["csv_folder_dir_input"]
            if val is not None:
                st.session_state["csv_folder_dir_persist"] = val

        csv_folder_str = st.text_input(
            "Folder with CSV files",
            key="csv_folder_dir_input",
            help="Path that contains the teasers CSVs (e.g., data/outputs)",
            on_change=_csv_folder_sync,
        )
        csv_folder = (
            Path(st.session_state["csv_folder_dir_persist"])
            .expanduser()
            .resolve()
        )

        site_key = st.session_state.get("site_key")
        csv_files = _list_csvs(csv_folder, site_key)

        if not csv_files:
            st.info(
                f"No CSV files starting with '{site_key}' found in\n{csv_folder}"
            )
        else:
            csv_labels = [p.name for p in csv_files]
            idx = st.selectbox(
                "Teasers CSV file",
                options=list(range(len(csv_files))),
                format_func=lambda i: csv_labels[i],
                key="csv_picker_file_idx",
            )
            csv_path = csv_files[idx]

            st.caption(f"Using: `{csv_path}`")

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df = None

            if df is not None and not df.empty:
                url_cols = list(df.columns)
                guessed = _guess_url_column(df) or (url_cols[0] if url_cols else None)
                url_col = st.selectbox(
                    "URL column",
                    options=url_cols,
                    index=url_cols.index(guessed) if guessed in url_cols else 0,
                    help="Column that contains the article URLs",
                    key="csv_url_column",
                )

                urls_series = df[url_col].dropna().astype(str)
                # Make unique while preserving order
                seen = set()
                urls = [u for u in urls_series.tolist() if not (u in seen or seen.add(u))]

                clmn1, clmn2 = st.columns([1, 1])
                with clmn1:
                    # option to keep only the first URL per section
                    one_per_section = st.checkbox(
                        "Get one URL from each section",
                        value=False,
                        help="Shows only the first URL per section (based on the selected path segment).",
                        key="csv_one_per_section",
                    )
                with clmn2:
                    # Which path segment defines the 'section' (1 = first, 2 = second, ...)
                    seg_idx_for_picker = st.number_input(
                        "Section segment index",
                        min_value=1,
                        max_value=6,
                        value=st.session_state.get("csv_seg_idx", 1),
                        step=1,
                        key="csv_seg_idx",
                        help=(
                            "Use 1 for https://site/<section>/..., "
                            "or 2 for https://site/news/<section>/..."
                        ),
                    )

                if one_per_section:
                    seen_sections = set()
                    urls_for_picker = []
                    for u in urls:
                        sec = _url_section_at(
                            u, int(st.session_state["csv_seg_idx"])
                        )
                        if sec not in seen_sections:
                            seen_sections.add(sec)
                            urls_for_picker.append(u)
                else:
                    urls_for_picker = urls

                if not urls_for_picker:
                    st.warning("No URLs match the filter (or column is empty).")
                else:
                    pick_idx = st.selectbox(
                        "Pick URL",
                        options=list(range(len(urls_for_picker))),
                        format_func=lambda i: urls_for_picker[i],
                        key="csv_pick_url_idx",
                    )
                    picked_url = urls_for_picker[pick_idx]

                    if st.button("Use this URL", key="btn_use_picked_url"):
                        # 1) stage for the Article URL input (will be promoted before it’s created)
                        st.session_state["__next_test_url"] = picked_url
                        # 2) keep regex testing in sync (later sections)
                        rev = st.session_state.get("editor_rev", 0)
                        st.session_state[f"tpl_regex_test_{rev}"] = picked_url
                        st.session_state["__regex_test_last_url"] = picked_url
                        st.session_state["__regex_test_dirty"] = False
                        _st_rerun()

    # -------------------------------------------------------------------------
    # 4) Build a section-balanced sample (for Matrix test)
    # -------------------------------------------------------------------------
    # st.markdown("---")
    st.subheader("Build a section-balanced sample (for Matrix test)")

    clm1, clm2, clm3 = st.columns([1, 1, 1])
    with clm1:
        per_section = st.number_input(
            "Max URLs per section",
            min_value=1,
            max_value=100,
            value=1,
            key="mtx_per_section",
        )
    with clm2:
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            value=42,
            key="mtx_seed",
        )

    with clm3:
        seg_idx_for_sample = st.number_input(
            "Section segment index (for sampling)",
            min_value=1,
            max_value=6,
            value=st.session_state.get("csv_seg_idx", 1),
            step=1,
            key="mtx_seg_idx",
        )

    if st.button("Make sample from this CSV", key="btn_make_matrix_sample"):
        # use the last computed `urls` from the CSV expander above.
        try:
            sample = _sample_by_section(
                urls,
                per_section=int(per_section),
                seed=int(seed),
                seg_idx=int(st.session_state.get("mtx_seg_idx", 1)),
            )
        except NameError:
            st.error("No CSV loaded yet – open 'Pick URL from teasers CSV' first.")
        else:
            st.session_state["matrix_urls"] = sample
            st.session_state["matrix_urls_rev"] = (
                st.session_state.get("matrix_urls_rev", 0) + 1
            )
            st.session_state["matrix_source"] = "teasers_csv"
            st.success(f"Prepared {len(sample)} sampled URLs across sections.")
            _st_rerun()

# -------------------------------------------------------------------------
# 5) Show current sample set (from CSV or from View Data / run logs)
# -------------------------------------------------------------------------
if st.session_state.get("matrix_urls"):
    src = st.session_state.get("matrix_source")
    label_map = {
        "teasers_csv": "Current sample set — from teasers CSV",
        "run_logs": "Current sample set — from Run Logs / View Data",
    }
    label = label_map.get(src, "Current sample set")

    with st.expander(label, expanded=True):
        urls_view = st.session_state.get("matrix_urls", [])
        st.caption(f"{len(urls_view)} URL(s) in sample")

        btn_cols = st.columns([1, 3])
        with btn_cols[0]:
            if st.button("Clear sample set", key="btn_clear_matrix_sample"):
                st.session_state.pop("matrix_urls", None)
                st.session_state.pop("matrix_urls_rev", None)
                st.session_state.pop("matrix_source", None)
                _st_rerun()

        st.dataframe(
            pd.DataFrame({"URL": urls_view}),
            width="stretch",
            height=220,
        )

# -----------------------------------------------------------------------------
# C) Fetch page
# -----------------------------------------------------------------------------
st.subheader("C) Fetch page")

test_url = st.session_state.get("test_url", "").strip()
site_cfg = st.session_state.get("site_cfg") or {}
site_key = st.session_state.get("site_key") or ""

if not test_url:
    st.info("Paste an article URL in **B)** to enable fetching.")
else:
    # Display URL and allow manual correction
    st.write("**URL to fetch:**")
    st.code(test_url)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Fetch now", key="btn_fetch_page"):
            # Reset caches before fetch
            st.session_state["html_cache"] = ""
            st.session_state["final_url"] = ""
            st.session_state["engine"] = ""

            if DEMO_MODE:
                # -------------------------------------------
                # DEMO: load HTML from the local snapshot
                # -------------------------------------------
                demo_key = (site_key or "demo").strip() or "demo"
                candidates = sorted(DEMO_HTML_DIR.glob(f"{demo_key}_*.html"))

                # Fallback: any HTML file in demo folder
                if not candidates:
                    candidates = sorted(DEMO_HTML_DIR.glob("*.html"))

                if not candidates:
                    st.error(
                        "Demo mode: no HTML snapshots found under "
                        "`data/demo/html/`. Add at least one "
                        f"`{demo_key}_*.html` file."
                    )
                else:
                    demo_path = candidates[0]
                    try:
                        html = demo_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception as e:
                        st.error(f"Demo mode: failed to read {demo_path.name}: {e}")
                    else:
                        st.session_state["html_cache"] = html
                        st.session_state["final_url"] = test_url
                        st.session_state["engine"] = "demo"
                        st.success(
                            f"Demo mode: loaded HTML snapshot `{demo_path.name}` "
                            "instead of fetching from the live site."
                        )
            else:
                # -------------------------------------------
                # NORMAL MODE: real HTTP fetch
                # -------------------------------------------

                try:
                    # Decide engine (requests / selenium) based on site config + URL
                    requires_js, wait_css, wait_timeout = _prefers_selenium(site_cfg, test_url)

                    # Site-level + template-level consent XPaths.
                    # None for tpl_name.
                    consent_xpaths = _effective_consent_xpaths(site_cfg, None)

                    # Perform the fetch using the same signature as the old page
                    res: FetchResult = fetch_with_fallback(
                        test_url,
                        container_css=wait_css or "",
                        item_css="",
                        force_js=requires_js,
                        wait_timeout=wait_timeout,
                        consent_click_xpaths=consent_xpaths or None,
                    )

                    # Store outcome
                    st.session_state["html_cache"] = res.html or ""
                    st.session_state["final_url"] = res.final_url or test_url
                    st.session_state["engine"] = res.engine or "requests"

                    engine_label = (res.engine or "requests").upper()
                    st.success(f"Fetched using **{engine_label}**")
                    if res.final_url and res.final_url != test_url:
                        st.caption(f"Redirected to: {res.final_url}")

                except Exception as e:
                    st.error(f"Fetch failed: {e}")

    with col2:
        if st.button("Clear fetched HTML", key="btn_clear_html"):
            st.session_state["html_cache"] = ""
            st.session_state["final_url"] = ""
            st.session_state["engine"] = ""
            st.info("Cleared cached HTML.")

    # After-fetch display
    html_cache = st.session_state.get("html_cache") or ""
    final_url = st.session_state.get("final_url") or ""
    engine = st.session_state.get("engine") or ""

    if html_cache:
        st.markdown("### Fetch summary")

        # Meta-info
        st.write(f"- **Final URL:** {final_url or test_url}")
        st.write(f"- **Engine:** {(engine or 'requests').upper()}")
        st.write(f"- **HTML size:** {len(html_cache):,} chars")

        # Raw HTML viewer
        with st.expander("Show raw HTML", expanded=False):
            st.text_area(
                "HTML source",
                html_cache,
                height=350,
            )

        # Rendered HTML (from cache)
        with st.expander("Rendered HTML preview (from cache)", expanded=False):
            st.caption("Renders the cached HTML below (scripts won't run). You can open inspector to see the tags and classes for various elements.")
            # Wrap cached HTML to force a white background for the preview
            preview_html = f"""
            <html>
              <head>
                <style>
                  html, body {{
                    background: #ffffff !important;
                    margin: 0;
                    padding: 0;
                  }}
                </style>
              </head>
              <body>
                {html_cache}
              </body>
            </html>
            """
            components.html(preview_html, height=400, scrolling=True)

    else:
        st.info("No HTML cached yet. Click **Fetch now**.")

# -----------------------------------------------------------------------------
# D) Templates (list, add, basic meta)
# -----------------------------------------------------------------------------
st.subheader("D) Templates")

site_key = st.session_state.get("site_key") or ""
site_cfg = st.session_state.get("site_cfg") or {}

if not site_key:
    st.info("Choose a **site** in A) to manage templates.")
else:
    templates: list[Dict[str, Any]] = site_cfg.get("templates") or []

    if not templates:
        st.warning(
            "No templates defined yet for this site. "
            "Create your first template below."
        )
    # Ensure we always have at least one template if site_cfg exists
    if not templates:
        tpl = dict(DEFAULT_GENERIC_TEMPLATE)
        tpl["name"] = "default-article"
        site_cfg["templates"] = [tpl]
        st.session_state["site_cfg"] = site_cfg
        st.session_state["editing_template_name"] = tpl["name"]
        _save_site_yaml(site_key, site_cfg)
        templates = site_cfg["templates"]

    # Available template names
    tpl_names = [t.get("name", "<unnamed>") for t in templates]

    # Pick current editing template
    current_name = st.session_state.get("editing_template_name")
    if not current_name or current_name not in tpl_names:
        current_name = tpl_names[0]
        st.session_state["editing_template_name"] = current_name

    # when template selection changes, load it into form state and rerun
    def _on_change_template_select():
        name = st.session_state.get("tpl_select_name", "")
        if not name:
            return

        # Update which template we're editing
        st.session_state["editing_template_name"] = name

        # Find that template in the current site config and load selectors
        site_cfg_local = st.session_state.get("site_cfg") or {}
        for t in site_cfg_local.get("templates", []):
            if t.get("name") == name:
                _load_template_into_form_state(t)
                break

        # Clear meta form state so the Template meta section re-initializes
        for k in ("tpl_name", "tpl_enabled", "tpl_priority", "tpl_regex"):
            st.session_state.pop(k, None)

        _st_rerun()

    # Template selector + add-new
    col_sel, col_add = st.columns([2, 1])
    with col_sel:
        selected_name = st.selectbox(
            "Choose template to edit",
            options=tpl_names,
            index=tpl_names.index(current_name),
            key="tpl_select_name",
            on_change=_on_change_template_select,
        )

    with col_add:
        new_name = st.text_input(
            "New template name",
            key="tpl_new_name",
            placeholder="e.g. article-main, opinion, liveblog",
        ).strip()
        if st.button("Add new template", key="btn_add_template"):
            if not new_name:
                st.error("Template name cannot be empty.")
            elif new_name in tpl_names:
                st.error("A template with this name already exists.")
            else:
                new_tpl = json.loads(json.dumps(DEFAULT_GENERIC_TEMPLATE))
                new_tpl["name"] = new_name
                _upsert_template(site_cfg, new_tpl)
                _save_site_yaml(site_key, site_cfg)
                st.session_state["site_cfg"] = site_cfg
                st.session_state["editing_template_name"] = new_name

                # Load selectors for the new template
                _load_template_into_form_state(new_tpl)

                # Clear meta form state so Template meta section initializes them
                for k in ("tpl_name", "tpl_enabled", "tpl_priority", "tpl_regex"):
                    st.session_state.pop(k, None)

                st.success(f"Template **{new_name}** created.")
                _st_rerun()

    # Editing banner
    _editing_banner()

    # Small vertical spacer between banner and controls
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # Reload template list & current template after any add/select change
    templates = st.session_state.get("site_cfg", {}).get("templates", [])
    tpl_names = [t.get("name", "<unnamed>") for t in templates]
    current_name = st.session_state.get("editing_template_name")
    current_tpl: Dict[str, Any] = {}
    for t in templates:
        if t.get("name") == current_name:
            current_tpl = t
            break

    if not current_tpl:
        st.warning("No active template selected.")
    else:
        st.markdown("### Template meta")
        st.caption(
            "These fields control where and how the template is used. "
            "Selectors come in the next section."
        )

        # Sync ONLY meta form state from the current template on first load
        if "tpl_name" not in st.session_state:
            st.session_state["tpl_name"] = current_tpl.get("name", "") or ""
            st.session_state["tpl_enabled"] = bool(current_tpl.get("enabled", True))
            st.session_state["tpl_priority"] = int(current_tpl.get("priority", 10) or 10)
            st.session_state["tpl_regex"] = current_tpl.get("url_regex", "") or ""

        with st.form("tpl_meta_form", clear_on_submit=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                name_val = st.text_input(
                    "Template name",
                    key="tpl_name",
                    help="Unique name for this template within the site.",
                )
            with col2:
                enabled_val = st.checkbox(
                    "Enabled",
                    key="tpl_enabled",
                    help="If disabled, this template will be ignored when matching URLs.",
                )

            col3, col4 = st.columns([1, 3])
            with col3:
                prio_val = st.number_input(
                    "Priority",
                    min_value=0,
                    max_value=1000,
                    step=1,
                    key="tpl_priority",
                    help="Higher priority wins when multiple templates match a URL.",
                )
            with col4:
                regex_val = st.text_input(
                    "URL regex (Python re)",
                    key="tpl_regex",
                    placeholder=r"https://www\.example\.gr/.+",
                    help="Only URLs matching this regex will use this template.",
                )

            save_meta = st.form_submit_button("Save template meta")

        # Handle save
        if save_meta:
            final_name = (name_val or "").strip()
            if not final_name:
                st.error("Template name cannot be empty.")
            else:
                # start from existing template so we don't lose extract/fetch blocks
                updated_tpl = json.loads(json.dumps(current_tpl))
                updated_tpl["name"] = final_name
                updated_tpl["enabled"] = bool(enabled_val)
                updated_tpl["priority"] = int(prio_val or 0)
                updated_tpl["url_regex"] = regex_val or ""

                # Persist
                _upsert_template(site_cfg, updated_tpl)
                _save_site_yaml(site_key, site_cfg)
                st.session_state["site_cfg"] = site_cfg
                st.session_state["editing_template_name"] = final_name

                st.success(f"Template **{final_name}** saved.")
                _st_rerun()

        # Clone template
        with st.expander("Clone this template", expanded=False):
            clone_name = st.text_input(
                "New template name (clone of this one)",
                key="tpl_clone_name",
                placeholder=f"{current_name}-copy",
            ).strip()

            if st.button("Clone template", key="btn_clone_template"):
                if not clone_name:
                    st.error("Clone name cannot be empty.")
                else:
                    # Check for duplicates
                    existing_names = [t.get("name", "") for t in site_cfg.get("templates", [])]
                    if clone_name in existing_names:
                        st.error("A template with this name already exists.")
                    else:
                        # Deep-copy current template and rename
                        cloned_tpl = json.loads(json.dumps(current_tpl))
                        cloned_tpl["name"] = clone_name

                        _upsert_template(site_cfg, cloned_tpl)
                        _save_site_yaml(site_key, site_cfg)
                        st.session_state["site_cfg"] = site_cfg
                        st.session_state["editing_template_name"] = clone_name

                        # Load selectors for the cloned template into the form
                        _load_template_into_form_state(cloned_tpl)

                        # Clear meta form state so Template meta section initializes it for the clone
                        for k in ("tpl_name", "tpl_enabled", "tpl_priority", "tpl_regex"):
                            st.session_state.pop(k, None)

                        st.success(f"Template **{current_name}** cloned as **{clone_name}**.")
                        _st_rerun()

        # Delete template
        if st.button("Delete this template", key="btn_delete_template"):
            remaining = [
                t for t in site_cfg.get("templates", [])
                if t.get("name") != current_name
            ]
            if not remaining:
                st.error("Cannot delete the last template of a site.")
            else:
                site_cfg["templates"] = remaining
                _save_site_yaml(site_key, site_cfg)
                st.session_state["site_cfg"] = site_cfg
                # switch editing to first remaining
                st.session_state["editing_template_name"] = remaining[0].get("name", "")
                st.success(f"Template **{current_name}** deleted.")
                _st_rerun()

# -----------------------------------------------------------------------------
# Selectors & extraction rules
# -----------------------------------------------------------------------------

# Helpers to read the template’s saved extract rules and produce a clean dict per URL/template
def _article_body_text(art: Dict[str, Any]) -> str:
    """
    Normalize where body text is stored in an article dict.
    Prefer 'body', fall back to 'text', then empty string.
    """
    if not isinstance(art, dict):
        return ""
    body = (art.get("body") or art.get("text") or "") or ""
    return str(body)


def _extract_article_with_template(html: str, url: str, tpl: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight extractor used by Matrix:
    - Uses the template's saved `extract` rules.
    - Returns a dict with title/section/dates/body + metrics.
    """
    rules = tpl.get("extract") or {}
    body_cfg = rules.get("body_config") or {}
    ct_cfg = rules.get("container_text") or {}
    accept_cfg = rules.get("accept") or {}

    main_css = (body_cfg.get("main_container_css") or "").strip()
    body_css = (body_cfg.get("body_container_css") or "").strip()
    exclude_sel = ct_cfg.get("exclude") or []
    boilerplate_rx = ct_cfg.get("boilerplate_regex") or []

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Scope to main container if set
    root = soup
    if main_css:
        node = soup.select_one(main_css)
        if node:
            root = node

    body_node = root
    if body_css:
        node = root.select_one(body_css)
        if node:
            body_node = node

    # Remove excluded selectors
    for sel in exclude_sel:
        sel = (sel or "").strip()
        if not sel:
            continue
        try:
            for n in body_node.select(sel):
                n.decompose()
        except Exception:
            # ignore bad selectors
            continue

    # Remove script/style
    for bad in body_node.find_all(["script", "style"]):
        bad.decompose()

    # Body text & boilerplate regex cleanup
    body_text = body_node.get_text(" ", strip=True)
    for pattern in boilerplate_rx:
        pattern = (pattern or "").strip()
        if not pattern:
            continue
        try:
            rx = re.compile(pattern, flags=re.MULTILINE | re.UNICODE)
            body_text = rx.sub("", body_text)
        except Exception:
            # bad regex → skip
            continue

    body_text = re.sub(r"\s+", " ", body_text).strip()
    body_len = len(body_text)

    try:
        min_chars = int(accept_cfg.get("min_body_chars", 400) or 400)
    except Exception:
        min_chars = 400

    def _first_text(css: Optional[str]) -> str:
        css = (css or "").strip()
        if not css:
            return ""
        try:
            nodes = root.select(css)
        except Exception:
            return ""
        for n in nodes:
            txt = n.get_text(" ", strip=True)
            if txt:
                return txt
        return ""

    def _all_texts(css: Optional[str], max_items: int = 50) -> List[str]:
        css = (css or "").strip()
        if not css:
            return []
        try:
            nodes = root.select(css)
        except Exception:
            return []
        out: List[str] = []
        for n in nodes[:max_items]:
            txt = n.get_text(" ", strip=True)
            if txt:
                out.append(txt)
        return out

    # Simple fields (title, lead, etc.) based on extract rules
    title_css, _ = _rule_to_css_attr(rules.get("title"))
    subtitle_css, _ = _rule_to_css_attr(rules.get("subtitle"))
    lead_css, _ = _rule_to_css_attr(rules.get("lead"))
    section_css, _ = _rule_to_css_attr(rules.get("section"))
    author_css, _ = _rule_to_css_attr(rules.get("author"))
    tags_css, _ = _rule_to_css_attr(rules.get("tags"))
    published_css, _ = _rule_to_css_attr(rules.get("published_time"))
    updated_css, _ = _rule_to_css_attr(rules.get("updated_time"))

    title = _first_text(title_css)
    subtitle = _first_text(subtitle_css)
    lead = _first_text(lead_css)
    section = _first_text(section_css)
    author = _first_text(author_css)
    tags = _all_texts(tags_css) if tags_css else []

    published_raw = _first_text(published_css)
    updated_raw = _first_text(updated_css)

    # ---- Postprocess with ISO dates + tidy text ---------------------------------
    # Build a minimal article dict for postprocess.py
    art_dict = {
        "url": url,
        "title": title,
        "subtitle": subtitle,
        "lead": lead,
        "section": section,
        "author": author,
        "tags": tags,
        # postprocess_article_dict expects these names for dates
        "published_time": published_raw,
        "updated_time": updated_raw,
        "body": body_text,
    }

    # We pass the main container HTML as fallback for sniffing meta tags
    art_clean = postprocess_article_dict(
        art_dict,
        html_for_fallback=str(root),
        site_hint=None,
        url_for_log=url,
    )

    # Pull back the (possibly normalized) values
    title_clean = art_clean.get("title", title)
    subtitle_clean = art_clean.get("subtitle", subtitle)
    lead_clean = art_clean.get("lead", lead)
    section_clean = art_clean.get("section", section)
    author_clean = art_clean.get("author", author)
    tags_clean = art_clean.get("tags", tags)
    published_iso = art_clean.get("published_time")
    updated_iso = art_clean.get("updated_time")

    # Acceptance logic:
    # - require title (raw or cleaned)
    # - require published date (ISO or raw)
    # - require body length > 0
    title_any = title_clean or title or ""
    has_title = bool((title_any or "").strip())

    # Prefer normalized ISO; fall back to raw CSS text
    published_any = published_iso if isinstance(published_iso, str) and published_iso.strip() else published_raw
    has_published = bool((published_any or "").strip())

    has_body = body_len > 0

    accepted = has_title and has_published and has_body

    # "full" vs "brief" uses min_chars, but only if accepted
    accepted_full = bool(accepted and body_len >= min_chars)

    if not has_title:
        reason = "missing: title"
    elif not has_published:
        reason = "missing: published_time"
    elif not has_body:
        reason = "missing: body"
    elif body_len < min_chars:
        reason = f"accepted: brief (<{min_chars})"
    else:
        reason = "accepted: full"

    return {
        "url": url,
        "template_name": tpl.get("name", ""),
        "title": title_clean,
        "subtitle": subtitle_clean,
        "lead": lead_clean,
        "section": section_clean,
        "author": author_clean,
        "tags": tags_clean,
        # keep raw strings for debugging
        "published_time_raw": published_raw,
        "updated_time_raw": updated_raw,
        # and expose normalized ISO datetimes
        "published_time": published_iso,
        "updated_time": updated_iso,
        "body": art_clean.get("body", body_text),
        "body_chars": body_len,
        "min_body_chars": min_chars,
        "accepted": accepted,
        "accepted_full": accepted_full,
        "reason": reason,
    }

st.subheader("Selectors & extraction rules (of the current template)")

html_cache = st.session_state.get("html_cache") or ""
if not html_cache:
    st.info("Fetch a page in **C) Fetch page** to preview selectors.")
else:
    site_key = st.session_state.get("site_key") or ""
    site_cfg = st.session_state.get("site_cfg") or {}
    templates = site_cfg.get("templates") or []

    if not site_key or not templates:
        st.info("Choose a site in A) and make sure it has at least one template in D).")
    else:
        current_name = st.session_state.get("editing_template_name") or templates[0].get("name")
        current_tpl: Dict[str, Any] = {}
        for t in templates:
            if t.get("name") == current_name:
                current_tpl = t
                break

        if not current_tpl:
            st.warning("No active template selected. Pick one in D) first.")
        else:
            # Ensure selector form state is initialized from current template
            if "fld_main_container_css" not in st.session_state:
                _load_template_into_form_state(current_tpl)

            container_css = st.session_state.get("fld_main_container_css", "") or ""

            # --- Group 1: Containers -------------------------------------------------
            with st.expander("Containers (main + body)", expanded=True):
                st.caption("Define where the article content lives inside the page.")

                main_sel = st.text_input(
                    "Main container CSS selector",
                    key="fld_main_container_css",
                    placeholder="e.g. article, main .article, #content .post",
                    help="Typically the <article> or main content wrapper.",
                )

                if html_cache and main_sel.strip():
                    _preview_field(html_cache, "", main_sel, "Main container")

                body_sel = st.text_input(
                    "Body container CSS selector (optional)",
                    key="fld_body_container_css",
                    placeholder="e.g. .article-body, .story-body",
                    help="If empty, the whole main container is used as body.",
                )

                if html_cache and body_sel.strip():
                    _preview_field(html_cache, main_sel, body_sel, "Body container within main")

            # --- Group 2: Core fields (title / subtitle / lead) ----------------------
            with st.expander("Core fields (title / subtitle / lead)", expanded=False):
                st.caption("Headline and main text fields for the article.")

                title_css = st.text_input(
                    "Title CSS selector",
                    key="fld_title_css",
                    placeholder="e.g. h1.article-title",
                )
                if title_css.strip():
                    _preview_field(html_cache, container_css, title_css, "Title")

                subtitle_css = st.text_input(
                    "Subtitle CSS selector (optional)",
                    key="fld_subtitle_css",
                    placeholder="e.g. h2.article-subtitle",
                )
                if subtitle_css.strip():
                    _preview_field(html_cache, container_css, subtitle_css, "Subtitle")

                lead_css = st.text_input(
                    "Lead / intro CSS selector (optional)",
                    key="fld_lead_css",
                    placeholder="e.g. .article-lead",
                )
                if lead_css.strip():
                    _preview_field(html_cache, container_css, lead_css, "Lead")

            # --- Group 3: Metadata (section / author / tags) -------------------------
            with st.expander("Metadata (section / author / tags)", expanded=False):
                st.caption("Section/category, author, and tags.")

                section_css = st.text_input(
                    "Section CSS selector (optional)",
                    key="fld_section_css",
                    placeholder="e.g. .breadcrumbs li:last-child",
                )
                if section_css.strip():
                    _preview_field(html_cache, container_css, section_css, "Section")

                author_css = st.text_input(
                    "Author CSS selector (optional)",
                    key="fld_author_css",
                    placeholder="e.g. .byline a",
                )
                if author_css.strip():
                    _preview_field(html_cache, container_css, author_css, "Author")

                tags_css = st.text_input(
                    "Tags CSS selector (optional)",
                    key="fld_tags_css",
                    placeholder="e.g. .tags a",
                    help="Selector for each tag element; texts will be joined in post-processing.",
                )
                if tags_css.strip():
                    _preview_field(html_cache, container_css, tags_css, "Tags")

            # --- Group 4: Dates (published / updated) --------------------------------
            with st.expander("Dates (published / updated)", expanded=False):
                st.caption("Raw date/time strings; conversion to ISO happens in post-process.")

                pub_css = st.text_input(
                    "Published time CSS selector (optional)",
                    key="fld_published_css",
                    placeholder='e.g. time[itemprop="datePublished"]',
                )
                if pub_css.strip():
                    _preview_field(html_cache, container_css, pub_css, "Published time")

                upd_css = st.text_input(
                    "Updated time CSS selector (optional)",
                    key="fld_updated_css",
                    placeholder='e.g. time[itemprop="dateModified"]',
                )
                if upd_css.strip():
                    _preview_field(html_cache, container_css, upd_css, "Updated time")

            # --- Group 5: Images -----------------------------------------------------
            with st.expander("Images", expanded=False):
                st.caption(
                    "Main article image selector. You can include an attribute, "
                    "e.g. 'img.main-image::attr(src)'."
                )

                main_img_css = st.text_input(
                    "Main image selector",
                    key="fld_main_image_css",
                    placeholder="e.g. figure img.main-image or img.main-image::attr(src)",
                )

                if main_img_css.strip():
                    _preview_image_field(html_cache, container_css, main_img_css, "Main image")

            # --- Group 6: Body text options ------------------------------------------
            with st.expander("Body text options (cleanup & acceptance)", expanded=False):
                st.caption("Remove noise from the body and require a minimum body length.")

                body_excl = st.text_area(
                    "Selectors to exclude from body (one per line)",
                    key="fld_body_exclude",
                    height=120,
                    placeholder="e.g.\nscript\nstyle\n.ad\nnav\nfooter\n.share\n.related",
                    help="These selectors will be removed from the body container before extracting text.",
                )

                body_boiler = st.text_area(
                    "Boilerplate regex lines to remove from body (one per line)",
                    key="fld_body_boilerplate",
                    height=120,
                    placeholder="e.g.\n^Διαβάστε επίσης:.*\n^Δείτε επίσης:.*",
                    help="Lines matching these regexes will be removed from the final body text.",
                )

                min_body_chars = st.number_input(
                    "Minimum body characters to accept article as full (otherwise brief)",
                    min_value=0,
                    max_value=20000,
                    value=int(st.session_state.get("fld_min_body_chars", 400) or 400),
                    step=50,
                    key="fld_min_body_chars",
                )

                st.markdown("**Body text preview**")
                _preview_body_text(
                    html_cache,
                    st.session_state.get("fld_main_container_css", ""),
                    st.session_state.get("fld_body_container_css", ""),
                    label="Body text",
                )

            st.subheader("Current template's previews")

            # YAML preview of this template with current selectors
            preview_tpl = json.loads(json.dumps(current_tpl))
            preview_tpl["extract"] = _build_extract_from_form(current_tpl)

            with st.expander("All selectors and rules (YAML)", expanded=False):
                st.code(
                    yaml.safe_dump(
                        preview_tpl,
                        allow_unicode=True,
                        sort_keys=False,
                    ),
                    language="yaml",
                )

            # --- YAML preview of extracted article with current selectors -----------
            preview_url = st.session_state.get("final_url") or st.session_state.get("test_url") or ""
            try:
                extract_preview = _extract_preview_from_form(html_cache, preview_url)
            except Exception as e:
                extract_preview = {"error": f"Preview extraction failed: {e}"}

            with st.expander("All extractions preview (YAML)", expanded=False):
                st.code(
                    yaml.safe_dump(
                        extract_preview,
                        allow_unicode=True,
                        sort_keys=False,
                    ),
                    language="yaml",
                )

            # Save template button
            if st.button("Save selectors for this template", type="primary", key="btn_save_selectors"):
                updated_tpl = json.loads(json.dumps(current_tpl))
                updated_tpl["extract"] = _build_extract_from_form(current_tpl)

                _upsert_template(site_cfg, updated_tpl)
                _save_site_yaml(site_key, site_cfg)
                st.session_state["site_cfg"] = site_cfg

                st.success(f"Selectors saved for template **{updated_tpl.get('name', '')}**.")

            st.markdown("---")

# -----------------------------------------------------------------------------
# E) Matrix test — fetch URLs and test templates
# -----------------------------------------------------------------------------
st.subheader("E) Matrix test")

site_cfg = st.session_state.get("site_cfg") or {}
site_key = st.session_state.get("site_key") or ""
templates = site_cfg.get("templates") or []

matrix_urls: List[str] = st.session_state.get("matrix_urls", []) or []
matrix_source = st.session_state.get("matrix_source", "manual")

# Show info about sample set
with st.expander("Source of URLs for Matrix", expanded=False):
    st.markdown(f"- Source: **{matrix_source}**")
    st.markdown(f"- URLs in sample set: **{len(matrix_urls)}**")

if not site_key or not templates:
    st.info("Choose a **site** in A) and make sure it has at least one template in D).")
elif not matrix_urls:
    st.info(
        "No URLs in the Matrix sample set yet.\n\n"
        "- Use **B) Article URL(s)** → 'Build a section-balanced sample', or\n"
        "- Send URLs from the **View data** page."
    )
else:
    col_opts1, col_opts2, col_opts3 = st.columns([1, 1, 1])
    with col_opts1:
        max_urls = st.number_input(
            "Max URLs to test",
            min_value=1,
            max_value=len(matrix_urls),
            value=min(20, len(matrix_urls)),
            step=1,
        )
    with col_opts2:
        shuffle_urls = st.checkbox(
            "Shuffle sample before testing",
            value=True,
        )
    with col_opts3:
        only_current_tpl = st.checkbox(
            "Use only currently editing template",
            value=False,
            help="If on, Matrix will test only the active template instead of all candidates.",
        )

    # Optional random delay between requests (to be polite to the site's server)
    cola, colb = st.columns([1,1])
    with cola:
        delay_min = st.number_input(
            "Min delay between requests (seconds)",
            min_value=0.0,
            max_value=60.0,
            value=1.0,
            step=0.1,
        )
    with colb:
        delay_max = st.number_input(
            "Max delay between requests (seconds)",
            min_value=0.0,
            max_value=60.0,
            value=2.5,
            step=0.1,
        )
    if delay_max < delay_min:
        st.info("Max delay is less than min delay; using the min delay only.")

    run_matrix = st.button("Run Matrix test on sample set", type="primary")

    if run_matrix:
        urls_for_test = list(matrix_urls)
        if shuffle_urls:
            random.shuffle(urls_for_test)
        urls_for_test = urls_for_test[: int(max_urls or 1)]

        results: List[Dict[str, Any]] = []

        def _matrix_row_score(row: Dict[str, Any]) -> tuple[int, int, int, int]:
            """
            Score for choosing the best template per URL:

            1) accepted templates (title + published + body_len > 0) beat non-accepted
            2) among accepted, more non-empty 'other' fields win
            3) then longer body_chars wins
            """
            body_len = int(row.get("body_chars", 0) or 0)
            accepted = bool(row.get("accepted"))

            # Prefer fully accepted over brief when both are accepted
            reason = row.get("reason") or ""
            is_full = 1 if isinstance(reason, str) and reason.startswith("accepted: full") else 0

            # Other fields beyond title/published/body (present or not)
            info_fields = ["subtitle", "lead", "section", "author", "tags", "updated"]
            filled = 0
            for fld in info_fields:
                v = row.get(fld)
                if isinstance(v, str):
                    if v.strip():
                        filled += 1
                elif isinstance(v, list):
                    # tags: count as filled if there is at least one non-empty tag
                    if any(str(x).strip() for x in v if x is not None):
                        filled += 1
                elif v:
                    filled += 1


            # Returned tuple is compared lexicographically
            # (accepted, full, filled, body_len)
            return (
                1 if accepted else 0,
                is_full,
                filled,
                body_len,
            )

        for i, url in enumerate(urls_for_test, start=1):
            # Optional delay between requests (skip before first)
            if i > 1 and (delay_min > 0 or delay_max > 0):
                if delay_max > delay_min:
                    sleep_for = random.uniform(delay_min, delay_max)
                else:
                    sleep_for = delay_min
                time.sleep(sleep_for)

            st.write(f"Processing [{i}/{len(urls_for_test)}] {url}")

            # 1) Choose templates to try
            candidate_tpls = _candidate_templates_for_url(site_cfg, url)
            if only_current_tpl:
                current_name = st.session_state.get("editing_template_name") or ""
                candidate_tpls = [t for t in candidate_tpls if t.get("name") == current_name]

            if not candidate_tpls:
                results.append({
                    "url": url,
                    "template": "",
                    "engine": "",
                    "body_chars": 0,
                    "accepted": False,
                    "reason": "no_enabled_template",
                    "title": "",
                    "subtitle": "",
                    "lead": "",
                    "section": "",
                    "author": "",
                    "tags": [],
                    "published": "",
                    "updated": "",
                })
                continue

            # 2) Fetch page (fresh, not from html_cache)
            force_js, wait_css, wait_timeout = _prefers_selenium(site_cfg, url)
            consent_xpaths = _effective_consent_xpaths(site_cfg, None)

            try:
                res: FetchResult = fetch_with_fallback(
                    url,
                    container_css=wait_css or "",
                    item_css="",
                    force_js=force_js,
                    wait_timeout=wait_timeout,
                    consent_click_xpaths=consent_xpaths or None,
                )
            except Exception as e:
                results.append({
                    "url": url,
                    "template": "",
                    "engine": "",
                    "body_chars": 0,
                    "accepted": False,
                    "reason": f"fetch_error: {e}",
                    "title": "",
                    "subtitle": "",
                    "lead": "",
                    "section": "",
                    "author": "",
                    "tags": [],
                    "published": "",
                    "updated": "",
                })
                continue

            page_html = res.html or ""
            engine_used = (res.engine or "requests").upper()
            final_url = res.final_url or url

            if not page_html.strip():
                results.append({
                    "url": final_url,
                    "template": "",
                    "engine": engine_used,
                    "body_chars": 0,
                    "accepted": False,
                    "reason": "empty_html",
                    "title": "",
                    "subtitle": "",
                    "lead": "",
                    "section": "",
                    "author": "",
                    "tags": [],
                    "published": "",
                    "updated": "",
                })
                continue

            # 3) Try templates in priority order and pick best
            best_row: Optional[Dict[str, Any]] = None
            best_score: Optional[tuple[int, int, int, int]] = None

            for tpl in candidate_tpls:
                try:
                    art = _extract_article_with_template(page_html, final_url, tpl)
                except Exception as e:
                    row = {
                        "url": final_url,
                        "template": tpl.get("name", ""),
                        "engine": engine_used,
                        "body_chars": 0,
                        "accepted": False,
                        "reason": f"extract_error: {e}",
                        "title": "",
                        "subtitle": "",
                        "lead": "",
                        "section": "",
                        "author": "",
                        "tags": [],
                        "published": "",
                        "updated": "",
                    }

                else:
                    body_len = art.get("body_chars", 0)
                    accepted = bool(art.get("accepted"))
                    reason = art.get("reason", "")
                    row = {
                        "url": final_url,
                        "template": art.get("template_name", tpl.get("name", "")),
                        "engine": engine_used,
                        "body_chars": body_len,
                        "accepted": accepted,
                        "reason": reason,
                        "title": art.get("title", ""),
                        "subtitle": art.get("subtitle", ""),
                        "lead": art.get("lead", ""),
                        "section": art.get("section", ""),
                        "author": art.get("author", ""),
                        "tags": art.get("tags", []),
                        # Prefer normalized ISO dates; fall back to raw if missing
                        "published": art.get("published_time") or art.get("published_time_raw", ""),
                        "updated": art.get("updated_time") or art.get("updated_time_raw", ""),
                    }

                # Compute score and keep the best
                score = _matrix_row_score(row)
                if best_score is None or score > best_score:
                    best_row = row
                    best_score = score

            if best_row:
                results.append(best_row)

        # 4) Show results
        if results:
            df = pd.DataFrame(results)
            df = df[
                ["url", "template", "engine", "body_chars",
                 "accepted", "reason", "title", "subtitle",
                 "lead", "section", "author", "tags", 
                 "published", "updated"]
            ]

            st.markdown("### Matrix results")
            st.dataframe(df, width='stretch')

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Matrix results as CSV",
                data=csv_bytes,
                file_name=f"matrix_{site_key or 'site'}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No usable results were produced by the Matrix test.")

