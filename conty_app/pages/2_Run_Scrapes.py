import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from pathlib import Path
from urllib.parse import urlparse
import json, time, hashlib, random, re
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import requests
import yaml
import fnmatch as _fnm
from bs4 import BeautifulSoup

from conty_core.extract import extract_and_postprocess
from conty_core.fetcher import fetch_with_fallback, FetchResult
from conty_core.consent import KNOWN_CONSENT_XPATHS
from conty_core.postprocess import normalize_datetime_smart as _smart_norm

# ----------------------------- paths & utils -----------------------------
DATA_DIR = PROJECT_ROOT / "data"
SCRAPERS_DIR = DATA_DIR / "scrapers"
OUTPUTS_DIR = DATA_DIR / "outputs"
ARTICLES_DIR = OUTPUTS_DIR / "articles"
ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

HTML_ROOT = ARTICLES_DIR / "html"
HTML_FULL_DIR = HTML_ROOT / "full"
for d in (HTML_ROOT, HTML_FULL_DIR):
    d.mkdir(parents=True, exist_ok=True)

RUN_LOGS_DIR = OUTPUTS_DIR / "run_logs"
RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Demo mode: read-only sample data shipped with the repo
DEMO_DIR         = DATA_DIR / "demo"
DEMO_HTML_DIR    = DEMO_DIR / "html"
DEMO_TEASERS_DIR = DEMO_DIR / "teasers"
DEMO_OUTPUTS_DIR = DEMO_DIR / "outputs"

# --- Demo mode flag (per-page, simple) --------------------------------------
DEMO_MODE = bool(os.environ.get("CONTY_DEMO", "").strip())

def show_demo_banner():
    if DEMO_MODE:
        st.info(
            "🔒 **Demo mode** is ON. This page is restricted to demo teaser CSVs and "
            "saved HTML snapshots under `data/demo/`. No live HTTP requests are made."
        )

USER_AGENT = "Mozilla/5.0"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
}

FIELDNAMES = [
    "site","url","final_url","title","subtitle","section","author",
"published_raw", "published_datetime","published_date",
"updated_raw", "updated_datetime","updated_date",    "lead","text","main_image","tags",
"status","reason","template_used","fetched_datetime","full_html_path"
]

# Helpers for requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s

def _referer_for(url: str) -> str:
    try:
        u = urlparse(url)
        return f"{u.scheme}://{u.netloc}/" if u.scheme and u.netloc else ""
    except Exception:
        return ""

def _polite_sleep(j: int):
    # per-URL jitter
    time.sleep(random.uniform(1.5, 3.0))
    # a longer pause every ~20 URLs
    if j % 20 == 0:
        time.sleep(random.uniform(10.0, 20.0))

def sha1_of(s: str) -> str:
    try:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
    except Exception:
        return "0"*16

def slugify(s: str) -> str:
    s=(s or "").strip().lower()
    s=re.sub(r"[^a-z0-9\-_. ]+","",s); s=re.sub(r"\s+","-",s); s=re.sub(r"-{2,}","-",s).strip("-")
    return s[:80]

def _write_full_html(site_key: str, seq: int, url: str, title: str|None, html: str) -> str:
    """
    Save full HTML directly under the site's folder (no run subfolder):
      data/outputs/articles/html/full/<site_key>/00001_sha16_slug.html
    Returns relative path from PROJECT_ROOT as str.
    """
    out_dir = HTML_FULL_DIR / site_key
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{seq:05d}_{sha1_of(url)}"; slug = slugify(title or "")
    fname = f"{base}_{slug}.html" if slug else f"{base}.html"
    path = out_dir / fname
    path.write_text(html or "", encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT))

def _looks_like_html(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    probe = s.lstrip()[:2048].lower()
    return any(tok in probe for tok in ("<!doctype", "<html", "<head", "<body", "</"))

def _looks_like_interstitial(html: str) -> bool:
    low = (html or "").lower()
    cues = (
        "enable javascript", "just a moment", "cloudflare",
        "captcha", "are you a robot", "access denied",
        "consent", "cookie", "gdpr", "onetrust", "quantcast", "didomi",
    )
    return any(c in low for c in cues)

def _html_is_usable(html: str, site_cfg: dict) -> bool:
    """Return True only if the HTML is not an interstitial and contains a likely article container."""
    if not html or len(html) < 800 or not _looks_like_html(html):
        return False
    # check presence of configured containers from any enabled template
    main_css = body_css = ""
    for t in (site_cfg.get("templates") or []):
        if not t.get("enabled", True):
            continue
        ct = ((t.get("extract") or {}).get("container_text") or {})
        main_css = (ct.get("main_container_css") or "").strip() or main_css
        body_css = (ct.get("body_container_css") or "").strip() or body_css
        if main_css and body_css:
            break
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        root = soup.select_one(main_css) if main_css else soup
        if body_css:
            node = root.select_one(body_css) if root else None
        else:
            node = root
        has_body = node is not None and node.get_text(strip=True)
    except Exception:
        has_body = False
    if not has_body:
        return False
    if _looks_like_interstitial(html):
        return False
    return True

def _effective_consent_xpaths(site_cfg: dict, tpl_name: str | None) -> list[str]:
    """
    Combine site-level and template-level consent XPaths, mirroring the Builder page.
    """
    fetch_cfg = site_cfg.get("fetch", {}) or {}
    consent_cfg = fetch_cfg.get("consent", {}) or {}

    site_xp = consent_cfg.get("xpaths", []) or []

    tpl_xp: list[str] = []
    if tpl_name:
        for t in site_cfg.get("templates", []) or []:
            if t.get("name") == tpl_name:
                tpl_xp = (
                    (t.get("fetch") or {})
                    .get("consent", {})
                    .get("xpaths", [])
                ) or []
                break

    # Normalize to clean strings
    all_xp = [x for x in list(site_xp) + list(tpl_xp) if isinstance(x, str) and x.strip()]
    return all_xp

def _css_hints_from_site_cfg(site_cfg: dict) -> tuple[str, str]:
    """
    Return (main_container_css, body_container_css) hints to help the fetcher decide when to fallback to Selenium.
    Picks the first non-empty from enabled templates.
    """
    main_css = body_css = ""
    for t in (site_cfg.get("templates") or []):
        if not t.get("enabled", True):
            continue
        ct = ((t.get("extract") or {}).get("container_text") or {})
        if not main_css:
            main_css = (ct.get("main_container_css") or "").strip()
        if not body_css:
            body_css = (ct.get("body_container_css") or "").strip()
        if main_css and body_css:
            break
    return main_css, body_css

def _templates_matching_url(site_cfg: dict, url: str) -> list[dict]:
    """Return enabled templates considered applicable to this URL, using optional url_patterns (glob) or url_regex."""
    matched = []
    for t in (site_cfg.get("templates") or []):
        if not t.get("enabled", True):
            continue
        pats = (t.get("url_patterns") or [])
        rgx  = (t.get("url_regex") or "").strip()
        ok = False
        if pats:
            for p in pats:
                if _fnm.fnmatch(url, p):
                    ok = True
                    break
        if rgx and not ok:
            try:
                if re.search(rgx, url):
                    ok = True
            except Exception:
                pass
        if not pats and not rgx:
            ok = True  # no constraints → applicable everywhere
        if ok:
            matched.append(t)
    return matched

def _prefers_selenium(site_cfg: dict, url: str) -> tuple[bool, str, float]:
    """
    Decide if this URL prefers JS (Selenium) engine.
    Mirrors the Builder page’s logic: site.fetch.requires_js + optional template overrides.
    Returns: (force_js, wait_css, wait_timeout)
    """
    # Site-wide defaults
    site_fetch = site_cfg.get("fetch", {}) or {}
    site_requires_js = bool(site_fetch.get("requires_js", False))
    site_wait_css = site_fetch.get("wait_css") or ""
    site_wait_timeout = float(site_fetch.get("wait_timeout") or 20)

    # Template-level overrides (first matching template wins)
    tpl_wait_css: str = ""
    tpl_wait_timeout: float | None = None
    tpl_requires_js: bool | None = None

    for t in _templates_matching_url(site_cfg, url):
        tf = t.get("fetch", {}) or {}
        tpl_requires_js = tf.get("requires_js")
        tpl_wait_css = tf.get("wait_css") or ""
        tpl_wait_timeout = tf.get("wait_timeout")
        break  # first match only, same as Builder

    requires_js = tpl_requires_js if tpl_requires_js is not None else site_requires_js
    wait_css = tpl_wait_css or site_wait_css
    wait_timeout = float(tpl_wait_timeout or site_wait_timeout or 20)

    return bool(requires_js), wait_css, wait_timeout

def _load_site_yaml(site_key: str) -> dict:
    yml = SCRAPERS_DIR / f"{site_key}.yml"
    if not yml.exists(): return {"site": site_key, "templates": []}
    try: return yaml.safe_load(yml.read_text(encoding="utf-8")) or {"site": site_key, "templates": []}
    except Exception: return {"site": site_key, "templates": []}

def _build_site_cfg_for_single_template(site_cfg: dict, tpl: dict) -> dict:
    return {"site": site_cfg.get("site"), "templates": [tpl]}

def _accepts_best(out: dict, accept_cfg: dict|None):
    # Only title is required for acceptance metrics; body length is used just for scoring.
    missing=[]
    if not (out.get("title") or "").strip(): missing.append("title")
    min_chars = int((accept_cfg or {}).get("min_body_chars") or 400)
#    body_len = len((out.get("text") or "").strip())
    # look at 'body' first, then 'text'
    body_src = (out.get("body") or out.get("text") or "") or ""
    body_len = len(body_src.strip())
    ok = len(missing)==0
    return ok, missing, min_chars, body_len

def _extract_best(url: str, html: str, site_cfg: dict):
    """
    Try all enabled templates, pick the best (fewest missing among core fields,
    then not brief, then longest body, then lowest priority).

    IMPORTANT:
    Uses conty_core.extract.extract_and_postprocess so that Builder and Runner
    share the exact same extraction + postprocessing pipeline
    (including datetime normalization and *_raw / *_datetime fields).
    If extract_and_postprocess returns no body/text, we fall back to the same
    body extraction logic used by Builder's Matrix (body_config + container_text).
    """
    templates = [t for t in (site_cfg.get("templates") or []) if t.get("enabled", True)]
    if not templates:
        return {}, "", "fail", ["no templates"], 400, 0

    scored = []

    for t in templates:
        try:
            # Same site_cfg-per-template convention as before
            tpl_site_cfg = _build_site_cfg_for_single_template(site_cfg, t)
            data = extract_and_postprocess(
                url=url,
                html=html,
                site_cfg=tpl_site_cfg,
                forced_template=None,
                lenient=False,
            ) or {}
        except Exception:
            data = {}

        # Ensure template name present
        if "template_used" not in data:
            data["template_used"] = t.get("name", "")

        # --- Builder-style body: always prefer container_text/body_config like Builder ---
        fb_body, fb_len = _fallback_body_from_template(html, t)
        if fb_len:
            # Override any conty_core body/text with the same logic Builder uses
            data["body"] = fb_body

        # Acceptance / scoring
        ok, _missing, _min_chars, _body_len = _accepts_best(
            data,
            (t.get("extract") or {}).get("accept"),
        )

        pref = ["title", "published_time", "author", "lead", "main_image", "tags", "subtitle"]
        miss_cnt = sum(1 for k in pref if not str(data.get(k) or "").strip())
        brief_pen = 1 if _body_len < _min_chars else 0
        neg_body = -_body_len
        prio = int(t.get("priority", 10))
        score = (miss_cnt, brief_pen, neg_body, prio)

        scored.append((score, t, data, _missing, _min_chars, _body_len, ok))

    ok_ones = [s for s in scored if s[-1] is True]
    if not scored:
        return {}, "", "fail", ["no templates"], 400, 0

    best = min(ok_ones or scored, key=lambda x: x[0])

    _, best_tpl, best_out, best_missing, best_min, best_body, best_ok = best
    status = "ok" if best_ok else "best-effort"

    if "status" not in best_out:
        best_out["status"] = status
    if "template_used" not in best_out:
        best_out["template_used"] = best_tpl.get("name", "")

    return best_out, best_tpl.get("name", ""), status, best_missing, best_min, best_body

def _fix_failures_for_file(
    site_key: str,
    failures_csv: Path,
    save_full_html: bool = True,
    ignore_saved_html: bool = False
) -> tuple[int,int,int]:

    """
    Re-scrape failures listed in failures_csv for a given site.
    Preference order: use saved full_html_path if exists, otherwise refetch.
    Returns: (processed, fixed, remaining)
    """
    import pandas as pd
    from datetime import datetime, timezone

    site_cfg = _load_site_yaml(site_key)
    df_fail = pd.read_csv(failures_csv)
    if df_fail.empty or "url" not in df_fail.columns:
        return (0,0,0)

    session = make_session()
    html_cache = {}

    # infer teasers basename -> articles csv path
    teasers_csv = df_fail["teasers_csv"].dropna().astype(str).iloc[0] if "teasers_csv" in df_fail.columns and not df_fail["teasers_csv"].dropna().empty else ""
    base_no_ext = Path(teasers_csv).stem if teasers_csv else failures_csv.name.replace("__failures.csv","")
    out_csv = ARTICLES_DIR / f"{base_no_ext}_articles.csv"

    # load existing articles for merging
    try:
        df_prev = pd.read_csv(out_csv)
    except Exception:
        df_prev = pd.DataFrame(columns=FIELDNAMES)

    processed = fixed = 0
    rows_new = []
    remaining = []
    softfails = []

    for _, row in df_fail.iterrows():
        url = str(row.get("url","")).strip()
        if not url: continue
        http_status = None  # fetch_with_fallback abstracts status - keep for failures CSV
        html = ""
        full_path = str(row.get("full_html_path") or "").strip()
        final_url_local = url

        # 1) Try disk first (ONLY if not ignoring and not forcing Selenium)
        if full_path and not ignore_saved_html and not st.session_state.get("force_js", False):
            if not full_path.startswith(("http://", "https://")):
                try:
                    p = Path(full_path)
                    if not p.is_absolute():
                        p = PROJECT_ROOT / p
                    if p.exists():
                        html = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    html = ""

        # 2) Decide if we must refetch
        needs_refetch = (
            bool(st.session_state.get("force_js", False)) or
            bool(ignore_saved_html) or
            not _html_is_usable(html, site_cfg)
        )

        if needs_refetch:
            try:
                main_hint, body_hint = _css_hints_from_site_cfg(site_cfg)
                prefer_js, tpl_wait_css, tpl_wait_to = _prefers_selenium(site_cfg, url)
                force_js = prefer_js or bool(st.session_state.get("force_js", False))
                wait_to  = float(tpl_wait_to or st.session_state.get("selenium_wait_timeout", 20.0) or 20)

                consent_xpaths = []
                if st.session_state.get("use_js_fallback", True) or force_js:
                    consent_xpaths = (
                        list(KNOWN_CONSENT_XPATHS)
                        + _effective_consent_xpaths(site_cfg, None)
                    )
                else:
                    consent_xpaths = _effective_consent_xpaths(site_cfg, None)

                container_css = tpl_wait_css or main_hint

                res: FetchResult = fetch_with_fallback(
                    url,
                    container_css=container_css,
                    item_css=body_hint,
                    force_js=force_js,
                    wait_timeout=wait_to,
                    consent_click_xpaths=consent_xpaths or None,
                )

                html = res.html or ""
                final_url_local = res.final_url or url
                if save_full_html and html:
                    full_path = _write_full_html(site_key, 1, final_url_local, "", html)

            except Exception:
                html = ""

        processed += 1

        # only bail out if HTML is not fetched at all.
        if not html or not _looks_like_html(html):
            remaining.append({
                "url": url,
                "failure_type": "fetch_error",
                "full_html_path": "",
                "teasers_csv": teasers_csv,
                "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "template_used": "",
                "http_status": http_status,
            })
            _polite_sleep(processed)
            continue

        # run best template extraction (same logic you use during scraping)
        try:
            out, tpl_used, status, missing, min_chars, body_len = _extract_best(url, html, site_cfg)
        except Exception:
            remaining.append({
                "url": url,
                "failure_type": "extract_error",
                "full_html_path": full_path,
                "teasers_csv": teasers_csv,
                "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "template_used": "",
                "http_status": http_status
            })
            _polite_sleep(processed)
            continue

        # title still mandatory
        title = (out.get("title") or "").strip()
        if not title:
            remaining.append({
                "url": url,
                "failure_type": "no_title",
                "full_html_path": full_path,
                "teasers_csv": teasers_csv,
                "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "template_used": tpl_used or "",
                "http_status": http_status
            })
            _polite_sleep(processed)
            continue

        # Normalize dates (write-time) + soft reasons
        pub_iso, pub_date, upd_iso, upd_date, soft = _normalize_dates_for_output(
    out, min_chars, body_len, site_key, url
)

        if soft:
            softfails.append({
                "url": url,
                "failure_type": ";".join(soft),
                "full_html_path": full_path,
                "teasers_csv": teasers_csv,
                "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "template_used": tpl_used or "",
                "http_status": http_status,
            })

        rows_new.append({
            "site": site_key, "url": url,
            "final_url": final_url_local,
            "title": title,
            "subtitle": out.get("subtitle") or "",
            "section": out.get("section") or "",
            "author": out.get("author") or "",
            "published_raw": (
                out.get("published_raw")
                or out.get("published_time")
                or out.get("published")
                or ""
            ),
            "published_datetime": pub_iso,
            "published_date": pub_date,
            "updated_raw": (
                out.get("updated_raw")
                or out.get("updated_time")
                or out.get("updated")
                or ""
            ),
            "updated_datetime": upd_iso,
            "updated_date": upd_date,
            "lead": out.get("lead") or "", 
#            "text": out.get("text") or "",
            "text": (out.get("body") or out.get("text") or ""),
            "main_image": out.get("main_image") or "", "tags": out.get("tags") or "",
            "status": status,
            "reason": "; ".join(filter(None, [
                "; ".join(missing) if (status != 'ok' and missing) else "",
                f"body_chars<{min_chars}" if (status != 'ok' and body_len < min_chars) else "",
                ";".join(soft) if soft else ""
            ])),
            "template_used": tpl_used or "",
            "fetched_datetime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "full_html_path": full_path
        })
        fixed += 1

        # polite delay
        _polite_sleep(processed)

    # merge to articles csv
    if rows_new:
        df_add = pd.DataFrame(rows_new)
        for col in FIELDNAMES:
            if col not in df_add.columns:
                df_add[col] = ""
        df_add = df_add[FIELDNAMES]

        # If there are existing rows, drop and clean up those we are replacing
        if not df_prev.empty:
            mask_replaced = df_prev["url"].isin(df_add["url"])
            df_replaced = df_prev[mask_replaced]

            # Delete old HTML files for replaced rows
            if "full_html_path" in df_replaced.columns:
                for html_path in df_replaced["full_html_path"].dropna().unique():
                    if not html_path or not isinstance(html_path, str):
                        continue
                    try:
                        p = Path(html_path)
                        if not p.is_absolute():
                            p = PROJECT_ROOT / p
                        if p.exists():
                            p.unlink()
                    except OSError:
                        # ignore deletion failures
                        pass

            # Keep only rows that are NOT being replaced
            df_prev = df_prev[~mask_replaced]

        # Append new rows; since old ones are removed, last-fetched wins
        df_out = pd.concat([df_prev, df_add], ignore_index=True)
        df_out.drop_duplicates(subset=["url"], keep="last", inplace=True)
        df_out.to_csv(out_csv, index=False, encoding="utf-8")

    # rewrite or remove failures.csv
    if remaining:
        pd.DataFrame(remaining, columns=["url","failure_type","full_html_path","teasers_csv","when","template_used","http_status"]).to_csv(
            failures_csv, index=False, encoding="utf-8"
        )
    else:
        failures_csv.unlink(missing_ok=True)

    # Soft failures sidecar for the same base
    site_fail_dir = RUN_LOGS_DIR / site_key
    base_no_ext = Path(teasers_csv).stem if teasers_csv else failures_csv.name.replace("__failures.csv","")
    soft_path = site_fail_dir / f"{base_no_ext}__softfailures.csv"
    if softfails:
        pd.DataFrame(
            softfails,
            columns=["url","failure_type","full_html_path","teasers_csv","when","template_used","http_status"]
        ).to_csv(soft_path, index=False, encoding="utf-8")
    else:
        soft_path.unlink(missing_ok=True)


    return processed, fixed, (len(remaining))

# ---- Datetime normalization helpers ----------------------------------------
ISO_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?$")

def _to_iso_naive(s: str) -> str:
    """Parse any datetime string to tz-naive ISO (UTC equivalent) used in CSVs. Returns '' on failure."""
    if not s or not isinstance(s, str):
        return ""
    try:
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            return ""
        # Convert to UTC and drop tz
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert("UTC")
        dt = ts.to_pydatetime().replace(tzinfo=None)
        # If looks like date-only (no 'T' originally and time is 00:00), keep date
        fmt = "%Y-%m-%d"
        if "T" in str(s) or (dt.hour or dt.minute or dt.second):
            fmt = "%Y-%m-%dT%H:%M" if dt.second == 0 else "%Y-%m-%dT%H:%M:%S"
        return dt.strftime(fmt)
    except Exception:
        return ""

HTML_TAG_RE = re.compile(r"<[^>]+>")

def _strip_html(s: str) -> str:
    """Remove HTML tags from a string for datetime parsing."""
    if not s or not isinstance(s, str):
        return ""
    return HTML_TAG_RE.sub("", s).strip()

def _normalize_dates_for_output(
    out: dict,
    min_chars: int,
    body_len: int,
    site_key_for_dt: str | None,
    url_for_log: str | None,
) -> tuple[str, str, str, str, list]:
    """
    Thin wrapper over the shared postprocess outputs.

    extract_and_postprocess(...) has already:
      - run postprocess_article_obj / postprocess_article_dict
      - filled published_time / updated_time with normalized strings (or None)
      - created published_datetime / updated_datetime / published_date / updated_date.

    Here we only:
      - pick those fields
      - derive dates if missing
      - compute soft-failure reasons.
    """

    # Normalized datetimes as produced by conty_core.extract.extract_and_postprocess
    pub_iso = (out.get("published_datetime") or out.get("published_time") or "").strip()
    upd_iso = (out.get("updated_datetime") or out.get("updated_time") or "").strip()

    # Date-only values – prefer explicit fields if present
    pub_date = (out.get("published_date") or "").strip()
    if not pub_date and pub_iso and len(pub_iso) >= 10:
        pub_date = pub_iso[:10]

    upd_date = (out.get("updated_date") or "").strip()
    if not upd_date and upd_iso and len(upd_iso) >= 10:
        upd_date = upd_iso[:10]

    # Raw strings (already cleaned in postprocess_article_dict)
    pub_raw = (out.get("published_raw") or "").strip()
    upd_raw = (out.get("updated_raw") or "").strip()

    # Soft reasons for __softfailures.csv
    soft: list[str] = []
    if not (pub_iso or upd_iso):
        soft.append("missing_both_dt")
    if pub_raw and not pub_iso:
        soft.append("parse_failed_pub")
    if upd_raw and not upd_iso:
        soft.append("parse_failed_upd")
    if body_len < int(min_chars or 400):
        soft.append("short_body")

    return pub_iso, pub_date, upd_iso, upd_date, soft

def _fallback_body_from_template(html: str, tpl: dict) -> tuple[str, int]:
    """
    Builder-style body extraction used as a fallback:
    - uses extract.body_config.main_container_css / body_container_css
    - applies extract.container_text.exclude and boilerplate_regex
    """
    rules = (tpl.get("extract") or {}) or {}
    body_cfg = rules.get("body_config") or {}
    ct_cfg = rules.get("container_text") or {}

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
        if node is not None:
            root = node

    body_node = root
    if body_css:
        node = root.select_one(body_css)
        if node is not None:
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
    return body_text, len(body_text)


# ----------- UI ------------------
st.title("Run scrapes (articles)")

# --- Page intro / quick help -------------------------------------------------
with st.expander("How this page works (quick guide)", expanded=False):
    st.markdown(
        """
**Goal:** Convert teaser CSVs into article rows (title, body, dates, etc.), with **normalized datetimes** and optional **full HTML** snapshots.

### 1) Pick a mode
- **Single site** → choose **one** site and manually pick its teaser CSVs.
- **Multi-site** → choose **several** sites; for each site, teaser CSVs are **auto-matched** by filename prefix: `<site>_*.csv`.

### 2) Configure shared options
- **URL column** → usually **<auto-detect>**.
- **Max URLs per CSV** → per-file cap (0 = no limit).
- **Save Full HTML** → store page HTML snapshots for reproducible debugging & View Data previews.

### 3) Run
The app fetches (Requests → Selenium when needed), picks the **best template**, extracts fields, **normalizes dates**, and writes:
- **`articles/`**: `<basename>_articles.csv`  
  Includes `published_datetime` / `updated_datetime` (ISO), derived `*_date`, and keeps raw inputs in `*_raw`.
- **`run_logs/<site>/`**:  
  - `*__failures.csv` → **Hard failures** (no acceptable row yet; e.g., `fetch_error`, `extract_error`, `no_title`).  
  - `*__softfailures.csv` → **Soft issues** (row written but needs attention; e.g., `missing_both_dt`, `parse_failed_pub`, `parse_failed_upd`, `short_body`).

### 4) After a run — what next?
- **Targeted re-scrape (replace rows)** *(Single-site only)*  
  Refresh a **specific subset** of URLs and **replace** their rows in the chosen `_articles.csv`.  
  Great for rows that exist but have wrong text/dates. Can prefer saved HTML.
- **Fix failures (re-scrape)**  
  Recover **missing rows** listed in `__failures.csv`. Tries saved HTML first; otherwise refetch.
- **View Data**  
  Filter/sort to locate silent issues; preview saved HTML; send problem URLs to **Build a Scraper → Matrix test**.

### Tips
- **Short bodies** usually mean a **container mismatch** or **JS-only content**; adjust template selectors or mark `requires_js`.
- If both `published` and `updated` are missing, review the **datetime selectors** or per-section layout differences.
- Prefer saved HTML when debugging to avoid layout drift and network noise.
        """
    )

show_demo_banner()

# --- SELECT WHAT TO SCRAPE ----------------------------------------------------
st.subheader("Select what to scrape")

# Explicit mode choice
mode = st.radio(
    "Scrape mode",
    ["Single site", "Multi-site"],
    horizontal=True,
    help="Choose whether to scrape one site or multiple sites sequentially."
)
# Hint for multi-site behavior
if mode == "Multi-site":
    st.info("In Multi-site mode, teaser CSVs are auto-matched per site by filename prefix: `<site>_*.csv`.", icon="ℹ️")

# Prepare site list once
site_files = sorted([p for p in SCRAPERS_DIR.glob("*.yml")])
site_options = [p.stem for p in site_files]

# Holders used later
sites_selected: list[str] = []
selected_csv_names: list[str] = []

if mode == "Single site":
    # --- Single site branch ---------------------------------------------------
    site_key = st.selectbox(
        "Site (YAML in data/scrapers/)",
        options=site_options,
        help="Pick one site to run using its YAML template."
    ) if site_options else st.text_input("Site key")

    # Load YAML now so engine defaults can be seeded (single-site only)
    site_cfg = _load_site_yaml(site_key) if site_key else {}

    # --- Seed engine options from site YAML (runs when site changes; single-site only) ---
    if site_key:
        sf = (site_cfg.get("fetch") or {})
        if st.session_state.get("__engine_seeded_for_run") != site_key:
            # Defaults from YAML (with fallbacks)
            default_engine = (sf.get("default_engine") or "requests").strip().lower()
            allow_fb = bool(sf.get("allow_js_fallback", True))
            wait_to  = float(sf.get("wait_timeout", 20) or 20)

            # Seed session state (no widget value= anywhere)
            st.session_state["use_js_fallback"] = allow_fb
            st.session_state["force_js"] = (default_engine == "selenium")
            st.session_state["selenium_wait_timeout"] = wait_to

            st.session_state["__engine_seeded_for_run"] = site_key

    # Teasers CSVs (manual pick)
    csv_dir_str = st.text_input(
        "Folder with teasers CSV files",
#        value=str(OUTPUTS_DIR),
        value=str(DEMO_TEASERS_DIR if DEMO_MODE else OUTPUTS_DIR),
        help="Folder that contains the teaser CSVs (e.g., outputs with lists of article URLs)."
    )
    csv_dir = Path(csv_dir_str).expanduser().resolve()

    if DEMO_MODE:
        # Force demo teasers folder regardless of what the user types
        csv_dir = DEMO_TEASERS_DIR
        st.caption("Demo mode: using teaser CSVs from `data/demo/teasers/`.")

    show_only_site_files = st.checkbox(
        f'Show only teasers CSVs starting with "{site_key}"',
        value=True,
        help="If ON, limits the list to CSVs that start with the current site key."
    ) if site_key else False

    all_csvs = sorted([p for p in csv_dir.glob("*.csv")])
    site_prefix = f"{(site_key or '').strip()}_".lower()
    filtered_csvs = [p for p in all_csvs if p.name.lower().startswith(site_prefix)] if site_key else all_csvs
    csv_pool = filtered_csvs if show_only_site_files else all_csvs
    csv_options = [p.name for p in csv_pool]

    selected_csv_names = st.multiselect(
        "Teasers CSV files to process",
        options=csv_options,
        default=csv_options,
        help="Pick one or more teaser CSVs to convert into article rows."
    )

    sites_selected = [site_key] if site_key else []

else:
    # --- Multi-site branch ----------------------------------------------------
    st.info("Multi-site mode runs sites sequentially with per-site progress and summaries.")
    sites_selected = st.multiselect(
        "Sites to process",
        options=site_options,
        default=site_options,
        help="Pick which sites to run. For each site, teaser CSVs are auto-matched by `<site>_` prefix at run time."
    )

    # CSV folder for all sites (we auto-match by site prefix later)
    csv_dir_str = st.text_input(
        "Folder with teasers CSV files",
#        value=str(OUTPUTS_DIR),
        value=str(DEMO_TEASERS_DIR if DEMO_MODE else OUTPUTS_DIR),
        help="Folder that contains the teaser CSVs for all sites."
    )
    csv_dir = Path(csv_dir_str).expanduser().resolve()

    if DEMO_MODE:
        csv_dir = DEMO_TEASERS_DIR
        st.caption("Demo mode: using teaser CSVs from `data/demo/teasers/`.")

# Shared controls (apply to both modes)
# URL column select (simple)
url_col = st.selectbox(
    "URL column",
    options=["<auto-detect>", "url", "final_url", "link"],
    index=0,
    help="Column in teasers CSVs that contains article URLs. Leave on <auto-detect> if unsure."
)

save_full_html = st.checkbox(
    "Save Full HTML",
    value=True,
    key="save_full_html",
    help="Stores a snapshot of the full page HTML to disk for debugging and View Data previews."
)

limit_per_file = st.number_input(
    "Max URLs per CSV (0 = all)",
    min_value=0, max_value=500000, value=0,
    help="For large files, set a cap to run a smaller test batch. Applied to each CSV separately."
)


def _run_for_site(site_key_local: str, csv_dir: Path, selected_csv_names: list[str], url_col: str, limit_per_file: int, save_full_html: bool):
    site_cfg_local = _load_site_yaml(site_key_local)
    if not (site_cfg_local.get("templates") or []):
        st.error(f"[{site_key_local}] No templates found for this site."); return (0,0,0,0)

    csv_files = [csv_dir / name for name in selected_csv_names]
    if not csv_files:
        st.warning(f"[{site_key_local}] No CSVs selected in {csv_dir}"); return (0,0,0,0)

    rows_total = hard_total = soft_total = files_done = 0

    pb_files = st.progress(0.0, text=f"[{site_key_local}] Starting…")
    status_files = st.empty()
    pb_urls = st.progress(0.0)
    status_urls = st.empty()
    total_files = len(csv_files)

    for file_idx, csv_path in enumerate(csv_files, start=1):
        status_files.write(f"[{site_key_local}] File {file_idx}/{total_files} — {csv_path.name}")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"[{site_key_local}] Failed reading {csv_path.name}: {e}")
            pb_files.progress(file_idx/total_files, text=f"[{site_key_local}] Skipped {csv_path.name}")
            continue

        base_no_ext = csv_path.stem
        out_csv = ARTICLES_DIR / f"{base_no_ext}_articles.csv"
        st.info(f"[{site_key_local}] Output CSV → {out_csv}")
        st.caption(f"↳ writing to: `{out_csv}`")
        st.caption("Hard failures → **__failures.csv**, soft issues → **__softfailures.csv**.")

        # detect URL column (prefer 'url' > 'link' > 'final_url' > any '*url*')
        col = None if url_col == "<auto-detect>" else url_col
        if not col or col not in df.columns:
            lower_map = {str(c).lower(): c for c in df.columns}
            if "url" in lower_map:
                col = lower_map["url"]
            elif "link" in lower_map:
                col = lower_map["link"]
            elif "final_url" in lower_map:
                col = lower_map["final_url"]
            else:
                for c in df.columns:
                    if "url" in str(c).lower():
                        col = c
                        break

        if not col:
            st.warning(f"[{site_key_local}] Skip {csv_path.name}: could not detect URL column.")
            pb_files.progress(file_idx/total_files, text=f"[{site_key_local}] Skipped {csv_path.name} (no URL col)")
            continue

        st.caption(f"Using teaser URL column: **{col}**")

        # counts and batch
        # Raw teaser URLs (may contain duplicates)
        urls_all = [str(u).strip() for u in df[col].tolist() if str(u).strip().startswith("http")]
        total_in_csv = len(urls_all)

        # Unique teaser URLs (preserve order)
        unique_teaser_urls = list(dict.fromkeys(urls_all))
        total_unique_in_csv = len(unique_teaser_urls)

        # Keep only URLs that belong to THIS site (by hostname)
        unique_site_urls: list[str] = []
        skipped_other = 0
        allowed_fragment = (site_key_local or "").lower()

        for u in unique_teaser_urls:
            host = ""
            try:
                host = urlparse(u).netloc.lower()
            except Exception:
                pass

            if allowed_fragment and allowed_fragment in host:
                unique_site_urls.append(u)
            else:
                skipped_other += 1

        if skipped_other:
            st.caption(
                f"[{site_key_local}] Skipping {skipped_other} URL(s) from {csv_path.name} "
                f"with host not containing '{allowed_fragment}'."
            )

        # URLs already present in this file's output CSV
        already = set()
        if out_csv.exists():
            try:
                df_prev = pd.read_csv(out_csv, dtype={"url": str}, usecols=["url"])
                prev_urls = (
                    df_prev["url"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                already = set(prev_urls.tolist())
                st.caption(f"Existing rows in `{out_csv.name}`: **{len(df_prev):,}** (unique url: **{prev_urls.nunique():,}**)")
            except Exception as e:
                st.warning(f"Could not read existing URLs from {out_csv.name}: {e}")

        # Intersection for THIS teaser file (not the whole articles file)
        already_in_this_csv = sum(1 for u in unique_site_urls if u in already)

        # Remaining = unique teaser URLs not already scraped
        remaining_pool = [u for u in unique_site_urls if u not in already]

        # Apply cap
        urls = remaining_pool[: int(limit_per_file)] if (limit_per_file and limit_per_file > 0) else remaining_pool

        # Clear, consistent caption
        st.caption(
            f"[{site_key_local}] **{csv_path.name}** — "
            f"in teasers: **{total_in_csv}** (unique: {total_unique_in_csv}) • "
            f"matching this site: {len(unique_site_urls)}, skipped: {skipped_other}) • "
            f"already scraped (in this CSV): **{already_in_this_csv}** • "
            f"remaining: **{len(remaining_pool)}** • will scrape now: **{len(urls)}**"
        )

        # per-file outputs
        rows_new=[]; seq=0
        failures=[]; softfails=[]
        total_urls = len(urls) or 1

        for j, url in enumerate(urls, start=1):
            pb_urls.progress(j/total_urls)
            status_urls.write(f"[{site_key_local}] URL {j}/{total_urls} from {csv_path.name}")
            seq += 1

            # fetch
            http_status = None
            try:
                main_hint, body_hint = _css_hints_from_site_cfg(site_cfg_local)
                prefer_js, tpl_wait_css, tpl_wait_to = _prefers_selenium(site_cfg_local, url)
                force_js = prefer_js or bool(st.session_state.get("force_js", False))
                wait_to  = float(tpl_wait_to or st.session_state.get("selenium_wait_timeout", 20.0) or 20)

                # Combine built-in consent XPaths with site-level ones (template-name unknown at fetch time)
                consent_xpaths = []
                if st.session_state.get("use_js_fallback", True) or force_js:
                    consent_xpaths = (
                        list(KNOWN_CONSENT_XPATHS)
                        + _effective_consent_xpaths(site_cfg_local, None)
                    )
                else:
                    consent_xpaths = _effective_consent_xpaths(site_cfg_local, None)

                # Prefer explicit wait_css from fetch config; fall back to main_hint if empty
                container_css = tpl_wait_css or main_hint

                res: FetchResult = fetch_with_fallback(
                    url,
                    container_css=container_css,
                    item_css=body_hint,
                    force_js=force_js,
                    wait_timeout=wait_to,
                    consent_click_xpaths=consent_xpaths or None,
                )
                final_url = res.final_url
                html = res.html or ""
                fetched_ok = bool(html)
            except Exception:
                fetched_ok = False; html=""; final_url = url

            if not fetched_ok:
                failures.append({
                    "url": url, "final_url": final_url, "failure_type": "fetch_error", "full_html_path": "",
                    "teasers_csv": csv_path.name, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "template_used": "", "http_status": http_status,
                })
                _polite_sleep(j); continue

            # save full html
            full_path = ""
            if save_full_html:
                try: full_path = _write_full_html(site_key_local, seq, final_url, "", html)
                except Exception: full_path = ""

            # extract best
            try:
                out, tpl_used, status, missing, min_chars, body_len = _extract_best(url, html, site_cfg_local)
            except Exception as e:
                failures.append({
                    "url":url, "final_url": final_url, "failure_type": "extract_error", "full_html_path": full_path,
                    "teasers_csv": csv_path.name, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "template_used": "", "http_status": http_status
                })
                rows_new.append({
                    "site":site_key_local, "url":url, "final_url":final_url, "title":"", "subtitle":"", "section":"", "author":"",
                    "published_datetime":"","published_date":"","updated_datetime":"","updated_date":"",
                    "lead":"","text":"","main_image":"","tags":"",
                    "status":"extract_error","reason": str(e),"template_used":"", "fetched_datetime":datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "full_html_path":full_path
                }); _polite_sleep(j); continue

            brief = body_len < min_chars
            reason = "; ".join(filter(None, [
                f"missing: {', '.join(missing)}" if (status!='ok' and missing) else "",
                f"body_chars<{min_chars}" if (status!='ok' and brief) else ""
            ]))

            if not (out.get("title") or "").strip():
                failures.append({
                    "url":url, "final_url": final_url, "failure_type": "no_title", "full_html_path": full_path,
                    "teasers_csv": csv_path.name, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "template_used": tpl_used or "", "http_status": http_status,
                })
                pub_iso, pub_date, upd_iso, upd_date, soft = _normalize_dates_for_output(
    out, min_chars, body_len, site_key_local, url
)
                if soft:
                    softfails.append({
                        "url": url, "final_url": final_url, "failure_type": ";".join(soft),
                        "full_html_path": full_path, "teasers_csv": csv_path.name,
                        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "template_used": tpl_used or "", "http_status": http_status,
                    })
                rows_new.append({
                    "site":site_key_local,"url":url,"final_url": final_url,
                    "title":"", "subtitle": out.get("subtitle") or "", "section": out.get("section") or "",
                    "author": out.get("author") or "",
                    "published_raw": (
                        out.get("published_raw")
                        or out.get("published_time")
                        or out.get("published")
                        or ""
                    ),
                    "published_datetime": pub_iso,
                    "published_date": pub_date,
                    "updated_raw": (
                        out.get("updated_raw")
                        or out.get("updated_time")
                        or out.get("updated")
                        or ""
                    ),
                    "updated_datetime": upd_iso,
                    "updated_date": upd_date,

                    "lead": out.get("lead") or "", 
#                    "text": out.get("text") or "",
                    "text": (out.get("body") or out.get("text") or ""),
                    "main_image": out.get("main_image") or "", "tags": out.get("tags") or "",
                    "status":"no_title",
                    "reason": "; ".join(filter(None, [reason, ";".join(soft) if soft else ""])),
                    "template_used": tpl_used or "",
                    "fetched_datetime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "full_html_path": full_path
                }); _polite_sleep(j); continue

            pub_iso, pub_date, upd_iso, upd_date, soft = _normalize_dates_for_output(
    out, min_chars, body_len, site_key_local, url
)
            if soft:
                softfails.append({
                    "url":url, "final_url":final_url, "failure_type": ";".join(soft),
                    "full_html_path": full_path, "teasers_csv": csv_path.name,
                    "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "template_used": tpl_used or "", "http_status": http_status,
                })

            rows_new.append({
                "site":site_key_local,"url":url,"final_url": final_url,
                "title": out.get("title") or "",
                "subtitle": out.get("subtitle") or "",
                "section": out.get("section") or "",
                "author": out.get("author") or "",
                "published_raw": (
                    out.get("published_raw")
                    or out.get("published_time")
                    or out.get("published")
                    or ""
                ),
                "published_datetime": pub_iso,
                "published_date": pub_date,
                "updated_raw": (
                    out.get("updated_raw")
                    or out.get("updated_time")
                    or out.get("updated")
                    or ""
                ),
                "updated_datetime": upd_iso,
                "updated_date": upd_date,

                "lead": out.get("lead") or "", 
#                "text": out.get("text") or "",
                "text": (out.get("body") or out.get("text") or ""),
                "main_image": out.get("main_image") or "", "tags": out.get("tags") or "",
                "status": status,
                "reason": "; ".join(filter(None, [reason, ";".join(soft) if soft else ""])),
                "template_used": tpl_used or "",
                "fetched_datetime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "full_html_path": full_path
            }); _polite_sleep(j)

        # write articles
        df_new = pd.DataFrame(rows_new)
        for col in FIELDNAMES:
            if col not in df_new.columns: df_new[col] = ""
        df_new = df_new[FIELDNAMES]
        if out_csv.exists():
            try: df_prev = pd.read_csv(out_csv)
            except Exception: df_prev = pd.DataFrame(columns=FIELDNAMES)

            if not df_prev.empty:
                # URLs we are re-scraping (old rows to be replaced)
                mask_replaced = df_prev["url"].isin(df_new["url"])
                df_replaced = df_prev[mask_replaced]

                # Delete old HTML files for those replaced rows
                if "full_html_path" in df_replaced.columns:
                    for html_path in df_replaced["full_html_path"].dropna().unique():
                        if not html_path or not isinstance(html_path, str):
                            continue
                        try:
                            p = Path(html_path)
                            if not p.is_absolute():
                                p = PROJECT_ROOT / p
                            if p.exists():
                                p.unlink()
                        except OSError:
                            # ignore deletion failures
                            pass

                # Keep only rows that are NOT being replaced
                df_prev = df_prev[~mask_replaced]

            # Now append new rows. Last fetched wins.
            df_out = pd.concat([df_prev, df_new], ignore_index=True)
            df_out.drop_duplicates(subset=["url"], keep="last", inplace=True)
        else:
            df_out = df_new
        df_out.to_csv(out_csv, index=False, encoding="utf-8")
        # DEBUG: verify that the file we just wrote truly contains the new URLs
        try:
            _dbg = pd.read_csv(out_csv, dtype={"url": str})
            _dbg_urls = _dbg["url"].dropna().astype(str).str.strip()
            st.info(f"Post-run `{out_csv.name}`: rows={_dbg.shape[0]:,}, unique url={_dbg_urls.nunique():,}")
        except Exception as _e:
            st.warning(f"Could not re-open {out_csv.name} to verify row count: {_e}")

        # failures/soft sidecars
        site_fail_dir = RUN_LOGS_DIR / site_key_local
        site_fail_dir.mkdir(parents=True, exist_ok=True)
        failures_path = site_fail_dir / f"{base_no_ext}__failures.csv"
        if failures:
            pd.DataFrame(failures, columns=["url","failure_type","full_html_path","teasers_csv","when","template_used","http_status"]).to_csv(
                failures_path, index=False, encoding="utf-8")
            st.warning(f"[{site_key_local}] {len(failures)} failure(s) → {failures_path}")
        else:
            if failures_path.exists(): failures_path.unlink()
            st.success(f"[{site_key_local}] No failures for this file")

        soft_path = site_fail_dir / f"{base_no_ext}__softfailures.csv"
        if softfails:
            pd.DataFrame(softfails, columns=["url","failure_type","full_html_path","teasers_csv","when","template_used","http_status"]).to_csv(
                soft_path, index=False, encoding="utf-8")
            st.info(f"[{site_key_local}] {len(softfails)} soft issue(s) → {soft_path.name}")
        else:
            if soft_path.exists(): soft_path.unlink()

        # per-file metrics
        rows_written = len(rows_new); hard_failures = len(failures); soft_issues = len(softfails)
        mr = st.columns(4)
        with mr[0]: st.metric("Rows written", f"{rows_written:,}")
        with mr[1]: st.metric("Hard failures", f"{hard_failures:,}")
        with mr[2]: st.metric("Soft issues", f"{soft_issues:,}")
        with mr[3]:
            denom = rows_written + hard_failures
            pct_ok = 0 if denom == 0 else int(100 * rows_written / denom)
            st.metric("Success rate", f"{pct_ok}%")

        # accumulate site totals
        rows_total += rows_written; hard_total += hard_failures; soft_total += soft_issues; files_done += 1
        pb_files.progress(file_idx/total_files, text=f"[{site_key_local}] Finished {csv_path.name} → {out_csv.name}")

    return (rows_total, hard_total, soft_total, files_done)

run_btn = st.button("Run")

if run_btn:
    # -------------------------- DEMO MODE PATH --------------------------
    if DEMO_MODE:
        if not sites_selected:
            st.error("Pick a site (or enable Multi-site mode and select sites).")
            st.stop()

        st.subheader("Demo run (precomputed outputs)")
        total_rows = 0
        demo_summaries = []

        for s in sites_selected:
            # Determine which teaser CSVs we *pretend* to run
            if mode == "Multi-site":
                sprefix = f"{(s or '').strip()}_".lower()
                all_csvs = sorted(DEMO_TEASERS_DIR.glob("*.csv"))
                csv_names_for_site = [p.name for p in all_csvs if p.name.lower().startswith(sprefix)]
            else:
                csv_names_for_site = list(selected_csv_names)

            if not csv_names_for_site:
                st.info(f"[{s}] No demo teaser CSVs in `data/demo/teasers/` for this site.")
                continue

            for csv_name in csv_names_for_site:
                base_no_ext = Path(csv_name).stem
                # Look for a matching demo articles file
                candidates = [
                    DEMO_OUTPUTS_DIR / f"{base_no_ext}_articles_demo.csv",
                    DEMO_OUTPUTS_DIR / f"{base_no_ext}_articles.csv",
                ]
                demo_file = next((p for p in candidates if p.exists()), None)

                if demo_file is None:
                    st.warning(
                        f"[{s}] No demo articles output found for `{csv_name}` "
                        "(looked in `data/demo/outputs/`)."
                    )
                    continue

                try:
                    df_demo = pd.read_csv(demo_file)
                except Exception as e:
                    st.error(f"[{s}] Failed to read demo output `{demo_file.name}`: {e}")
                    continue

                rows = len(df_demo)
                total_rows += rows
                demo_summaries.append((s, csv_name, demo_file, rows))

                st.markdown(f"#### [Demo] {s} · {csv_name}")
                st.caption(f"Loaded demo output from `{demo_file}`")
                st.dataframe(df_demo.head(30))

        if demo_summaries:
            st.divider()
            st.subheader("Demo summary")

            for (s, csv_name, demo_file, rows) in demo_summaries:
                cols = st.columns(3)
                with cols[0]:
                    st.write(f"**{s}**")
                with cols[1]:
                    st.write(csv_name)
                with cols[2]:
                    st.metric("Rows", f"{rows:,}")

            st.success(f"Demo run finished. Total rows across demo outputs: {total_rows:,}")
        else:
            st.warning(
                "No demo outputs were found. "
                "Populate `data/demo/teasers/` and `data/demo/outputs/` with small example files."
            )

        st.stop()

    # ------------------------- NORMAL MODE PATH -------------------------
    if not sites_selected:
        st.error("Pick a site (or enable Multi-site mode and select sites).")
        st.stop()

    total_rows = total_hard = total_soft = 0
    site_summaries = []

    for s in sites_selected:
        # Determine which CSV files to run for this site.
        # If Multi-site mode is ON, we auto-pick CSVs whose filename starts with "<site>_".
        # If Multi-site mode is OFF, we respect the page's CSV selection (selected_csv_names).
        if mode == "Multi-site":
            sprefix = f"{(s or '').strip()}_".lower()
            all_csvs = sorted([p for p in csv_dir.glob("*.csv")])
            csv_names_for_site = [p.name for p in all_csvs if p.name.lower().startswith(sprefix)]
        else:
            csv_names_for_site = list(selected_csv_names)

        if not csv_names_for_site:
            st.info(f"[{s}] No CSVs selected/matching prefix.")
            continue

        st.markdown(f"### Site: **{s}**")
        rows, hard, soft, files_done = _run_for_site(
            s, csv_dir, csv_names_for_site, url_col, limit_per_file, save_full_html
        )

        site_summaries.append((s, rows, hard, soft, files_done))
        total_rows += rows
        total_hard += hard
        total_soft += soft

    # -------- Per-site summary + overall totals --------
    st.divider()
    st.subheader("Summary")
    for (s, rows, hard, soft, files_done) in site_summaries:
        cols = st.columns(5)
        with cols[0]:
            st.write(f"**{s}**")
        with cols[1]:
            st.metric("Rows", f"{rows:,}")
        with cols[2]:
            st.metric("Hard", f"{hard:,}")
        with cols[3]:
            st.metric("Soft", f"{soft:,}")
        with cols[4]:
            denom = rows + hard
            pct_ok = 0 if denom == 0 else int(100 * rows / denom)
            st.metric("Success", f"{pct_ok}%")

    if site_summaries:
        st.divider()
        cols = st.columns(4)
        with cols[0]:
            st.metric("ALL sites · Rows", f"{total_rows:,}")
        with cols[1]:
            st.metric("ALL sites · Hard", f"{total_hard:,}")
        with cols[2]:
            st.metric("ALL sites · Soft", f"{total_soft:,}")
        with cols[3]:
            denom = total_rows + total_hard
            pct_ok = 0 if denom == 0 else int(100 * total_rows / denom)
            st.metric("ALL sites · Success", f"{pct_ok}%")

    st.success("Run finished.")

# Internal respectful delay (no UI)
#def _sleep_between_requests():
#    time.sleep(random.uniform(1.0, 2.2))

# --- Re-scrape subset -----
# --- Targeted re-scrape (replace rows in _articles.csv) ---------------------
st.divider()
st.subheader("Targeted re-scrape")

with st.expander("Targeted re-scrape (replace rows in _articles.csv)", expanded=False):
    st.markdown(
        "Fetch a small set of URLs again (Requests → Selenium as needed) and **replace** their rows "
        "in an existing `_articles.csv`."
    )
    st.caption(
        "Use this for **specific rows that already exist** in the base file (e.g., wrong text/dates). "
        "For missing rows, use **Fix failures (re-scrape)** below."
    )

    # Pick site explicitly so this works in both Single- and Multi-site modes
    site_for_target = st.selectbox(
        "Site",
        options=site_options,
        index=(site_options.index(site_key) if 'site_key' in locals() and site_key in site_options else 0),
        help="Which site's _articles.csv to patch."
    )

    # URL source for the subset
    src = st.radio(
        "URL source",
        [
            "From builder_matrix_urls.json",
            "All rows in chosen _articles.csv",
            "Paste list",
            "Upload CSV",
        ],
        horizontal=True,
        help="Pick where the subset of URLs will come from."
    )

    urls_subset = []
    if src == "From builder_matrix_urls.json":
        bm_path = Path("data/tmp/builder_matrix_urls.json")
        if not bm_path.exists():
            st.warning("No data/tmp/builder_matrix_urls.json found. Use View Data → “Send filtered to Builder” first.")
        else:
            try:
                payload = json.loads(bm_path.read_text(encoding="utf-8"))
                urls_subset = payload.get("urls") or []
                st.caption(f"Loaded {len(urls_subset):,} URLs from builder_matrix_urls.json")
            except Exception as e:
                st.error(f"Failed to read builder_matrix_urls.json: {e}")

    elif src == "All rows in chosen _articles.csv":
        # URLs will be loaded from the selected base _articles.csv further down
        urls_subset = []

    elif src == "Paste list":
        txt = st.text_area(
            "Paste URLs (one per line)",
            height=140,
            help="One URL per line. Empty lines are ignored."
        )
        urls_subset = [u.strip() for u in (txt or "").splitlines() if u.strip()]

    elif src == "Upload CSV":
        up = st.file_uploader("Upload CSV with a URL column", type=["csv"])
        if up is not None:
            try:
                tmp_df = pd.read_csv(up, low_memory=False)
                guess = None
                for c in tmp_df.columns:
                    cl = str(c).lower()
                    if cl in ("url", "final_url", "link", "article_url"):
                        guess = c
                        break
                url_col_up = st.selectbox(
                    "URL column",
                    options=list(tmp_df.columns),
                    index=(list(tmp_df.columns).index(guess) if guess in tmp_df.columns else 0),
                )
                urls_subset = tmp_df[url_col_up].astype(str).dropna().tolist()
                st.caption(f"Loaded {len(urls_subset):,} URLs from uploaded file")
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")

    # Base _articles.csv candidates for the chosen site
    base_candidates = sorted(ARTICLES_DIR.glob(f"{site_for_target}_*_articles.csv"))
    base_pick = st.selectbox(
        "Base _articles.csv to replace rows in",
        options=base_candidates,
        format_func=lambda p: p.name if isinstance(p, Path) else str(p),
        help="Rows matching by url (or final_url as fallback) will be replaced in this file."
    ) if base_candidates else None

    # If requested, use *all* URLs from the chosen base _articles.csv
    if src == "All rows in chosen _articles.csv":
        if base_pick is None:
            st.warning("Pick a base _articles.csv first to load all its URLs.")
            urls_subset = []
        else:
            try:
                base_df_for_urls = pd.read_csv(base_pick, low_memory=False)
                # Prefer 'url', fallback to 'final_url'
                url_col_candidates = [c for c in ["url", "final_url"] if c in base_df_for_urls.columns]
                if not url_col_candidates:
                    st.warning("Base CSV has neither 'url' nor 'final_url' column. Cannot load URLs.")
                    urls_subset = []
                else:
                    url_col = url_col_candidates[0]
                    urls_subset = (
                        base_df_for_urls[url_col]
                        .astype(str)
                        .dropna()
                        .tolist()
                    )
                    st.caption(f"Loaded {len(urls_subset):,} URLs from {base_pick.name}")
            except Exception as e:
                st.error(f"Failed to read base CSV for URLs: {e}")
                urls_subset = []

    # Options
    col_opt = st.columns([1,1,2])
    with col_opt[0]:
        prefer_saved = st.checkbox(
            "Prefer saved HTML if available",
            value=False,
            help="If a saved full_html_path exists in the base CSV row, try using it first; otherwise do a live fetch."
        )
    with col_opt[1]:
        cap = st.number_input("Limit (0 = all)", min_value=0, max_value=50000, value=0, step=100,
                              help="Cap how many URLs will be processed in this subset run.")
    with col_opt[2]:
        st.caption("Live re-scrape uses the same extraction/normalization pipeline as the main run.")
        save_snapshots = st.checkbox(
            "Save full HTML snapshots for re-scraped rows",
            value=False,
            help="If ON, live-fetched HTML will be saved to disk and its path recorded in full_html_path."
        )

    do_subset = st.button("Run re-scrape and replace", type="primary",
                          disabled=(len(urls_subset) == 0 or base_pick is None))

# Run the subset when clicked
if 'do_subset' in locals() and do_subset:
    if not urls_subset:
        st.error("No URLs provided.")
    elif base_pick is None:
        st.error("Pick a base _articles.csv.")
    else:
        # Use the selected site
        site_cfg = _load_site_yaml(site_for_target)

        # 1) Load base CSV & index for updates
        try:
            base_df = pd.read_csv(base_pick, low_memory=False)
        except Exception as e:
            st.error(f"Failed to read base CSV: {e}")
            base_df = pd.DataFrame()

        if base_df.empty:
            st.error("Base CSV is empty or unreadable.")
        else:
            # prefer 'url' (new convention), fallback to 'final_url' (legacy)
            join_key = "url" if "url" in base_df.columns else ("final_url" if "final_url" in base_df.columns else None)
            if not join_key:
                st.error("Base CSV has neither 'url' nor 'final_url' column to join on.")
            else:
                # 2) Prepare subset and cap
                to_process = [u for u in urls_subset if isinstance(u, str) and u.strip()]
                if cap and cap > 0:
                    to_process = to_process[:cap]

                st.info(f"Processing {len(to_process):,} URLs…")
                prog = st.progress(0.0)
                new_rows = []

                # Helper: extract from saved HTML first (if enabled & exists), else live fetch
                def _extract_one(url: str) -> dict:
                    # Try saved HTML from base_df
                    if prefer_saved and join_key in base_df.columns:
                        try:
                            r = base_df.loc[base_df[join_key] == url]
                            if r.empty and join_key == "final_url" and "url" in base_df.columns:
                                r = base_df.loc[base_df["url"] == url]
                            fullp = None
                            if not r.empty and "full_html_path" in base_df.columns:
                                fullp = r.iloc[0].get("full_html_path")
                            if fullp and isinstance(fullp, str):
                                p = Path(fullp)
                                if not p.is_absolute():
                                    p = PROJECT_ROOT / p
                                if p.exists():
                                    html = p.read_text(encoding="utf-8", errors="ignore")
                                    from conty_core.extract import extract_article_from_html as _extract_html
                                    out = _extract_html(html, base_url=url) or {}
                                    out["final_url"] = out.get("final_url") or url
                                    # pass back the used path (prefer relative to project root)
                                    used = p
                                    try:
                                        used = used.relative_to(PROJECT_ROOT)
                                    except Exception:
                                        pass
                                    out["__used_full_html_path"] = str(used)
                                    return out
                        except Exception:
                            pass

                    # Live fetch (mirror main loop)
                    try:
                        main_hint, body_hint = _css_hints_from_site_cfg(site_cfg)
                        prefer_js, tpl_wait_css, tpl_wait_to = _prefers_selenium(site_cfg, url)
                        force_js = prefer_js or bool(st.session_state.get("force_js", False))
                        wait_to  = float(tpl_wait_to or st.session_state.get("selenium_wait_timeout", 20.0) or 20)
                        res: FetchResult = fetch_with_fallback(
                            url,
                            container_css=main_hint,
                            item_css=body_hint,
                            force_js=force_js,
                            wait_timeout=wait_to,
                            consent_click_xpaths=KNOWN_CONSENT_XPATHS if (st.session_state.get("use_js_fallback", True) or force_js) else None,
                        )
                        final_url = res.final_url or url
                        html = res.html or ""
                        if not html:
                            return {}
                        out, tpl_used, status, missing, min_chars, body_len = _extract_best(url, html, site_cfg)
                        out["final_url"] = final_url
                        out["template_used"] = out.get("template_used") or tpl_used
                        out["status"] = out.get("status") or status
                        out["__min_chars"] = min_chars
                        out["__body_len"] = body_len
                        # Optionally save a new snapshot and return its path
                        if save_snapshots and html:
                            try:
                                snap_path = _write_full_html(site_for_target, 1, final_url, "", html)  # returns relative path
                                out["__used_full_html_path"] = snap_path
                            except Exception:
                                pass
                        return out
                    except Exception:
                        return {}

                # 3) Process
                for i, u in enumerate(to_process, start=1):
                    out = _extract_one(u) or {}
                    final_u = out.get("final_url") or u
                    text_val = out.get("text") or ""
                    body_len = int(out.get("__body_len") or len(text_val))
                    min_chars = int(out.get("__min_chars") or (site_cfg.get("min_body_chars") or 400))
                    pub_iso, pub_date, upd_iso, upd_date, soft = _normalize_dates_for_output(out, min_chars, body_len, site_for_target, u)
                    row = {
                        "site": site_for_target, "url": u, "final_url": final_u,
                        "title": out.get("title") or "",
                        "subtitle": out.get("subtitle") or "",
                        "section": out.get("section") or "",
                        "author": out.get("author") or "",
                        "published_raw": (
                            out.get("published_raw")
                            or out.get("published_time")
                            or out.get("published")
                            or ""
                        ),
                        "published_datetime": pub_iso,
                        "published_date": pub_date,
                        "updated_raw": (
                            out.get("updated_raw")
                            or out.get("updated_time")
                            or out.get("updated")
                            or ""
                        ),
                        "updated_datetime": upd_iso,
                        "updated_date": upd_date,                        

                        "lead": out.get("lead") or "", 
#                        "text": text_val,
                        "text": (out.get("body") or out.get("text") or ""),
                        "main_image": out.get("main_image") or "", "tags": out.get("tags") or "",
                        "status": out.get("status") or "ok",
                        "reason": ";".join(soft) if soft else (out.get("reason") or ""),
                        "template_used": out.get("template_used") or "",
                        "fetched_datetime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "full_html_path": out.get("__used_full_html_path", ""),
                    }
                    new_rows.append(row)
                    prog.progress(i / max(1, len(to_process)))

                # 4) Replace into base CSV (prefer join on 'url', fallback to 'final_url')
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    if "url" in base_df.columns and "url" in new_df.columns:
                        join_key_use = "url"
                    elif "final_url" in base_df.columns and "final_url" in new_df.columns:
                        join_key_use = "final_url"
                    else:
                        st.error("No common join key between subset results and base CSV.")
                        st.stop()

                    base_idx = base_df.set_index(join_key_use)
                    new_idx  = new_df.set_index(join_key_use)

                    common = base_idx.index.intersection(new_idx.index)
                    if len(common) == 0:
                        st.warning("No matching keys between subset results and base CSV. Nothing to replace.")
                    else:
                        replace_cols = [c for c in new_idx.columns if c in base_idx.columns and c != join_key_use]
                        for c in replace_cols:
                            try:
                                if base_idx[c].dtype != 'O':   # not object
                                    base_idx[c] = base_idx[c].astype('object')
                            except Exception:
                                pass
                            try:
                                if new_idx[c].dtype != 'O':
                                    new_idx[c] = new_idx[c].astype('object')
                            except Exception:
                                pass
                        backup = base_pick.with_suffix(".bak")
                        try:
                            pd.read_csv(base_pick, low_memory=False).to_csv(backup, index=False)
                        except Exception:
                            pass
                        # If the new full_html_path is empty, keep the old one
                        if "full_html_path" in replace_cols:
                            common_list = list(common)
                            # consider empty if NaN or blank
                            new_paths = new_idx.loc[common_list, "full_html_path"].astype(str).str.strip()
                            empties = new_paths.eq("")
                            if empties.any():
                                idx_to_fill = [i for i, is_empty in zip(common_list, empties.tolist()) if is_empty]
                                if idx_to_fill:
                                    new_idx.loc[idx_to_fill, "full_html_path"] = base_idx.loc[idx_to_fill, "full_html_path"]

                        base_idx.loc[common, replace_cols] = new_idx.loc[common, replace_cols]
                        base_idx.reset_index().to_csv(base_pick, index=False, encoding="utf-8")
                        st.success(f"Replaced {len(common):,} rows in {base_pick.name}. Backup saved as {backup.name}.")
                        st.metric("Subset processed", f"{len(new_rows):,}")
                else:
                    st.info("No rows produced by the subset run.")

# Fix failures
st.divider()
st.subheader("Fix failures")
with st.expander("Fix failures (re-scrape)", expanded=False):
    st.caption(
        "Use this to **recover rows that didn’t make it into `_articles.csv` at all** (hard failures: "
        "`fetch_error`, `extract_error`, `no_title`). "
        "It will try saved full HTML when available; otherwise it refetches."
    )
    st.caption(
        "Tip: **Soft failures** (e.g., missing/failed dates, short body) already have rows in the CSV. "
        "Handle those via **Targeted re-scrape** above or in **View Data**."
    )

    # pick site & failures file
    sites_with_logs = sorted([p.name for p in RUN_LOGS_DIR.iterdir() if p.is_dir()])
    if not sites_with_logs:
        st.info("No failures files yet.")
    else:
        site_pick = st.selectbox("Site", options=sites_with_logs, help="Choose the site folder under run_logs/")
        if site_pick:
            fail_files = sorted((RUN_LOGS_DIR / site_pick).glob("*__failures.csv"))
            if not fail_files:
                st.info("No failures files for this site.")
            else:
                chosen = st.selectbox("Failures file", options=fail_files, format_func=lambda p: p.name, help="Pick a __failures.csv to re-scrape.")
                ignore_saved_html = st.checkbox(
                    "Ignore saved HTML and re-fetch",
                    value=False,
                    help="Use network (Requests→Selenium) even if a full_html_path exists."
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    do_fix = st.button("Re-scrape failures now", type="primary")
                with c2:
                    ignore = st.button("Ignore & remove this failures file")
                with c3:
                    st.caption("Reuses saved full HTML if present; otherwise refetches.")

                if do_fix:
                    with st.spinner("Re-scraping failed URLs..."):
                        processed, fixed, remaining = _fix_failures_for_file(
                            site_pick, chosen,
                            save_full_html=st.session_state.get("save_full_html", True),
                            ignore_saved_html=ignore_saved_html
                        )
                    st.success(f"Processed {processed} | fixed {fixed} | remaining {remaining}")
                if ignore:
                    chosen.unlink(missing_ok=True)
                    st.warning("Failures file removed.")


