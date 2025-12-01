# --- bootstrap project root ---
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ------------------------------------------------------------

from pathlib import Path
from datetime import datetime, date
import json
import os
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np

from urllib.parse import urljoin, urlparse

st.set_page_config(page_title="5 · View Data (Articles)", layout="wide")
st.title("View Data (Articles)")

DEMO_MODE = bool(os.environ.get("CONTY_DEMO", "").strip())

# --- Page intro / quick help -------------------------------------------------
with st.expander("How this page works (quick guide)", expanded=False):
    st.markdown(
        """
This page is for **diagnosing scraping quality** and **finding silent failures** that were not caught during `Run Scrapes`.

### What to do here
1. **Select a project / dataset**  
   Choose a `_articles.csv` file generated during `Run Scrapes`.

2. **Review / Filter rows**  
   Use the filters to spot **problem cases**, especially:
   - Articles **missing both published & updated datetime**
   - Articles where **datetime parsing failed**
   - Articles with **very short body text** (likely wrong container selector)
   - Articles marked **OK but suspicious** (looks structurally wrong, not outright failed)

3. **Inspect individual rows**  
   Click **“Row details / Preview”** to view fields, text, and saved HTML paths.

4. **Act on filtered rows**
   - **Send filtered to Builder (matrix)** to fix selectors on those exact pages.
   - Or **Download filtered CSV/JSONL** for offline review or external tooling.
   - If you see many `parse_failed_*` issues, inspect the raw datetime fields and adjust your selectors in **Build a Scraper**.    

5. **Option A: Re-extract locally (fast, uses saved HTML)**
   Use **“Re-extract from saved HTML”** to re-run extraction **without fetching**.  
   This is ideal when:
   - You only need to fix **a subset** of URLs
   - You have **saved HTML** and want **deterministic debugging**
   
   After running it:
   - Compare **redo vs redo_out** in the Diff expander
   - If the results look correct → **Apply redo_out → replace** to update the existing `_articles.csv` **in place**

6. **Option B: Full rebuild via Run Scrapes (canonical regeneration)**
   Return to **Run Scrapes** and reprocess from the original teasers.
   This is ideal when:
   - You changed template logic more broadly
   - You want to refresh **all** articles, not just the filtered subset
   - Or saved HTML snapshots are missing / outdated

**Summary:**  
- Use **Option A** for quick, targeted corrections (no network needed).  
- Use **Option B** for a clean, complete re-run of the dataset.

### Tips
- **Short bodies** almost always mean **main container mismatch** or **JS needed**.
- **No published AND no updated** usually means the datetime selector is wrong or varies by section.
- Use saved HTML first to debug — it avoids network noise and site layout changes.
- Your goal here is not to fix the data, but to **identify where the scraper needs improving**.

> This page = **triage**  
> Build a Scraper = **fix**  
> Run Scrapes = **produce final clean output**
        """
    )

# ------------------ Defaults & helpers ------------------
if DEMO_MODE:
    OUTPUTS_BASE = Path("data/demo")
    DEFAULT_DIR = OUTPUTS_BASE / "outputs"
else:
    OUTPUTS_BASE = Path("data/outputs")
    DEFAULT_DIR = OUTPUTS_BASE / "articles"
DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

def _abs_url(maybe_url, base) -> str | None:
    # treat None/NaN/empty as missing
    try:
        import pandas as pd  # already imported above
        if maybe_url is None or pd.isna(maybe_url):
            return None
    except Exception:
        if maybe_url is None:
            return None

    # coerce to string safely
    if not isinstance(maybe_url, str):
        try:
            maybe_url = str(maybe_url)
        except Exception:
            return None

    u = maybe_url.strip()
    if not u:
        return None

    # protocol-relative
    if u.startswith("//"):
        return "https:" + u

    # absolute URL already
    if urlparse(u).scheme in ("http", "https"):
        return u

    # make base safe too (can be NaN/None)
    base_s = ""
    try:
        if base is not None and not (pd.isna(base) if "pd" in globals() else False):
            base_s = str(base)
    except Exception:
        base_s = str(base or "")

    return urljoin(base_s, u)

def _list_csvs(folder: Path) -> list[Path]:
    try:
        return sorted(folder.glob("*.csv"))
    except Exception:
        return []

def _read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    # tolerant CSV reader
    try:
        df = pd.read_csv(path, low_memory=False, nrows=nrows)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {path}\n{e}")
        return pd.DataFrame()

def _to_datetime(s) -> pd.Timestamp | None:
    """Parse as UTC (handles mixed tz safely), then drop tz to keep display/filtering naive."""
    try:
        return pd.to_datetime(s, errors="coerce", utc=True).tz_convert(None)
    except Exception:
        return None

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort conversions for known fields:
    - Parse *_datetime columns to naive pandas Timestamps.
    - Ensure *_date columns are date-like (YYYY-MM-DD).
    - Keep text/html columns as strings.
    - Backfill from older columns if needed (published_time, updated_time, fetched_at).
    - If we encounter Greek human-readable dates, try conty_core.postprocess.normalize_datetime.
    """
    out = df.copy()

    # 1) unify the datetime columns we expect in runners
    datetime_cols = {
        "published_datetime": None,
        "updated_datetime": None,
        "fetched_datetime": None,
    }

    # Support old names too (builder/legacy)
    legacy_map = {
        "published_time": "published_datetime",
        "updated_time": "updated_datetime",
        "fetched_at": "fetched_datetime",
    }

    for old, new in legacy_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    # 2) try parsing datetime columns (prefer explicit ISO formats to avoid warnings)
    for c in list(datetime_cols.keys()):
        if c in out.columns:
            s = out[c].astype(str)

            mask_min = s.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
            mask_sec = s.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
            parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

            if mask_min.any():
                parsed.loc[mask_min] = pd.to_datetime(s.loc[mask_min], format="%Y-%m-%dT%H:%M", utc=True, errors="coerce")
            if mask_sec.any():
                parsed.loc[mask_sec] = pd.to_datetime(s.loc[mask_sec], format="%Y-%m-%dT%H:%M:%S", utc=True, errors="coerce")

            # fallback for anything else (silence pandas' "could not infer format" warning)
            rem = ~(mask_min | mask_sec)
            if rem.any():
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    parsed.loc[rem] = pd.to_datetime(s.loc[rem], errors="coerce", utc=True)

            out[c] = parsed.dt.tz_convert(None)

    # 3) date-only columns: trust runner if already present; else derive from *_datetime
    if "published_date" not in out.columns and "published_datetime" in out.columns:
        out["published_date"] = out["published_datetime"].dt.date
    if "updated_date" not in out.columns and "updated_datetime" in out.columns:
        out["updated_date"] = out["updated_datetime"].dt.date

    try:
        from conty_core.postprocess import normalize_datetime as _norm_dt
        if "published_date" in out.columns:
            mask = out["published_date"].notna() & out["published_date"].astype(str).str.contains(r"[Α-Ωα-ω]|Οκτωβ|Ιαν|Φεβ|Μαρ|Απρ|Μαΐ|Μαι|Ιουν|Ιούλ|Ιουλ|Αυγ|Σεπ|Οκτ|Νοε|Δεκ|·|\|", regex=True)
            if mask.any():
                iso = out.loc[mask, "published_date"].astype(str).apply(lambda s: (_norm_dt(s) or "")[:10])
                out.loc[mask, "published_date"] = iso.replace("", pd.NA)
                # if we don’t have published_datetime, also fill it from the same source
                if "published_datetime" in out.columns:
                    # only fill missing published_datetime
                    miss = out["published_datetime"].isna()
                    fill_iso = out.loc[mask & miss, "published_date"].astype(str).apply(lambda d: (d + "T00:00") if isinstance(d, str) and len(d) == 10 else pd.NA)
                    out.loc[mask & miss, "published_datetime"] = pd.to_datetime(fill_iso, format="%Y-%m-%dT%H:%M", errors="coerce")
    except Exception:
        # postprocess import failed – ignore, just leave the raw strings
        pass

    # 5) keep text/html columns as strings (display-friendly)
    for c in ["title","subtitle","lead","author","section","tags","url","final_url","template_used","status","reason"]:
        if c in out.columns:
            out[c] = out[c].astype(str)

    return out

def _safe_contains(series: pd.Series, needle: str) -> pd.Series:
    try:
        return series.astype(str).str.contains(needle, case=False, na=False, regex=True)
    except Exception:
        return series.astype(str).str.contains(needle, case=False, na=False, regex=False)

def _guess_text_col(df: pd.DataFrame):
    for c in ["text", "body_text", "article_text", "content", "body"]:
        if c in df.columns:
            return c
    return None

def _guess_url_col(df: pd.DataFrame):
    """
    Choose the URL column that actually has usable http(s) URLs
    in the *current* DataFrame (e.g. filtered rows).

    Preference order: final_url, then url – but only if they
    contain at least one http(s) URL.
    """
    candidates = []
    for c in ["final_url", "url"]:
        if c in df.columns:
            # count how many real http(s) URLs this column has
            s = df[c].dropna().astype(str).str.strip()
            n_valid = s.str.startswith(("http://", "https://")).sum()
            candidates.append((n_valid, c))

    if not candidates:
        return None

    # pick the column with the most valid URLs
    candidates.sort(reverse=True)
    best_n, best_col = candidates[0]
    return best_col if best_n > 0 else None

def _infer_site_from_filename(path: Path) -> str:
    """Infer site key from file name prefix (e.g., kathimerini_..._articles.csv → kathimerini)."""
    stem = path.stem  # e.g., kathimerini_2025_01_14_articles
    if stem.endswith("_articles"):
        stem = stem[: -len("_articles")]
    return stem.split("_", 1)[0] if "_" in stem else stem

# ------------------ Quality flags (derived) ------------------
def _add_quality_flags(df: pd.DataFrame, short_body_threshold: int = 200) -> pd.DataFrame:
    out = df.copy()

    # Booleans for the presence of normalized datetimes
    has_pub_dt = out["published_datetime"].notna() if "published_datetime" in out.columns else pd.Series(False, index=out.index)
    has_upd_dt = out["updated_datetime"].notna() if "updated_datetime" in out.columns else pd.Series(False, index=out.index)

    # Missing both (most important "silent failure" signal)
    out["__missing_both_dt"] = ~(has_pub_dt | has_upd_dt)

    # Parse issues: prefer explicit raw fields if present; else infer
    pub_raw = out["published_raw"] if "published_raw" in out.columns else None
    upd_raw = out["updated_raw"] if "updated_raw" in out.columns else None

    parse_issue_pub = pd.Series(False, index=out.index)
    parse_issue_upd = pd.Series(False, index=out.index)

    if pub_raw is not None:
        parse_issue_pub = pub_raw.notna() & ~has_pub_dt
    else:
        # heuristic: we have a date-like string but no parsed datetime
        if "published_date" in out.columns:
            parse_issue_pub = out["published_date"].notna() & ~has_pub_dt

    if upd_raw is not None:
        parse_issue_upd = upd_raw.notna() & ~has_upd_dt
    else:
        if "updated_date" in out.columns:
            parse_issue_upd = out["updated_date"].notna() & ~has_upd_dt

    out["__dt_parse_issue"] = parse_issue_pub | parse_issue_upd

    # Short body (relative to guessed text column)
    tcol = _guess_text_col(out)
    if tcol and tcol in out.columns:
        lengths = out[tcol].astype(str).str.len()
        out["__short_body"] = lengths < int(short_body_threshold)
        out["__text_len"] = lengths
    else:
        out["__short_body"] = False
        out["__text_len"] = pd.NA

    # Suspicious even though status says ok
    ok_mask = (out["status"].astype(str) == "ok") if "status" in out.columns else pd.Series(False, index=out.index)
    out["__ok_but_suspicious"] = ok_mask & (out["__missing_both_dt"] | out["__dt_parse_issue"] | out["__short_body"])

    return out

# ------------------ Sidebar: pick folder + csv ------------------
with st.sidebar:
    st.header("Source")
    folder_str = st.text_input("Folder with article CSVs", value=str(DEFAULT_DIR))
    folder = Path(folder_str).expanduser()
    csvs = _list_csvs(folder)

    if not csvs:
        st.info("No CSVs found here yet. Run scrapes to generate `_articles.csv` files.")
        st.stop()

    # choose one file (like teasy)
    csv_names = [p.name for p in csvs]
    i_default = max(0, len(csv_names) - 1)  # default to the newest-looking by name
    chosen_name = st.selectbox("Choose data file", options=csv_names, index=i_default)
    chosen_path = folder / chosen_name

    st.caption(f"File: `{chosen_path}`")

    # Site hint (from session or filename)
    site_key_hint = st.session_state.get("__viewdata_site_hint") or _infer_site_from_filename(chosen_path)

# Fixed defaults (no sidebar controls)
sample_n = 0                # 0 = read all rows
hide_html_cols = True       # keep heavy HTML columns hidden in the main table
show_text_preview = True    # always show the per-row details expanders

# ------------------ Load data ------------------
df = _read_csv(chosen_path, nrows=(sample_n if sample_n > 0 else None))
if df.empty:
    st.warning("This file has no rows (or failed to load).")
    st.stop()

df = _normalize_cols(df)

# Add quality flags with a sensible default threshold
df = _add_quality_flags(df, short_body_threshold=200)
# --- pre-filter quality counters (defaults use 200 threshold) ---
_q_missing_both = int(df["__missing_both_dt"].sum()) if "__missing_both_dt" in df.columns else 0
_q_parse_issue  = int(df["__dt_parse_issue"].sum()) if "__dt_parse_issue" in df.columns else 0
_q_short_body   = int(df["__short_body"].sum()) if "__short_body" in df.columns else 0
_q_ok_susp      = int(df["__ok_but_suspicious"].sum()) if "__ok_but_suspicious" in df.columns else 0

# ------------------ Quick info ------------------
cols = st.columns(4)
with cols[0]:
    st.metric("Rows", f"{len(df):,}")
with cols[1]:
    st.metric("Columns", f"{len(df.columns):,}")
with cols[2]:
    st.metric("OK rows", str(int((df["status"] == "ok").sum())) if "status" in df.columns else "?")
with cols[3]:
    st.metric("Templates", str(df["template_used"].nunique()) if "template_used" in df.columns else "?")

sub = st.columns(4)
with sub[0]:
    st.metric("Missing both datetimes", f"{_q_missing_both:,}")
with sub[1]:
    st.metric("Datetime parse issues", f"{_q_parse_issue:,}")
with sub[2]:
    st.metric("Short body (≤200 chars)", f"{_q_short_body:,}")
with sub[3]:
    st.metric("OK but suspicious", f"{_q_ok_susp:,}")


# ------------------ Filters ------------------
with st.expander("Filters"):
    fcols = st.columns(4)

    with fcols[0]:
        status_opts = sorted(df["status"].dropna().unique().tolist()) if "status" in df.columns else []
        status_choice = st.multiselect("Status", status_opts, default=status_opts, help="Filter by scrape status. `ok` = extracted successfully, others indicate fetch or parsing issues.")

    with fcols[1]:
        tmpl_opts = sorted(df["template_used"].dropna().unique().tolist()) if "template_used" in df.columns else []
        tmpl_choice = st.multiselect("Template", tmpl_opts, default=tmpl_opts, help="Restrict to articles extracted using specific templates (as defined in Build a Scraper).")

    with fcols[2]:
        date_from = st.date_input("Published from", value=None, help="Show only articles published on or after this date.")

    with fcols[3]:
        date_to = st.date_input("Published to", value=None, help="Show only articles published on or before this date.")

    # full-text search (title / subtitle / lead / author / section / tags / text)
    search = st.text_input("Search in title/subtitle/lead/author/section/tags/text", value="", help="Case-insensitive substring search across common article fields and text.")

    # column visibility
    st.caption("Columns to display")
    default_show = [
        "title","subtitle","lead","section","author","published_date",
        "tags","main_image","status","reason","template_used","url",
        "container_html_path","full_html_path",
        "__text_len"
    ]
    show_cols = st.multiselect("Visible columns", options=list(df.columns), default=[c for c in default_show if c in df.columns])

    st.markdown("### Quality")
    qc1, qc2, qc3, qc4 = st.columns([1,1,1,1])
    with qc1:
        flg_missing_both = st.checkbox("No published & no updated dates", value=False, help="Rows where both published_datetime and updated_datetime are missing.")
    with qc2:
        flg_parse_issue = st.checkbox("Datetime parse issue", value=False, help="Row had a raw/derived date but parsed datetime is still missing.")
    with qc3:
        # let users tune body threshold; default to 200
        short_body_thr = st.number_input("Short body threshold (chars)", min_value=0, max_value=20000, value=200, step=100)
        flg_short_body = st.checkbox("Body too short", value=False, help=f"Text length below the threshold ({short_body_thr}).")

    with qc4:
        flg_ok_suspicious = st.checkbox("Status OK but suspicious", value=False, help="status == 'ok' AND any of the quality flags above.")

    # Recompute short-body flag with the chosen threshold
    tcol_all = _guess_text_col(df)
    if tcol_all and tcol_all in df.columns:
        df["__short_body"] = df[tcol_all].astype(str).str.len() < int(short_body_thr)
    else:
        df["__short_body"] = False

    # Show live counts for each flag under the checkboxes
    c_counts = st.columns(4)
    with c_counts[0]:
        st.caption(f"{int(df['__missing_both_dt'].sum()) if '__missing_both_dt' in df.columns else 0:,} rows")
    with c_counts[1]:
        st.caption(f"{int(df['__dt_parse_issue'].sum()) if '__dt_parse_issue' in df.columns else 0:,} rows")
    with c_counts[2]:
        st.caption(f"{int(df['__short_body'].sum()) if '__short_body' in df.columns else 0:,} rows")
    with c_counts[3]:
        st.caption(f"{int(df['__ok_but_suspicious'].sum()) if '__ok_but_suspicious' in df.columns else 0:,} rows")

# apply filters
df_f = df.copy()

if "status" in df_f.columns and status_choice:
    df_f = df_f[df_f["status"].isin(status_choice)]
if "template_used" in df_f.columns and tmpl_choice:
    df_f = df_f[df_f["template_used"].isin(tmpl_choice)]

date_col = "published_date" if "published_date" in df_f.columns else ("updated_date" if "updated_date" in df_f.columns else None)
if date_col and (date_from or date_to):
    # coerce to datetime.date if not already
    if not np.issubdtype(df_f[date_col].dtype, np.datetime64):
        # try parsing strings to dates; if they are datetime64[ns], take .date()
        try:
            tmp = pd.to_datetime(df_f[date_col], errors="coerce").dt.date
        except Exception:
            tmp = df_f[date_col]
        df_f = df_f.assign(**{date_col: tmp})

    if date_from:
        df_f = df_f[df_f[date_col] >= date_from]
    if date_to:
        df_f = df_f[df_f[date_col] <= date_to]

if search.strip():
    # build a mask across common text-bearing columns
    cols_to_search = [c for c in ["title","subtitle","lead","author","section","tags","text"] if c in df_f.columns]
    if cols_to_search:
        mask = np.zeros(len(df_f), dtype=bool)
        for c in cols_to_search:
            mask |= _safe_contains(df_f[c], search.strip())
        df_f = df_f[mask]

# Quality filters
if "__missing_both_dt" in df_f.columns and flg_missing_both:
    df_f = df_f[df_f["__missing_both_dt"]]

if "__dt_parse_issue" in df_f.columns and flg_parse_issue:
    df_f = df_f[df_f["__dt_parse_issue"]]

if "__short_body" in df_f.columns and flg_short_body:
    df_f = df_f[df_f["__short_body"]]

if "__ok_but_suspicious" in df_f.columns and flg_ok_suspicious:
    df_f = df_f[df_f["__ok_but_suspicious"]]

# Hide heavy columns by default in the table (still available in expanders / downloads)
heavy_cols = [c for c in ["text","body_html","container_html"] if c in df_f.columns]
table_cols = [c for c in show_cols if c in df_f.columns]
if hide_html_cols:
    table_cols = [c for c in table_cols if c not in heavy_cols]

st.subheader("Results")

# Quick toggle: only rows with datetime issues (missing both OR parse issues)
only_dt_issues = st.checkbox(
    "Show only rows with datetime issues",
    value=False,
    help="Shortcut filter: keeps rows where both datetimes are missing OR where parsing failed for published/updated."
)

_df_for_view = df_f.copy()
if only_dt_issues and {"__missing_both_dt","__dt_parse_issue"}.issubset(_df_for_view.columns):
    _df_for_view = _df_for_view[_df_for_view["__missing_both_dt"] | _df_for_view["__dt_parse_issue"]]

st.caption(f"Showing {_df_for_view.shape[0]:,} rows")

st.dataframe(_df_for_view[table_cols], width="stretch", height=420)

# ------------------ Row details (optional text/html preview) ------------------

# --- Simple HTML renderer (uses <base> so relative links resolve) ---
def _render_html_block(html_str: str, base_url: str = "", height: int = 900):
    import streamlit.components.v1 as components
    safe_base = f'<base href="{base_url}">' if base_url else ""
    style = """
    <style>
      html, body {
        background: #ffffff !important;
        color: #111111;
        margin: 0;
        padding: 0;
      }
      body {
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.5;
        padding: 1rem;
        max-width: 880px;
        margin: 0 auto;
      }
      img, video, iframe { max-width: 100%; height: auto; }
      figure { max-width: 100%; }
      pre, code { white-space: pre-wrap; word-break: break-word; }
      * { box-sizing: border-box; }
    </style>
    """
    # Force light color scheme inside the iframe to avoid dark-theme CSS from sites
    head_flags = '<meta name="color-scheme" content="light">'
    html_doc = f"<!doctype html><html><head>{safe_base}{head_flags}{style}</head><body>{html_str or ''}</body></html>"
    components.html(html_doc, height=height, scrolling=True)


def _load_html_from_row(r: pd.Series) -> dict:
    """
    Return {'full_html': str|None, 'source_path': str|None} by reading full_html_path if present.
    No CSV columns are expected to contain HTML.
    """
    p_full = r.get("full_html_path")
    if p_full is None or (isinstance(p_full, float) and pd.isna(p_full)):
        return {"full_html": None, "source_path": None}
    try:
        p = Path(str(p_full))
        html = p.read_text(encoding="utf-8", errors="ignore")
        return {"full_html": html, "source_path": str(p)}
    except Exception:
        return {"full_html": None, "source_path": str(p_full)}


# -- helper to render one row's details the same way as the preview loop --
def _render_row_details(row, idx, df_f, key_prefix: str = "rd"):
    from urllib.parse import urlparse
    header_title = (row.get("title") or "").strip() or (row.get(_guess_url_col(df_f) or "") or "")
    st.markdown(f"**Row {idx}** — {header_title[:160]}")
    url_col = _guess_url_col(df_f) or (df_f.columns[0] if len(df_f.columns) else None)
    key_cols = [c for c in ["title", "subtitle", "lead", "author", "section", "published_date", "updated_date", "tags", url_col] if c in df_f.columns]

    # key fields
    for k in key_cols:
        if k in row and pd.notna(row[k]):
            st.write(f"**{k}**: {row[k]}")

    # --- Datetime status & quick actions ------------------------------------
    pub_iso = row.get("published_datetime")
    upd_iso = row.get("updated_datetime")
    pub_raw = row.get("published_raw")
    upd_raw = row.get("updated_raw")

    # Determine parse flags heuristically (raw present but iso missing)
    parse_failed_pub = bool((pub_raw not in (None, "", np.nan)) and not (pd.notna(pub_iso)))
    parse_failed_upd = bool((upd_raw not in (None, "", np.nan)) and not (pd.notna(upd_iso)))
    missing_both = not (pd.notna(pub_iso) or pd.notna(upd_iso))

    st.caption("Dates ISO report")
    bcols = st.columns([2, 2, 2, 2])
    with bcols[0]:
        st.markdown("**Published date**")
        st.code(str(pub_iso) if pd.notna(pub_iso) else "—", language="")
    with bcols[1]:
        st.markdown("**Updated date**")
        st.code(str(upd_iso) if pd.notna(upd_iso) else "—", language="")
    with bcols[2]:
        st.markdown("**Dates ISO Status**")
        if missing_both:
            st.error("Missing both datetimes")
        elif parse_failed_pub and parse_failed_upd:
            st.warning("Parse failed: published & updated")
        elif parse_failed_pub:
            st.warning("Parse failed: published")
        elif parse_failed_upd:
            st.warning("Parse failed: updated")
        else:
            st.success("OK")
    with bcols[3]:
        st.markdown("**Actions**")
        # # Easy copy fields
        # with st.popover("Copy raw datetimes"):
        #     st.text_input("published_raw", value=str(pub_raw or "—"), key=f"{key_prefix}_copy_pub_{idx}", disabled=True)
        #     st.text_input("updated_raw", value=str(upd_raw or "—"), key=f"{key_prefix}_copy_upd_{idx}", disabled=True)

        # Tip
        st.caption(
            "If parsing failed, open **Build a Scraper**, inspect these raw values, "
            "and adjust datetime selectors or fall back to a different meta/date field, then re-run."
        )
    st.divider()

    # main image
    mi_raw = row.get("main_image") or ""
    if mi_raw:
        base_url = row.get("final_url") or row.get("url") or ""
        mi_resolved = _abs_url(mi_raw, base_url)
        st.markdown("**Main image**")
        st.code(mi_resolved or mi_raw, language="")
        if mi_resolved:
            st.image(mi_resolved, caption="Main image", width="stretch")
        st.divider()

    # text
    tcol = _guess_text_col(df_f)
    if tcol and pd.notna(row.get(tcol)):
        t = str(row[tcol])
        st.caption(f"Text - {len(t):,} chars")
        safe_idx = str(idx).replace(" ", "_")
        st.text_area("text", value=t, height=240, key=f"{key_prefix}_text_{safe_idx}")


    # full-screen style preview trigger
    if st.button("Open fullscreen preview ▶", key=f"{key_prefix}_open_full_{idx}", help="Show the saved HTML rendering for this article in a large view."):
        st.session_state["__vd_full_row_idx"] = idx

# ------------------ Actions on filtered set ------------------
act_cols = st.columns([1,1,1])

# A) Send filtered to Builder (matrix)
with act_cols[0]:
    if st.button("Send filtered to Builder (matrix)", help="Pass these URLs to the Build a Scraper page → Matrix Test to debug template selectors on exactly these cases."):
        url_col = _guess_url_col(df_f)
        if not url_col:
            st.error("No URL column found to send.")
        else:
            # Take URLs from the chosen column
            urls_series = df_f[url_col].dropna().astype(str)

            # Keep only proper http(s) URLs
            urls = [
                u.strip()
                for u in urls_series.tolist()
                if u.strip().startswith(("http://", "https://"))
            ]

            if not urls:
                st.error("No valid http(s) URLs found in the filtered rows to send to Builder.")
            else:
                if len(urls) > 5000:
                    urls = urls[:5000]
                    st.warning("Truncated to first 5000 URLs to keep things snappy.")

                tmp_dir = Path("data/tmp")
                tmp_dir.mkdir(parents=True, exist_ok=True)

                payload = {"source": "view_data", "file": str(chosen_path), "urls": urls}
                (tmp_dir / "builder_matrix_urls.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                st.success(
                    f"Sent {len(urls):,} URLs to Builder (matrix). "
                    "Open 'Build a Scraper' → J) Matrix test."
                )

# B) Download filtered CSV
with act_cols[1]:
    st.download_button(
        "Download filtered CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name=f"filtered_{chosen_path.name}",
        mime="text/csv",
        width='stretch',
    )

# C) Download filtered JSONL
with act_cols[2]:
    def _to_jsonl(rows):
        def _coerce(v):
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            if isinstance(v, (pd.Timestamp, datetime, date)):
                try:
                    return v.isoformat()
                except Exception:
                    return str(v)
            return v
        fixed = [{k: _coerce(v) for k, v in r.items()} for r in rows]
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in fixed)

    st.download_button(
        "Download filtered JSONL",
        data=_to_jsonl(df_f.to_dict("records")).encode("utf-8"),
        file_name=f"filtered_{chosen_path.stem}.jsonl",
        mime="application/json",
        width='stretch',
    )

if show_text_preview:
    st.divider()
    st.subheader("Row details")

    df_idx_list = _df_for_view.index.to_list()
    nrows = len(df_idx_list)

    if nrows == 0:
        st.info("No rows to preview after filters.")
    else:
        mode = st.radio("Pick row by", ["Index", "URL"], horizontal=True, key="row_pick_mode", help="Choose how to select which article to inspect: by position in filtered table or by URL.")

        if mode == "Index":
            pos = st.number_input(
                "Row position (0-based, on filtered table)",
                min_value=0,
                max_value=max(nrows - 1, 0),
                value=0,
                step=1,
                help="0 selects the first row currently shown above (after filtering).",
                key="pick_by_index_pos",
            )
            idx = df_idx_list[int(pos)]
            row = _df_for_view.loc[idx]
            with st.expander(f"Row {idx} (position {int(pos)}) — preview", expanded=True):
                _render_row_details(row, idx, df_f, key_prefix="pos")

        else:
            # URL mode
            url_col = _guess_url_col(_df_for_view)
            if not url_col or url_col not in df_f.columns:
                st.warning("No URL column found in this table to select by URL.")
            else:
                tabs = st.tabs(["Pick from list", "Enter URL"])
                with tabs[0]:
                    url_options = _df_for_view[url_col].astype(str).dropna().unique().tolist()
                    # keep UI snappy
                    if len(url_options) > 500:
                        url_options = url_options[:500]
                    sel_url = st.selectbox("Choose URL", ["(choose)"] + url_options, index=0, key="pick_by_url_select", help="Pick an article by URL from the filtered table.")
                with tabs[1]:
                    typed_url = st.text_input("Or paste/enter URL", value="", key="pick_by_url_typed", help="Paste a URL exactly as it appears in the table above.")

                chosen_url = (typed_url or "").strip() or (sel_url if sel_url != "(choose)" else "")
                if chosen_url:
                    # locate first match in either url_col or a common alternative (final_url)
                    candidates = _df_for_view.index[_df_for_view[url_col].astype(str) == chosen_url]
                    if len(candidates) == 0 and "final_url" in _df_for_view.columns:
                        candidates = _df_for_view.index[_df_for_view["final_url"].astype(str) == chosen_url]
                    if len(candidates) > 0:
                        idx2 = candidates[0]
                        row2 = _df_for_view.loc[idx2]
                        with st.expander(f"Row {idx2} (by URL) — preview", expanded=True):
                            _render_row_details(row2, idx2, df_f, key_prefix="url")
                    else:
                        st.warning("Selected/entered URL not found in the filtered table.")

# ------------------ Fullscreen preview panel (rendered full HTML only) -------
if "__vd_full_row_idx" in st.session_state and st.session_state["__vd_full_row_idx"] in _df_for_view.index:
    fidx = st.session_state["__vd_full_row_idx"]
    frow = _df_for_view.loc[fidx]
    base_url = str(frow.get("final_url") or frow.get(_guess_url_col(df_f) or "") or "")

#    st.divider()
    st.subheader(f"Fullscreen preview — Row {fidx}")

    htmls = _load_html_from_row(frow)

    tabs = st.tabs(["Rendered: full_html", "Raw: full_html", "Fields"])
    with tabs[0]:
        if htmls["full_html"]:
            _render_html_block(htmls["full_html"], base_url=base_url, height=900)
            if htmls["source_path"]:
                st.caption(f"Source: `{htmls['source_path']}`")
        else:
            st.info("No full_html_path available or the file could not be read for this row.")

    with tabs[1]:
        if htmls["full_html"]:
            st.code(htmls["full_html"][:200_000], language="html")
        else:
            st.info("No raw full HTML to show.")

    with tabs[2]:
        # Quick field recap
        show_keys = [k for k in [
            "title","subtitle","lead","author","section",
            "published_datetime","updated_datetime","published_date","updated_date",
            "status","reason","template_used","final_url","url","full_html_path"
        ] if k in df_f.columns]
        for k in show_keys:
            st.write(f"**{k}**: {frow.get(k)}")

    cols_close = st.columns([1,3])
    with cols_close[0]:
        if st.button("Close preview", key="btn_close_full_preview"):
            del st.session_state["__vd_full_row_idx"]

# ------------------ Re-extract (saved HTML) ----------------------------------
st.divider()
st.subheader("Re-extract from saved HTML")
rx_hdr = st.columns([1, 2])
with rx_hdr[0]:
    # Header-aligned button
    do_reextract = st.button("Re-extract from saved HTML (make redo CSV)", width='stretch', help="Re-run extraction from saved HTML snapshots—no network fetch. This creates _redo.csv and (if extractor available) _redo_out.csv.")

if do_reextract:
    # pick useful columns
    keep_cols = [c for c in [
        _guess_url_col(df_f),
        "final_url",
        "template_used",
        "container_html_path",
        "full_html_path",
        "title","subtitle","lead","author","section",
        "published_datetime","updated_datetime","published_date","updated_date",
        _guess_text_col(df_f),
        "status","reason"
    ] if c and c in df_f.columns]
    redo = df_f[keep_cols].copy()

    # write manifest sidecar
    redo_name = chosen_path.with_suffix("").name + "_redo.csv"
    out_path = chosen_path.parent / redo_name
    redo.to_csv(out_path, index=False)
    st.success(f"Created redo manifest: {out_path}")

    # Optional: attempt local re-extraction if conty_core is present
    try:
        from conty_core.extract import extract_article_from_html as _extract_html  # hypothetical helper
    except Exception:
        _extract_html = None

    if _extract_html is None:
        st.info("Local re-extraction not available (conty_core.extract helper not found). Use the redo CSV in Builder or implement a local extractor here.")
    else:
        rows = []
        for _, r in redo.iterrows():
            base = str(r.get("final_url") or r.get(_guess_url_col(df_f) or "") or "")
            htmlp = r.get("full_html_path") or r.get("container_html_path")
            if not htmlp or (isinstance(htmlp, float) and np.isnan(htmlp)):
                rows.append(r.to_dict()); continue
            try:
                html_text = Path(str(htmlp)).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                rows.append(r.to_dict()); continue
            try:
                art = _extract_html(html_text, base_url=base)
                for k in ["title","subtitle","lead","author","section","published_datetime","updated_datetime","published_date","updated_date","text","main_image"]:
                    if k in art:
                        r[k] = art[k]
            except Exception:
                pass
            rows.append(r.to_dict())

        out_df = pd.DataFrame(rows)
        out2_path = chosen_path.parent / (chosen_path.with_suffix("").name + "_redo_out.csv")
        out_df.to_csv(out2_path, index=False)
        st.success(f"Local re-extract attempt complete: {out2_path}")

# ------------------ Diff: redo vs redo_out ------------------
#st.divider()
with st.expander("Review Differences: redo vs redo_out (side-by-side)"):
    st.caption("Shows what changed between original extracted rows and the re-extracted versions.")
    # detect sidecar files
    redo_path = chosen_path.parent / (chosen_path.with_suffix("").name + "_redo.csv")
    redo_out_path = chosen_path.parent / (chosen_path.with_suffix("").name + "_redo_out.csv")

    if not redo_path.exists():
        st.info(f"No redo manifest found at: {redo_path.name}. Create it via 'Re-extract from saved HTML'.")
    elif not redo_out_path.exists():
        st.info(f"No redo output found at: {redo_out_path.name}. Run local re-extract or generate it via Builder.")
    else:
        try:
            df_redo = pd.read_csv(redo_path, low_memory=False)
            df_out  = pd.read_csv(redo_out_path, low_memory=False)
        except Exception as e:
            st.error(f"Failed to read diff inputs: {e}")
            df_redo = pd.DataFrame(); df_out = pd.DataFrame()

        if not df_redo.empty and not df_out.empty:
            # choose a join key
            join_key = None
            for candidate in ["final_url","url"]:
                if candidate in df_redo.columns and candidate in df_out.columns:
                    join_key = candidate
                    break
            if not join_key:
                st.error("No common URL column to join on (expected 'final_url' or 'url').")
            else:
                # compute text lengths
                tcol_old = _guess_text_col(df_redo) or "text"
                tcol_new = _guess_text_col(df_out) or "text"
                if tcol_old in df_redo.columns:
                    df_redo["__text_len"] = df_redo[tcol_old].astype(str).str.len()
                if tcol_new in df_out.columns:
                    df_out["__text_len"] = df_out[tcol_new].astype(str).str.len()

                # pick fields to compare
                cmp_fields = ["title","subtitle","lead","author","section","published_datetime","updated_datetime","published_date","updated_date","__text_len","main_image"]
                left = df_redo[[c for c in [join_key] + cmp_fields if c in df_redo.columns]].copy()
                right = df_out[[c for c in [join_key] + cmp_fields if c in df_out.columns]].copy()

                merged = left.merge(right, on=join_key, how="inner", suffixes=("_old","_new"))
                # detect changes
                changed_cols = []
                for f in cmp_fields:
                    col_old, col_new = f + "_old", f + "_new"
                    if col_old in merged.columns and col_new in merged.columns:
                        ch = (merged[col_old].astype(str) != merged[col_new].astype(str))
                        merged[f"chg_{f}"] = ch
                        changed_cols.append(f"chg_{f}")

                merged["changed_any"] = merged[[c for c in changed_cols if c in merged.columns]].any(axis=1) if changed_cols else False

                st.caption(f"Joined on `{join_key}` — {len(merged):,} rows")
                only_changed = st.checkbox("Show only changed rows", value=True, key="diff_only_changed")
                if only_changed:
                    merged = merged[merged["changed_any"]]

                # A compact view
                view_cols = [join_key,
                             "title_old","title_new",
                             "published_datetime_old","published_datetime_new",
                             "updated_datetime_old","updated_datetime_new",
                             "__text_len_old","__text_len_new",
                             "changed_any"]
                view_cols = [c for c in view_cols if c in merged.columns]
                st.dataframe(merged[view_cols], width='stretch', height=360)

                # Download diff
                st.download_button(
                    "Download diff CSV",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name=f"diff_{chosen_path.stem}.csv",
                    mime="text/csv",
                )

# --- Apply redo to base articles CSV (replace matching rows) ---
apply_cols = st.columns([1,2])
with apply_cols[0]:
    if st.button("Apply redo_out → replace rows in base CSV", help="Replace the matching rows in the main _articles.csv using the re-extracted data. A .bak backup is always saved."):
        try:
            base = pd.read_csv(chosen_path, low_memory=False)
            redo_out_path = chosen_path.parent / (chosen_path.with_suffix("").name + "_redo_out.csv")
            if not redo_out_path.exists():
                st.error("No redo_out CSV found. Run 'Re-extract from saved HTML' first.")
            else:
                redo_out = pd.read_csv(redo_out_path, low_memory=False)

                # join key
                join_key = "final_url" if ("final_url" in base.columns and "final_url" in redo_out.columns) else (
                           "url" if ("url" in base.columns and "url" in redo_out.columns) else None)
                if not join_key:
                    st.error("No common key to join on (need 'final_url' or 'url').")
                else:
                    # index for replacement
                    base_idxed = base.set_index(join_key)
                    redo_idxed = redo_out.set_index(join_key)

                    # align schemas (only replace columns that exist in base)
                    replace_cols = [c for c in redo_idxed.columns if c in base_idxed.columns and c not in [join_key]]

                    # replace values for the keys that exist in both
                    common_keys = base_idxed.index.intersection(redo_idxed.index)
                    base_idxed.loc[common_keys, replace_cols] = redo_idxed.loc[common_keys, replace_cols]

                    # backup then write
                    backup = chosen_path.with_suffix(".bak")
                    base.to_csv(backup, index=False)
                    base_idxed.reset_index().to_csv(chosen_path, index=False)
                    st.success(f"Replaced {len(common_keys):,} rows in {chosen_path.name}. Backup saved as {backup.name}.")
        except Exception as e:
            st.error(f"Apply redo failed: {e}")

# ------------------ Simple stats ------------------
st.divider()
st.subheader("Simple Stats")

with st.expander("Stats"):
    cols = st.columns(3)
    with cols[0]:
        st.write("Top authors")
        if "author" in df_f.columns:
            top_auth = df_f["author"].fillna("").value_counts().head(10)
            st.table(top_auth)
    with cols[1]:
        st.write("Top sections")
        if "section" in df_f.columns:
            top_sec = df_f["section"].fillna("").value_counts().head(10)
            st.table(top_sec)
    with cols[2]:
        st.write("Top tags")
        if "tags" in df_f.columns:
            top_tags = df_f["tags"].fillna("").value_counts().head(10)            
            st.table(top_tags)
