import json
from pathlib import Path
import streamlit as st
import os

DEMO_MODE = bool(os.environ.get("CONTY_DEMO", "").strip())

def show_demo_banner():
    if DEMO_MODE:
        st.info(
            "🔒 **Demo mode** is ON. This is a read-only public demo using data under `data/demo/`."
        )

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="conty · News Article Scraper Suite", page_icon="📰", layout="wide")

# Tiny helpers
def _safe_page_link(path: str, label: str, icon: str = "", key: str | None = None):
    """
    Try to render a page_link to another app page. If the file isn't present
    (e.g., renamed/hidden), fall back to a button with a short instruction.
    """
    try:
        st.page_link(path, label=f"{icon} {label}".strip(), use_container_width=True)
    except Exception:
        st.button(f"{icon} {label}".strip(), key=key, use_container_width=True, disabled=True, help=f"Add {path} to /app/pages to enable this link.")

# ── Hero / intro ──────────────────────────────────────────────────────────────
st.markdown(
    """
# 📰 conty
""")

show_demo_banner()

if DEMO_MODE:
    st.success(
        "You are running the **read-only demo**. "
        "All examples use bundled data under `data/demo/`. "
        "To enable full functionality, unset `CONTY_DEMO` and provide your own data."
    )
    with st.expander("Demo walkthrough", expanded=False):
        st.markdown(
            """
    1. **Browse the pages from the left sidebar.**  
       - *Build/Edit a Scraper* → see how a YAML template + selectors turn raw HTML into an article.  
       - *Run scrapes* → inspect how teaser CSVs are turned into article rows.  
       - *View Data* / *Explore Data* → look at the article tables and basic summaries.

    2. **In demo mode, use the sample data.**  
       - Fetch/extraction runs are limited to bundled examples.  
       - You can still open HTML previews, inspect selectors, and see the final article fields.

    3. **Use this as a read-only tour.**  
       Walk through the pipeline end-to-end without needing your own URLs, CSVs, or write access.
            """
        )

col_l, col_r = st.columns([7, 5])
with col_l:
    st.markdown(
        """
Build, run, and audit news article scrapers — fast.

Use the tools on the left to **design selectors**, **run scrapes**, **inspect failures**, **triage data**, and **explore results**.
"""
    )
    st.markdown(
        """
**What you can do here**
- **Build / edit / test** site scrapers (YAML templates)
- **Run scrapes** from teaser CSVs → article datasets
- **Review run logs** (hard & soft failures)
- **Filter & preview data**, re-extract from saved HTML, and patch rows
- **Explore** volumes, sections, timelines, and keyword trends across sites
"""
    )
with col_r:
    with st.container(border=True):
        st.markdown("**Quick start**")
        st.markdown(
            "1. Go to **Build a scraper** → create selectors and test on a few URLs.\n"
            "2. Go to **Run scrapes** → convert teaser CSVs into `_articles.csv` (optional HTML snapshots).\n"
            "3. Open **View data / View run logs** → fix gaps."
        )
        st.markdown("---")
        _safe_page_link("pages/1_Build_a_Scraper.py", "Start: Build a scraper", "🧩", key="link_build_start")

st.markdown("---")

# ── Workflow cards ────────────────────────────────────────────────────────────
st.subheader("Workflow")

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("### 🧩 Build / Edit / Test a scraper")
        st.caption("Design templates (selectors, engine hints, consent XPaths). Matrix-test across multiple URLs. Add **Datetime overrides** for unusual date formats.")
        _safe_page_link("pages/1_Build_a_Scraper.py", "Open Build a scraper", "🧩", key="link_build")

with c2:
    with st.container(border=True):
        st.markdown("### ⚙️ Run scrapes")
        st.caption("Turn teaser CSVs into article datasets with normalized datetimes. Supports **single** or **multi-site** mode, per-file metrics, saved HTML snapshots.")
        # Use your actual filename; you've renamed this page to 2_Run_Scrapes.py
        _safe_page_link("pages/2_Run_Scrapes.py", "Open Run scrapes", "⚙️", key="link_run")

with c3:
    with st.container(border=True):
        st.markdown("### 🧾 View run logs")
        st.caption("Browse hard failures (no row written) and soft issues (row written but needs attention). Jump to **View data** or **Run scrapes (targeted)** quickly.")
        _safe_page_link("pages/3_View_Run_Logs.py", "Open View run logs", "🧾", key="link_logs")

c4, c5 = st.columns(2)
with c4:
    with st.container(border=True):
        st.markdown("### 🔎 View data")
        st.caption("Filter & preview `_articles.csv`, spot missing/failed dates, short bodies. Re-extract from saved HTML, **diff** changes, and patch rows safely.")
        _safe_page_link("pages/4_View_Data.py", "Open View data", "🔎", key="link_data")

with c5:
    with st.container(border=True):
        st.markdown("### 📊 Explore data")
        st.caption(
            "Interactively explore article datasets: filter by site/category/date, "
            "see articles-per-day timelines, and analyze keyword frequencies per site and overall."
        )
        _safe_page_link("pages/5_Explore_Data.py", "Open Explore data", "📊", key="link_explore")

st.markdown("---")

# ── Helpful notes ─────────────────────────────────────────────────────────────
with st.expander("How things fit together", expanded=False):
    st.markdown(
        """
**Data flow**

- **Teaser CSVs** → (Run scrapes) → **`*_articles.csv`** + optional **full HTML snapshots**  
- **Run logs** → `run_logs/<site>/...__failures.csv` & `...__softfailures.csv`  
- **View data** reads `*_articles.csv` and can re-extract from saved HTML without refetching

**Soft vs Hard failures**
- **Hard**: fetch/extract errors or empty title — row not written (see `__failures.csv`)
- **Soft**: row written but flagged: missing both datetimes, parse failures, short body, etc. (see `__softfailures.csv`)
"""
    )

with st.expander("Tips for accuracy & speed", expanded=False):
    st.markdown(
        """
- Prefer **saved HTML** when debugging to avoid network noise and layout drift.
- If bodies are short, fix **Main container / Body container** first.
- In **Run scrapes**, use a small **Max URLs per CSV** to iterate quickly.
- Use **Multi-site** mode for batch runs; per-site summaries help spot outliers fast.
"""
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="opacity:.7; font-size: 0.9rem; margin-top: .75rem;">
Built with Streamlit. Data paths: <code>data/scrapers</code>, <code>data/outputs/articles</code>, <code>run_logs/&lt;site&gt;</code>, <code>data/tmp</code>.
</div>
""",
    unsafe_allow_html=True,
)

