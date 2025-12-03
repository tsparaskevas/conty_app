import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import time, json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="5 · View Run Logs (Failures)", layout="wide")
st.title("View Run Logs (Failures)")
st.caption("**Hard failures** = no acceptable row yet (e.g., fetch/extract/no-title).  "
           "**Soft failures** = row was written but needs attention (missing/failed dates, short body).")

# --- Quick guide --------------------------------------------------------------
with st.expander("How this page works (quick guide)", expanded=False):
    st.markdown(
        """
**Goal:** Triage and act on failures emitted by *Run Scrapes*.

### What's included
- **Metrics**: counts for **Hard failures**, **Soft issues**, and **Unique URLs**.
- **Two tabs**:
  - **Hard failures** (`__failures.csv`): rows that didn’t make it into `_articles.csv` (e.g., fetch_error, extract_error, no_title).
  - **Soft failures** (`__softfailures.csv`): rows that exist but need attention (e.g., missing_both_dt, parse_failed_pub, short_body).

### Typical flow
1) **Pick a site** (auto-selects the first site that currently has hard failures).
2) In a tab, optionally **limit to a specific file** (one `__failures.csv` or `__softfailures.csv`).
3) (Soft only) **Filter by failure type** to focus your triage.
4) **Select URLs** and:
   - **Send to Builder (single URL)** to test/edit selectors on a single page.
   - **Send to Builder (matrix set)** to test multiple pages at once.
5) Optional file-level action:
   - **Hard**: *Ignore these failures* (deletes that `__failures.csv`).
   - **Soft**: *Remove this soft-failures file* (deletes that `__softfailures.csv`).

### When to use what
- **Hard tab → Fix Failures (Run Scrapes page)** to rescrape and recover rows that were missing.
- **Soft tab → View Data or Targeted re-scrape** to update rows that exist but have issues.
        """
    )

OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
RUN_LOGS_DIR = OUTPUTS_DIR / "run_logs"
TMP_DIR = PROJECT_ROOT / "data" / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
HANDOFF_SINGLE = TMP_DIR / "builder_single_url.json"
HANDOFF_MATRIX = TMP_DIR / "builder_matrix_urls.json"

def _sites_with_logs():
    if not RUN_LOGS_DIR.exists():
        return []
    return sorted([p.name for p in RUN_LOGS_DIR.iterdir() if p.is_dir()])

def _site_has_any_failures(site: str) -> bool:
    base = RUN_LOGS_DIR / site
    return any(base.glob("*__failures.csv")) or any(base.glob("*__softfailures.csv"))

def _load_failures(site: str):
    base = RUN_LOGS_DIR / site
    if not base.exists():
        return pd.DataFrame(), []
    files = sorted(base.glob("*__failures.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__fail_path"] = str(f.relative_to(PROJECT_ROOT))
            dfs.append(df)
        except Exception:
            continue
    return (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()), files

def _load_softfailures(site: str):
    base = RUN_LOGS_DIR / site
    if not base.exists():
        return pd.DataFrame(), []
    files = sorted(base.glob("*__softfailures.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__soft_path"] = str(f.relative_to(PROJECT_ROOT))
            dfs.append(df)
        except Exception:
            continue
    return (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()), files

def _pick_url_col(df):
    """Return the most plausible URL column name in df, or None."""
    if df is None or df.empty:
        return None
    # exact preferences first
    for cand in ["url", "final_url", "page_url", "link"]:
        if cand in df.columns:
            return cand
        # case-insensitive match
        for c in df.columns:
            if c.lower() == cand:
                return c
    # any column that contains 'url'
    for c in df.columns:
        if "url" in c.lower():
            return c
    return None


with st.sidebar:
    sites = _sites_with_logs()
    if not sites:
        st.info("No run logs yet.")
        st.stop()

    # If none of the sites have failures (hard or soft), tell the user and stop.
    if not any(_site_has_any_failures(s) for s in sites):
        st.success("No hard or soft failures found for any site.")
        st.caption("When new runs produce failures, they’ll appear here.")
        st.stop()

    # Otherwise, auto-select the first site that has failures.
    default_idx = 0
    for i, s in enumerate(sites):
        if _site_has_any_failures(s):
            default_idx = i
            break

    site = st.selectbox(
        "Site",
        options=sites,
        index=default_idx,
        help="Auto-selects the first site that currently has hard/soft failures."
    )
    st.caption("Looks for files like `data/outputs/run_logs/<site>/*__failures.csv` and `*__softfailures.csv`.")


if not site:
    st.info("No sites with failures yet.")
    st.stop()

# Load hard and soft files
df_hard, files_hard = _load_failures(site)
df_soft, files_soft = _load_softfailures(site)

if df_hard.empty and df_soft.empty:
    st.success("No failures for this site")
    st.stop()

# Metrics
m = st.columns(3)
with m[0]:
    st.metric("Hard failures", f"{len(df_hard):,}")
with m[1]:
    st.metric("Soft issues", f"{len(df_soft):,}")
with m[2]:
    uc_h = _pick_url_col(df_hard)
    uc_s = _pick_url_col(df_soft)
    urls_h = df_hard[uc_h].dropna().astype(str).tolist() if uc_h else []
    urls_s = df_soft[uc_s].dropna().astype(str).tolist() if uc_s else []
    total_urls = len(set(urls_h + urls_s))
    st.metric("Unique URLs", f"{total_urls:,}")

# ----------- Tabs for Hard and Soft Failures ------------
tabs = st.tabs(["Hard failures", "Soft failures"])

# ---------------------- Tab 1: Hard failures ----------------------
with tabs[0]:
    st.subheader(f"Hard failures for **{site}**")
    wanted_h = ["url","failure_type","teasers_csv","when","template_used","full_html_path","__fail_path"]
    show_h = [c for c in wanted_h if c in df_hard.columns]
    dfh_disp = df_hard[show_h] if show_h else df_hard
    sort_by = ["when"] if "when" in dfh_disp.columns else (show_h if show_h else None)
    if sort_by:
        dfh_disp = dfh_disp.sort_values(by=sort_by, ascending=False)
    st.dataframe(dfh_disp, width="stretch", height=420)

    # Limit to a specific failures file
    fail_files = ["(all)"] + [f.name for f in files_hard]
    pick_src = st.selectbox(
        "Limit to a specific hard-failures file",
        options=fail_files, index=0,
        help="Filter the table and actions to a single __failures.csv."
    )
    if pick_src != "(all)":
        sel_path = next((p for p in files_hard if p.name == pick_src), None)
        if sel_path is not None:
            if st.button("Ignore these failures (delete file)"):
                sel_path.unlink(missing_ok=True)
                st.warning(f"Removed {sel_path.name}")
                st.rerun()
    df_h_view = df_hard if pick_src == "(all)" else df_hard[df_hard["__fail_path"].str.endswith(pick_src)]

    # Pick URLs and send to Builder
    ucol_h = _pick_url_col(df_h_view)
    if not ucol_h:
        st.warning(f"No URL-like column found in failures file. Columns: {list(df_h_view.columns)}")
        urls_h = []
    else:
        urls_h = df_h_view[ucol_h].dropna().astype(str).unique().tolist()

    st.caption(f"{len(urls_h)} failing URL(s) in selection.")
    if "fail_picks_hard" not in st.session_state:
        st.session_state["fail_picks_hard"] = []
    def _toggle_all_h():
        st.session_state["fail_picks_hard"] = list(urls_h) if st.session_state.get("pick_all_h", False) else []
    st.checkbox("Select all", value=False, key="pick_all_h", on_change=_toggle_all_h)
    st.multiselect("Pick URLs to send to Builder", options=urls_h, key="fail_picks_hard", help="Choose the failing URLs you want to open in Build a Scraper (single URL or Matrix test)."
)

    c1, c2 = st.columns(2)
    with c1:
        btn1 = st.button("Send to Builder (single URL)", key=f"hard_send_single_{site}", disabled=(len(st.session_state["fail_picks_hard"]) != 1),
                         help="Send exactly one URL to Builder → B) Article page URL(s).")
        if btn1:
            payload = {"ts": time.time(), "url": st.session_state["fail_picks_hard"][0], "site_hint": site}
            HANDOFF_SINGLE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            st.success("Sent 1 URL to Builder → B) Article page URL(s).")
    with c2:
        btn2 = st.button("Send to Builder (matrix set)", key=f"hard_send_matrix_{site}", disabled=(len(st.session_state["fail_picks_hard"]) == 0),
                         help="Send selected URLs to Builder → K) Matrix test.")
        if btn2:
            payload = {"ts": time.time(), "urls": st.session_state["fail_picks_hard"], "site_hint": site}
            HANDOFF_MATRIX.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            st.success(f"Sent {len(st.session_state['fail_picks_hard'])} URL(s) to Builder → K) Matrix test.")

# ---------------------- Tab 2: Soft failures ----------------------
with tabs[1]:
    hdr = st.columns([3, 2])
    with hdr[0]:
        st.subheader(f"Soft failures for **{site}**")
    with hdr[1]:
        # Convenience links
        c = st.columns(1)
        with c[0]:
            try:
                st.page_link(
                    "app/pages/4_View_Data.py",
                    label="Open View Data for this site",
                    width="stretch",
                )
                st.session_state["__viewdata_site_hint"] = site
            except Exception:
                if st.button("Open 'View Data' (sidebar link)", width="stretch"):
                    st.session_state["__viewdata_site_hint"] = site
                    st.success("Site hint set. Now click “View Data” in the sidebar.")

    if df_soft.empty:
        st.info("No soft issues for this site.")
    else:
        wanted_s = ["url","failure_type","teasers_csv","when","template_used","full_html_path","__soft_path"]
        show_s = [c for c in wanted_s if c in df_soft.columns]

        # Filters: by file, by failure_type
        soft_files = ["(all)"] + [f.name for f in files_soft]
        pick_soft = st.selectbox(
            "Limit to a specific soft-failures file",
            options=soft_files, index=0,
            help="Filter the table and actions to a single __softfailures.csv."
        )
        df_s_view = df_soft if pick_soft == "(all)" else df_soft[df_soft["__soft_path"].str.endswith(pick_soft)]

        types = sorted(df_s_view["failure_type"].dropna().astype(str).str.split(";").explode().str.strip().unique().tolist()) if "failure_type" in df_s_view.columns else []
        filt_types = st.multiselect(
            "Filter by soft failure type",
            options=types, default=types,
            help="Common soft issues: missing_both_dt, parse_failed_pub, parse_failed_upd, short_body."
        )
        if filt_types:
            df_s_view = df_s_view[df_s_view["failure_type"].fillna("").apply(lambda s: any(ft in s for ft in filt_types))]

        dfsv_disp = df_s_view[show_s] if show_s else df_s_view
        sort_by_s = ["when"] if "when" in dfsv_disp.columns else (show_s if show_s else None)
        if sort_by_s:
            dfsv_disp = dfsv_disp.sort_values(by=sort_by_s, ascending=False)
        st.dataframe(dfsv_disp, width="stretch", height=420)

        # Optional: delete selected soft file
        if pick_soft != "(all)":
            sel_soft = next((p for p in files_soft if p.name == pick_soft), None)
            if sel_soft is not None and st.button("Remove this soft-failures file"):
                sel_soft.unlink(missing_ok=True)
                st.warning(f"Removed {sel_soft.name}")
                st.rerun()

        # Pick URLs and send to Builder
        ucol_s = _pick_url_col(df_s_view)
        if not ucol_s:
            st.warning(f"No URL-like column found in soft-failures file. Columns: {list(df_s_view.columns)}")
            urls_s = []
        else:
            urls_s = df_s_view[ucol_s].dropna().astype(str).unique().tolist()


        st.caption(f"{len(urls_s)} soft-issue URL(s) in selection.")
        if "fail_picks_soft" not in st.session_state:
            st.session_state["fail_picks_soft"] = []
        def _toggle_all_s():
            st.session_state["fail_picks_soft"] = list(urls_s) if st.session_state.get("pick_all_s", False) else []
        st.checkbox("Select all", value=False, key="pick_all_s", on_change=_toggle_all_s)
        st.multiselect("Pick URLs to send to Builder", options=urls_s, key="fail_picks_soft", help="Choose the soft-issue URLs you want to open in Build a Scraper (single URL or Matrix test).")

        d1, d2 = st.columns(2)
        with d1:
            b1 = st.button("Send to Builder (single URL)", disabled=(len(st.session_state["fail_picks_soft"]) != 1), key=f"soft_send_single_{site}",
                           help="Send exactly one URL to Builder → B) Article page URL(s).")
            if b1:
                payload = {"ts": time.time(), "url": st.session_state["fail_picks_soft"][0], "site_hint": site}
                HANDOFF_SINGLE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                st.success("Sent 1 URL to Builder → B) Article page URL(s).")
        with d2:
            b2 = st.button("Send to Builder (matrix set)", disabled=(len(st.session_state["fail_picks_soft"]) == 0), key=f"soft_send_matrix_{site}",
                           help="Send selected URLs to Builder → K) Matrix test.")
            if b2:
                payload = {"ts": time.time(), "urls": st.session_state["fail_picks_soft"], "site_hint": site}
                HANDOFF_MATRIX.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                st.success(f"Sent {len(st.session_state['fail_picks_soft'])} URL(s) to Builder → K) Matrix test.")

    st.divider()
    # --- Open Run Scrapes → Targeted re-scrape for this site ---------------------
    col_rs = st.columns([2, 3])
    with col_rs[0]:
        # Write selected URLs (if any) to builder_matrix_urls.json so Run Scrapes can pick them up
        seed_subset = st.checkbox("Seed selected URLs to Run Scrapes", value=True,
                                  help="If ON, writes selected URLs to data/tmp/builder_matrix_urls.json so Targeted re-scrape can use them immediately.")
    with col_rs[1]:
        try:
            if st.page_link("app/pages/2_Run_Scrapes.py", label="Open Run Scrapes → Targeted re-scrape", width="stretch"):
                pass
            st.session_state["__runscrapes_site_hint"] = site
            if seed_subset and st.session_state.get("fail_picks_soft"):
                payload = {
                    "ts": time.time(),
                    "source": "view_run_logs_soft",
                    "site_hint": site,
                    "urls": st.session_state["fail_picks_soft"],
                }
                (TMP_DIR / "builder_matrix_urls.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            if st.button("Open 'Run Scrapes' (sidebar link)", width="stretch"):
                st.session_state["__runscrapes_site_hint"] = site
                if seed_subset and st.session_state.get("fail_picks_soft"):
                    payload = {
                        "ts": time.time(),
                        "source": "view_run_logs_soft",
                        "site_hint": site,
                        "urls": st.session_state["fail_picks_soft"],
                    }
                    (TMP_DIR / "builder_matrix_urls.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                st.success("Site hint set (and URLs seeded if selected). Now click “Run Scrapes” in the sidebar.")


