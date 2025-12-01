import re
import os
from pathlib import Path
from typing import List, Dict, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEMO_MODE = bool(os.environ.get("CONTY_DEMO", "").strip())

DATA_DIR = Path("data")
if DEMO_MODE:
    ARTICLES_DIR = DATA_DIR / "demo" / "outputs"
else:
    ARTICLES_DIR = DATA_DIR / "outputs" / "articles"

# Expected filename pattern: site_second_third_articles.csv
FILENAME_RE = re.compile(r"^(?P<site>[^_]+)_(?P<second>[^_]+)_(?P<third>[^_]+)_articles\.csv$")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _scan_article_files(base_dir: Path) -> pd.DataFrame:
    """
    Scan base_dir for *_articles.csv and parse:
      - site (first part)
      - second (second part)
      - third (third part)
    Return a DataFrame: [path, filename, site, second, third]
    """
    rows: List[Dict[str, str]] = []
    for p in sorted(base_dir.glob("*_articles.csv")):
        m = FILENAME_RE.match(p.name)
        if not m:
            # ignore files not matching the expected pattern
            continue
        rows.append({
            "path": str(p),
            "filename": p.name,
            "site": m.group("site"),
            "second": m.group("second"),
            "third": m.group("third"),
        })
    if not rows:
        return pd.DataFrame(columns=["path", "filename", "site", "second", "third"])
    return pd.DataFrame(rows)


def _load_articles(paths: List[str]) -> pd.DataFrame:
    """
    Load and concatenate article CSVs from the given paths.
    Assumes Run Scrapes FIELDNAMES (including site, author, section,
    published_date/datetime, text, etc.)
    """
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_file"] = Path(p).name
            frames.append(df)
        except Exception as e:
            st.warning(f"Could not read {p}: {e}")
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)

    if "published_date" in df_all.columns:
        s = df_all["published_date"].astype("string")
        pub_dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
        df_all["published_date"] = pub_dt
    else:
        df_all["published_date"] = pd.NaT

    # Ensure site column: fallback from filename if missing
    if "site" not in df_all.columns:
        df_all["site"] = df_all["__source_file"].str.extract(r"^([^_]+)_")[0]

    # Ensure string columns exist
    for col in ["author", "section", "title", "lead", "text", "tags"]:
        if col not in df_all.columns:
            df_all[col] = ""

    return df_all


def _compute_stats(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty:
        return {
            "articles": "0",
            "sites": "0",
            "authors": "0",
            "sections": "0",
            "min_date": "-",
            "max_date": "-",
        }
    articles = len(df)
    sites = df["site"].nunique() if "site" in df.columns else 0
    authors = df["author"].replace("", np.nan).nunique()
    sections = df["section"].replace("", np.nan).nunique()

    if "published_date" in df.columns:
        dates = df["published_date"].dropna()
        if dates.empty:
            min_date = max_date = "-"
        else:
            min_date = dates.min().date().isoformat()
            max_date = dates.max().date().isoformat()
    else:
        min_date = max_date = "-"

    return {
        "articles": str(articles),
        "sites": str(sites),
        "authors": str(authors),
        "sections": str(sections),
        "min_date": min_date,
        "max_date": max_date,
    }


def _render_stats(stats: Dict[str, str], label: str = "Current selection") -> None:
    st.markdown(f"#### Stats — {label}")
    # Simple "card" styling with HTML
    cols = st.columns(4)
    cards = [
        ("Articles", stats["articles"]),
        ("Sites", stats["sites"]),
        ("Authors", stats["authors"]),
        ("Sections", stats["sections"]),
    ]
    for col, (title, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color:#f9fafb;
                    border-radius:0.75rem;
                    padding:0.75rem 1rem;
                    border:1px solid #e5e7eb;
                    box-shadow:0 1px 2px rgba(15,23,42,0.05);
                ">
                    <div style="font-size:0.8rem;color:#6b7280;">{title}</div>
                    <div style="font-size:1.4rem;font-weight:600;color:#111827;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col_min, col_max = st.columns(2)
    with col_min:
        st.markdown(
            f"""
            <div style="
                margin-top:0.5rem;
                background-color:#f9fafb;
                border-radius:0.75rem;
                padding:0.5rem 0.75rem;
                border:1px solid #e5e7eb;
            ">
                <div style="font-size:0.8rem;color:#6b7280;">Min date</div>
                <div style="font-size:1rem;font-weight:500;color:#111827;">{stats["min_date"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_max:
        st.markdown(
            f"""
            <div style="
                margin-top:0.5rem;
                background-color:#f9fafb;
                border-radius:0.75rem;
                padding:0.5rem 0.75rem;
                border:1px solid #e5e7eb;
            ">
                <div style="font-size:0.8rem;color:#6b7280;">Max date</div>
                <div style="font-size:1rem;font-weight:500;color:#111827;">{stats["max_date"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _prepare_articles_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute article counts per (site, published_date) and for ALL sites.
    Returns DataFrame with columns: [site, published_date, articles]
    """
    if df.empty or "published_date" not in df.columns:
        return pd.DataFrame(columns=["site", "published_date", "articles"])

    tmp = df.dropna(subset=["published_date"]).copy()
    tmp["published_date"] = tmp["published_date"].dt.date

    per_site = (
        tmp.groupby(["site", "published_date"])
        .size()
        .reset_index(name="articles")
    )

    # Global ALL across sites
    per_all = (
        tmp.groupby(["published_date"])
        .size()
        .reset_index(name="articles")
    )
    per_all.insert(0, "site", "ALL")

    return pd.concat([per_site, per_all], ignore_index=True)


def _chart_articles_per_day_precomputed(grp: pd.DataFrame, site_choice: str) -> None:
    """
    Single-line chart of article counts per day, using precomputed group data.
    """
    if grp.empty:
        st.info("No articles with valid published_date to plot.")
        return

    data = grp.copy()
    if site_choice != "ALL":
        data = data[data["site"] == site_choice]
    else:
        data = data[data["site"] == "ALL"]

    if data.empty:
        st.info("No data for the selected site / date range.")
        return

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("published_date:T", title="Date"),
            y=alt.Y("articles:Q", title="Articles per day"),
            tooltip=["published_date:T", "articles:Q"],
        )
        .properties(height=400)
    )

    st.altair_chart(chart, use_container_width=True)

    # Export data for this chart
    csv_articles = data.sort_values(["site", "published_date"]).to_csv(index=False)
    st.download_button(
        "Download articles-per-day data as CSV",
        data=csv_articles,
        file_name="articles_per_day.csv",
        mime="text/csv",
    )


def _count_keywords(
    df: pd.DataFrame,
    words: List[str],
    case_sensitive: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Count keyword occurrences per day per site (based on published_date).

    Returns:
      - per_day: columns [site, published_date, word, count]
      - totals:  columns [site, word, total_count]
    """
    if df.empty or not words:
        return pd.DataFrame(), pd.DataFrame()

    # Build a text blob per row from title + lead + text
    blob = (
        df["title"].fillna("").astype(str)
        + " "
        + df["lead"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
    )

    flags = 0 if case_sensitive else re.IGNORECASE

    counts_per_word = {}
    for w in words:
        if not w:
            continue
        pattern = re.compile(rf"\b{re.escape(w)}\b", flags)
        counts = blob.map(lambda s: len(pattern.findall(s)))
        counts_per_word[w] = counts

    if not counts_per_word:
        return pd.DataFrame(), pd.DataFrame()

    df_counts = df[["site", "published_date"]].copy()
    for w, series in counts_per_word.items():
        df_counts[w] = series

    # Long format per word: one row per (site, date, word)
    per_word_frames = []
    for w in words:
        if w not in df_counts.columns:
            continue
        sub = df_counts[["site", "published_date", w]].copy()
        sub.rename(columns={w: "count"}, inplace=True)
        sub["word"] = w
        per_word_frames.append(sub)

    if not per_word_frames:
        return pd.DataFrame(), pd.DataFrame()

    per_day = pd.concat(per_word_frames, ignore_index=True)
    per_day = per_day[per_day["count"] > 0]
    per_day.dropna(subset=["published_date"], inplace=True)
    per_day["published_date"] = per_day["published_date"].dt.date

    # Totals per site/word
    totals = (
        per_day.groupby(["site", "word"])["count"]
        .sum()
        .reset_index(name="total_count")
    )

    # Add a pseudo-site "ALL" for global totals per word
    global_totals = (
        per_day.groupby(["word"])["count"]
        .sum()
        .reset_index(name="total_count")
    )
    global_totals.insert(0, "site", "ALL")
    totals = pd.concat([totals, global_totals], ignore_index=True)

    # Also add per-day ALL (so we can chart keywords for ALL)
    global_per_day = (
        per_day.groupby(["published_date", "word"])["count"]
        .sum()
        .reset_index(name="count")
    )
    global_per_day.insert(0, "site", "ALL")
    per_day = pd.concat([per_day, global_per_day], ignore_index=True)

    return per_day, totals

def _chart_keywords_for_site(
    per_day: pd.DataFrame,
    site_choice: str,
    normalized: bool = False,
) -> None:
    """
    Keyword chart for ALL or a single site.

    - If normalized is False: line chart, y = absolute counts.
    - If normalized is True: stacked bar chart, y = share per day (sum to 1.0).
    """
    if per_day.empty:
        st.info("No keyword occurrences to plot.")
        return

    data = per_day.copy()
    if site_choice != "ALL":
        data = data[data["site"] == site_choice]
    else:
        data = data[data["site"] == "ALL"]

    if data.empty:
        st.info("No keyword data for the selected site / date range.")
        return

    # Absolute vs normalized
    if normalized:
        # For each date, divide each word's count by total for that date
        totals = data.groupby("published_date")["count"].transform("sum")
        data["value"] = np.where(totals > 0, data["count"] / totals, 0.0)

        y_encoding = alt.Y(
            "value:Q",
            axis=alt.Axis(title="Keyword share per day", format="%"),
            stack="normalize",
        )
        mark_type = "bar"
        tooltip_fields = [
            "word:N",
            "published_date:T",
            "count:Q",
            alt.Tooltip("value:Q", title="share", format=".1%"),
        ]
    else:
        data["value"] = data["count"]
        y_encoding = alt.Y(
            "value:Q",
            axis=alt.Axis(title="Keyword count per day"),
        )
        mark_type = "line"
        tooltip_fields = ["word:N", "published_date:T", "count:Q"]

    selection_word = alt.selection_point(fields=["word"], bind="legend")

    base = (
        alt.Chart(data)
        .encode(
            x=alt.X("published_date:T", title="Date"),
            y=y_encoding,
            color="word:N",
            opacity=alt.condition(selection_word, alt.value(1.0), alt.value(0.25)),
            tooltip=tooltip_fields,
        )
        .add_params(selection_word)
        .properties(height=400)
    )

    if mark_type == "bar":
        chart = base.mark_bar()
    else:
        chart = base.mark_line(point=True)

    st.altair_chart(chart, use_container_width=True)

    # Export data (includes both count and value if normalized)
    csv_kw_day = data.sort_values(["word", "site", "published_date"]).to_csv(index=False)
    st.download_button(
        "Download keyword per-day data (for current site) as CSV",
        data=csv_kw_day,
        file_name="keyword_counts_per_day_site.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="Explore data", layout="wide")

st.title("Explore data")

# 1) Select CSVs from folder
st.sidebar.header("Data source")

base_dir = st.sidebar.text_input(
    "Articles folder",
    value=str(ARTICLES_DIR),
    help="Folder containing *_articles.csv files.",
)
base_dir_path = Path(base_dir)

if not base_dir_path.exists():
    st.error(f"Folder does not exist: {base_dir}")
    st.stop()

df_files = _scan_article_files(base_dir_path)
if df_files.empty:
    st.sidebar.info("No *_articles.csv files found.")
    st.warning("No *_articles.csv files found matching pattern 'site_second_third_articles.csv'.")
    st.stop()

# Site selection
sites_available = sorted(df_files["site"].unique())
sites_selected = st.sidebar.multiselect(
    "Sites",
    options=sites_available,
    default=sites_available,
)

df_files_site = df_files[df_files["site"].isin(sites_selected)]
if df_files_site.empty:
    st.sidebar.info("No files for the selected site(s).")
    st.warning("No files for the selected site(s).")
    st.stop()

# Show only the sites count in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Sites selected:** {len(sites_selected)}")

# Second-part selection (e.g. search, opinion)
seconds_available = sorted(df_files_site["second"].unique())
seconds_selected = st.sidebar.multiselect(
    "Category (2nd part of filename)",
    options=seconds_available,
    default=seconds_available,
)

df_files_sec = df_files_site[df_files_site["second"].isin(seconds_selected)]
if df_files_sec.empty:
    st.sidebar.info("No files for the selected sites/categories.")
    st.warning("No files for the selected site/category combination.")
    st.stop()

# Third-part selection only when "search" is among seconds
third_filter_active = "search" in seconds_selected
if third_filter_active:
    df_search = df_files_sec[df_files_sec["second"] == "search"]
    thirds_available = sorted(df_search["third"].unique())
    thirds_selected = st.sidebar.multiselect(
        "Search topic (3rd part of filename)",
        options=thirds_available,
        default=thirds_available,
    )
    # apply third filter only to "search" rows
    mask_search = (df_files_sec["second"] == "search")
    mask_search_filtered = mask_search & df_files_sec["third"].isin(thirds_selected)
    df_files_final = pd.concat(
        [
            df_files_sec[~mask_search],
            df_files_sec[mask_search_filtered],
        ],
        ignore_index=True,
    )
else:
    df_files_final = df_files_sec

if df_files_final.empty:
    st.sidebar.info("No files match the current filters.")
    st.warning("No files remain after applying all filters.")
    st.stop()

# Load the selected CSVs
df_all = _load_articles(df_files_final["path"].tolist())
if df_all.empty:
    st.warning("No data loaded from the selected files.")
    st.stop()

# Drop articles without a usable published_date (we don't care about them here)
before = len(df_all)
df_all = df_all.dropna(subset=["published_date"]).copy()
after = len(df_all)
st.info(f"Dropped {before - after} articles without published_date.")

st.markdown("### 1. Overview of selected data")
stats_all = _compute_stats(df_all)
_render_stats(stats_all, label="All selected files")

# 3) Filter by date range (using start/end date inputs)
st.markdown("### 2. Date range filter")

if df_all["published_date"].dropna().empty:
    st.info("No valid published_date values; date filtering and date-based charts will be limited.")
    df_filtered = df_all.copy()
else:
    min_date = df_all["published_date"].dropna().min().date()
    max_date = df_all["published_date"].dropna().max().date()

    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
    with col_end:
        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    mask = df_all["published_date"].between(
        pd.to_datetime(start_date),
        pd.to_datetime(end_date),
        inclusive="both",
    )
    df_filtered = df_all[mask].copy()

stats_filtered = _compute_stats(df_filtered)
_render_stats(stats_filtered, label="After date filter")

# Precompute articles-per-day once
articles_per_day = _prepare_articles_per_day(df_filtered)

# 4) Articles per day graph (one line, choose ALL or a single site)
st.markdown("### 3. Articles per day (based on published_date)")

sites_for_chart = sorted(df_filtered["site"].dropna().unique().tolist())
site_choice = st.radio(
    "Show articles per day for:",
    options=["ALL"] + sites_for_chart,
    index=0,
    horizontal=True,
)
_chart_articles_per_day_precomputed(articles_per_day, site_choice)

# 5) Keyword exploration
st.markdown("### 4. Keyword exploration")

kw_input = st.text_input(
    "Enter one or more words (separated by space or comma)",
    value="",
    help="Counts are computed over title + lead + text.",
)

if kw_input.strip():
    # parse words
    tokens = re.split(r"[,\s]+", kw_input.strip())
    words = [t for t in tokens if t]
    case_sensitive = st.checkbox("Case sensitive", value=False)

    per_day_counts, totals = _count_keywords(df_filtered, words, case_sensitive=case_sensitive)

    # 4.1a Stacked normalized bar per site & ALL (keyword shares)
    st.markdown("#### 4.1 Keyword distribution per site (normalized)")

    if not totals.empty:
        bar_data = totals.copy()
        # remove sites that have 0 total for all words (shouldn't happen, but safe)
        bar_data = bar_data[bar_data["total_count"] > 0]

        # stacked normalized bar: site on x, word on color
        bar_chart = (
            alt.Chart(bar_data)
            .mark_bar()
            .encode(
                x=alt.X("site:N", title="Site"),
                y=alt.Y(
                    "total_count:Q",
                    stack="normalize",
                    axis=alt.Axis(format="%", title="Share of keyword counts"),
                ),
                color=alt.Color("word:N", title="Keyword"),
                tooltip=["site:N", "word:N", "total_count:Q"],
            )
            .properties(height=350)
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # export data used for this bar chart (with percentages)
        # compute share per site/word
        sums = bar_data.groupby("site")["total_count"].transform("sum")
        bar_data_export = bar_data.copy()
        bar_data_export["share"] = (bar_data_export["total_count"] / sums * 100).round(2)
        csv_bar = bar_data_export.sort_values(["site", "word"]).to_csv(index=False)
        st.download_button(
            "Download keyword distribution data (for stacked bar) as CSV",
            data=csv_bar,
            file_name="keyword_distribution_per_site.csv",
            mime="text/csv",
        )
    else:
        st.info("No keyword totals to visualize.")

    # 4.1b Keyword counts per day (timelines) with site selector
    st.markdown("#### 4.2 Keyword counts per day")

    if not per_day_counts.empty:
        sites_kw = sorted(per_day_counts["site"].dropna().unique().tolist())
        # ensure ALL is first if present
        if "ALL" in sites_kw:
            sites_kw = ["ALL"] + [s for s in sites_kw if s != "ALL"]

        col_site, col_mode = st.columns([2, 1])
        with col_site:
            site_kw_choice = st.radio(
                "Show keyword timelines for site:",
                options=sites_kw,
                index=0,
                horizontal=True,
                key="kw_site_choice",
            )
        with col_mode:
            normalized = st.radio(
                "Y-axis",
                options=["Counts", "Percentage"],
                index=0,
                horizontal=True,
                key="kw_norm_choice",
            ) == "Percentage"

        _chart_keywords_for_site(
            per_day_counts,
            site_choice=site_kw_choice,
            normalized=normalized,
        )
    else:
        st.info("No keyword occurrences to plot.")

    # 4.3 Totals per site and word (pivot) + articles column
    st.markdown("#### 4.3 Totals per site and word")

    if not totals.empty:
        # Articles per site (for current filtered data)
        articles_per_site = df_filtered.groupby("site").size().rename("articles").reset_index()
        # Add ALL row
        all_row = pd.DataFrame({"site": ["ALL"], "articles": [len(df_filtered)]})
        articles_per_site = pd.concat([articles_per_site, all_row], ignore_index=True)

        pivot_totals = (
            totals
            .pivot_table(index="site", columns="word", values="total_count", fill_value=0)
            .reset_index()
        )
        # Join articles counts
        pivot_totals = pivot_totals.merge(articles_per_site, on="site", how="left")
        # Reorder columns: site, articles, words...
        word_cols = [c for c in pivot_totals.columns if c not in ("site", "articles")]
        pivot_totals = pivot_totals[["site", "articles"] + word_cols]

        st.dataframe(pivot_totals)
    else:
        pivot_totals = pd.DataFrame()
        st.info("No keyword totals to show.")

    # Download buttons for raw per-day and totals
    csv_day_all = per_day_counts.sort_values(["word", "site", "published_date"]).to_csv(index=False)
    csv_tot = pivot_totals.to_csv(index=False)

    st.download_button(
        "Download keyword per-day data (ALL sites) as CSV",
        data=csv_day_all,
        file_name="keyword_counts_per_day_all.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download totals per site (words as columns) as CSV",
        data=csv_tot,
        file_name="keyword_totals_per_site_pivot.csv",
        mime="text/csv",
    )
else:
    st.info("Enter one or more keywords above to see counts and graphs.")

