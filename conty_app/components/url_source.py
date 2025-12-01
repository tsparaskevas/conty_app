import streamlit as st
from conty_core.loaders import list_csvs_in_folder, read_urls_from_csv

def url_source_widget(key_prefix: str = "urlsrc"):
    """
    Render a Streamlit widget for selecting a URL, either from a CSV in a folder
    or by manually pasting a single URL.

    UI behavior:
      - Shows a "URL Source" subheader.
      - Lets the user choose between:
          1. "From CSV in a folder":
             * User types a folder path containing CSV files.
             * Lists available CSVs in that folder (using list_csvs_in_folder).
             * User picks a CSV and optionally a URL column name.
             * Reads URLs from that CSV (using read_urls_from_csv).
             * User can filter the list of URLs with a text query.
             * User selects one URL from the filtered list.
          2. "Paste single URL":
             * User manually types/pastes a URL in a text input.
             * Shows an error if the URL does not start with http:// or https://.

    Args:
        key_prefix: Prefix used for Streamlit widget keys so that multiple
                    instances of this widget can coexist on the same page.

    Returns:
        dict: A dictionary describing the current selection with keys:
            - "mode": either "csv" or "single".
            - "folder": folder path used when mode == "csv", else None.
            - "csv": stringified path/name of the chosen CSV when mode == "csv",
                     or None if no CSV is chosen or mode == "single".
            - "url": the selected URL string (from CSV or typed), or None if
                    no valid URL is currently selected.
    """
    st.subheader("URL Source")
    mode = st.radio("Pick source", ["From CSV in a folder", "Paste single URL"], key=f"{key_prefix}_mode", horizontal=True)
    selected_url, chosen_csv = None, None

    if mode == "From CSV in a folder":
        folder = st.text_input("Folder path with CSV files", key=f"{key_prefix}_folder", placeholder="data/inputs/articles")
        csvs = list_csvs_in_folder(folder) if folder else []
        if not csvs:
            st.info("No CSVs found in that folder yet.")
            return {"mode": "csv", "folder": folder, "csv": None, "url": None}

        csv_map = {c.name: c for c in csvs}
        csv_name = st.selectbox("Choose CSV", list(csv_map.keys()), key=f"{key_prefix}_csvname")
        chosen_csv = csv_map[csv_name]
        url_col = st.text_input("URL column name (blank = auto)", key=f"{key_prefix}_col", value="")
        urls = read_urls_from_csv(chosen_csv, url_col or None)

        q = st.text_input("Filter URLs (optional)", key=f"{key_prefix}_filter", placeholder="type to filter")
        if q:
            urls = [u for u in urls if q.lower() in u.lower()]

        if urls:
            selected_url = st.selectbox("Pick URL", urls, key=f"{key_prefix}_url")
        else:
            st.warning("No valid http(s) URLs detected.")
        return {"mode": "csv", "folder": folder, "csv": str(chosen_csv) if chosen_csv else None, "url": selected_url}

    # Paste single
    typed_url = st.text_input("Article URL", key=f"{key_prefix}_typed", placeholder="https://example.com/news/...")
    if typed_url and not typed_url.startswith(("http://","https://")):
        st.error("URL must start with http:// or https://")
    return {"mode": "single", "folder": None, "csv": None, "url": typed_url or None}
