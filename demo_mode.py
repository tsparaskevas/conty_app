import os
import streamlit as st

DEMO_MODE = bool(os.environ.get("CONTY_DEMO", "").strip())


def show_demo_banner() -> None:
    """Small banner in the sidebar when running in demo mode."""
    if not DEMO_MODE:
        return
    st.sidebar.info(
        "**Demo mode**\n\n"
        "- Changes to scrapers/templates are **not saved** to disk.\n"
        "- Runs are capped so that the demo stays snappy.",
    )

