"""
Visual Analytics Suite — single-page Streamlit application.

Run with:   streamlit run app.py

This file only wires things together: page setup, sidebar, header and one
tab per capability. All logic lives in the `vas/` package.
"""

import streamlit as st

from config import APP_SETTINGS, ensure_directories

st.set_page_config(page_title=APP_SETTINGS.app_title, page_icon=":material/policy:", layout="wide",
                   initial_sidebar_state="expanded")

from vas.ui.components import render_header  # noqa: E402  (imports after set_page_config)
from vas.ui.sidebar import render_sidebar  # noqa: E402
from vas.ui.styles import inject_css  # noqa: E402
from vas.ui.tabs import (ask_tab, connections_tab, document_tab, gallery_tab, grounding_tab,  # noqa: E402
                         summary_tab, tagging_tab)

TABS = [
    (":material/photo_library: Case Gallery", gallery_tab),
    (":material/forum: Ask", ask_tab),
    (":material/center_focus_strong: Show Me Where", grounding_tab),
    (":material/sell: Tags", tagging_tab),
    (":material/summarize: Summary", summary_tab),
    (":material/hub: Connect the Dots", connections_tab),
    (":material/document_scanner: Documents & OCR", document_tab),
]


def render_welcome() -> None:
    """Shown when no case exists yet."""
    st.markdown("### Welcome")
    st.markdown(
        "1. Create a **new case** or click **Load sample case** for a ready-made demo.\n"
        "2. Upload images, pick an **active image**, and use the tabs to ask, highlight, tag, summarise, "
        "connect and read documents."
    )


def main() -> None:
    ensure_directories()
    inject_css()
    ctx, backend_ok, backend_msg = render_sidebar()
    render_header(ctx, backend_ok, backend_msg)
    if ctx.case is None:
        render_welcome()
        return
    for tab, (_, module) in zip(st.tabs([label for label, _ in TABS]), TABS):
        with tab:
            module.render(ctx)


if __name__ == "__main__":
    main()
