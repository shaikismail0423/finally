"""
Visual Analytics Suite — talk to your images with Qwen2.5-VL.

Run with:  streamlit run app.py

This file only builds the page: the sidebar (cases and images) and one tab per capability.
The tabs are in ui/, the model calls in backend/.
"""

import streamlit as st

from ui import tab_ask, tab_connect, tab_documents, tab_gallery, tab_locate, tab_summary, tab_tags
from ui.sidebar import render_sidebar

TABS = {
    "Case Gallery": tab_gallery.render,
    "Ask": tab_ask.render,
    "Show Me Where": tab_locate.render,
    "Tags": tab_tags.render,
    "Summary": tab_summary.render,
    "Connect the Dots": tab_connect.render,
    "Documents & OCR": tab_documents.render,
}

st.set_page_config(page_title="Visual Analytics Suite", layout="wide")
case = render_sidebar()
st.title("Visual Analytics Suite")
st.caption("Ask, locate, tag, summarise, connect and read images with Qwen2.5-VL"
           + (f" · Case: **{case.name}**" if case else ""))

if case is None:
    st.info("Create a **New case** in the sidebar, or click **Load sample case** for a ready-made demo.")
else:
    for tab, render in zip(st.tabs(list(TABS)), TABS.values()):
        with tab:
            render(case)
