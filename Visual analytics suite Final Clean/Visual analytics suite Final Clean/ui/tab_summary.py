"""Summary tab: one-line caption, short paragraph or detailed description of the active image."""

import streamlit as st

from backend import analysis
from backend.cases import Case, now
from backend.prompts import SUMMARY_LENGTHS
from ui.common import VERIFY_NOTE, image_picker, preview, run_model


def render(case: Case) -> None:
    image_id = image_picker(case, "summary")
    if not image_id:
        return
    left, right = st.columns([1, 1.2], gap="large")
    left.image(preview(case, image_id), caption=case.images[image_id]["filename"], width="stretch")

    with right:
        st.caption(VERIFY_NOTE)
        length = st.segmented_control("Length", list(SUMMARY_LENGTHS), default="Short paragraph") or "Short paragraph"
        focus = st.text_input("Focus (optional)", placeholder="e.g. people and vehicles", key="summary_focus")
        if st.button("Generate summary", type="primary"):
            image = case.load_image(image_id)
            text = run_model(lambda: analysis.summarize(image, length, focus), "Summarising…")
            if text is not None:
                summaries = case.images[image_id].get("summaries", {})
                summaries[length] = {"text": text, "focus": focus.strip(), "created_at": now()}
                case.update_image(image_id, summaries=summaries)

        summaries = case.images[image_id].get("summaries", {})
        if not summaries:
            st.caption("No summaries yet.")
        for name in SUMMARY_LENGTHS:
            if name in summaries:
                summary = summaries[name]
                st.markdown(f"**{name}**" + (f" · focus: *{summary['focus']}*" if summary.get("focus") else "")
                            + f" · {summary['created_at'][:16].replace('T', ' ')}")
                st.code(summary["text"], language=None, wrap_lines=True)  # code block = one-click copy button
