"""Summary tab (4.4): caption / paragraph / detailed description."""

import streamlit as st

from vas.core.case_store import now_iso
from vas.services.prompts import SUMMARY_LENGTHS
from vas.services.summarization import summarize_image
from vas.ui.components import (image_picker, image_with_quality, no_image_placeholder, show_confidence,
                               verify_banner)
from vas.ui.context import AppContext
from vas.utils.confidence import estimate_confidence


def _summarize(ctx: AppContext, image_id: str, length: str, focus: str) -> None:
    """Summarise one image and save under its length key."""
    case = ctx.case
    image = case.load_image(image_id)
    text = ctx.run_model(
        lambda: summarize_image(ctx.backend, image, length, focus, ctx.settings.max_pixels_single),
        spinner=f"Summarising {case.images[image_id]['filename']}…")
    if text is None:
        return
    rec = case.images[image_id]
    summaries = rec.get("summaries", {})
    summaries[length] = {"text": text, "focus": focus, "created_at": now_iso(),
                         "confidence": estimate_confidence(text, rec["quality"]["score"])}
    case.update_image(image_id, summaries=summaries)


def render(ctx: AppContext) -> None:
    case = ctx.case
    image_id = image_picker(ctx, "summary")
    if not image_id:
        no_image_placeholder()
        return

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        image_with_quality(ctx, image_id)
    with right:
        st.markdown("#### Summarise this image")
        verify_banner()
        length = st.segmented_control("Length", list(SUMMARY_LENGTHS), default="Short paragraph", key="sum_len") \
            or "Short paragraph"
        focus = st.text_input("Focus (optional)", placeholder="e.g. people and vehicles", key="sum_focus")
        if st.button("Generate summary", type="primary", icon=":material/summarize:"):
            _summarize(ctx, image_id, length, focus)

        summaries = case.images[image_id].get("summaries", {})
        if not summaries:
            st.caption("No summaries yet.")
        for name in SUMMARY_LENGTHS:
            if name not in summaries:
                continue
            s = summaries[name]
            with st.container(border=True):
                st.markdown(f"**{name}**" + (f" · focus: *{s['focus']}*" if s.get("focus") else "")
                            + f" <span class='vas-muted'>· {s['created_at'][:16].replace('T', ' ')}</span>",
                            unsafe_allow_html=True)
                st.code(s["text"], language=None, wrap_lines=True)  # code block = one-click copy button
                show_confidence(s.get("confidence"))
