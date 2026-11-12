"""Ask tab (4.1): chat about the active image, with follow-up memory."""

import streamlit as st

from vas.services.qa import ask_question
from vas.ui.components import (image_picker, image_with_quality, no_image_placeholder, show_confidence,
                               verify_banner)
from vas.ui.context import AppContext
from vas.utils.confidence import estimate_confidence

EXAMPLE_QUESTIONS = [
    "What vehicles are visible?",
    "Is anyone carrying a bag?",
    "What does any visible text or sign say?",
    "What time of day does this appear to be?",
]


def _handle_question(ctx: AppContext, image_id: str, question: str) -> None:
    """Send a question to the model and store both turns against the image."""
    rec = ctx.case.images[image_id]
    history = [{"role": t["role"], "content": t["content"]} for t in rec.get("chat", []) if not t.get("blocked")]
    image = ctx.case.load_image(image_id)
    result = ctx.run_model(
        lambda: ask_question(ctx.backend, image, history, question, ctx.settings.max_pixels_single),
        spinner="Looking at the image…")
    if result is None:
        return
    chat = rec.get("chat", [])
    chat.append({"role": "user", "content": question, "blocked": result["blocked"]})
    confidence = None if result["blocked"] else estimate_confidence(result["answer"], rec["quality"]["score"])
    chat.append({"role": "assistant", "content": result["answer"], "confidence": confidence,
                 "blocked": result["blocked"]})
    ctx.case.update_image(image_id, chat=chat)


def render(ctx: AppContext) -> None:
    image_id = image_picker(ctx, "ask")
    if not image_id:
        no_image_placeholder()
        return
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        image_with_quality(ctx, image_id)

    with right:
        st.markdown("#### Ask about this image")
        verify_banner()
        # Example questions as quick buttons
        cols = st.columns(2)
        clicked = None
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
                clicked = q

        typed = st.chat_input("Ask a question or a follow-up…", key="ask_input")
        question = typed or clicked
        if question:
            _handle_question(ctx, image_id, question)

        chat = ctx.case.images[image_id].get("chat", [])
        box = st.container(height=480, border=True)
        with box:
            if not chat:
                st.caption("No questions yet. Follow-up questions remember the earlier conversation about this image.")
            for turn in chat:
                avatar = ":material/person:" if turn["role"] == "user" else ":material/smart_toy:"
                with st.chat_message(turn["role"], avatar=avatar):
                    st.markdown(turn["content"])
                    if turn["role"] == "assistant":
                        show_confidence(turn.get("confidence"))
        if chat and st.button("Clear conversation", icon=":material/delete_sweep:", key="clear_chat"):
            ctx.case.update_image(image_id, chat=[])
            st.rerun()
