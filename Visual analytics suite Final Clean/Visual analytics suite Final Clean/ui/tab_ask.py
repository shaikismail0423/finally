"""Ask tab: chat about the active image; follow-up questions remember the conversation."""

import streamlit as st

from backend import analysis
from backend.cases import Case
from ui.common import VERIFY_NOTE, image_picker, preview, run_model

EXAMPLE_QUESTIONS = ["What vehicles are visible?", "Is anyone carrying a bag?",
                     "What does any visible text or sign say?", "What time of day does this appear to be?"]


def render(case: Case) -> None:
    image_id = image_picker(case, "ask")
    if not image_id:
        return
    left, right = st.columns([1.1, 1], gap="large")
    left.image(preview(case, image_id), caption=case.images[image_id]["filename"], width="stretch")

    with right:
        st.caption(VERIFY_NOTE)
        columns = st.columns(2)
        clicked = None
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if columns[i % 2].button(example, key=f"example_{i}", width="stretch"):
                clicked = example
        question = st.chat_input("Ask a question or a follow-up…") or clicked

        if question:
            chat_log = case.images[image_id].get("chat", [])
            history = [{"role": turn["role"], "content": turn["content"]} for turn in chat_log]
            image = case.load_image(image_id)
            answer = run_model(lambda: analysis.ask(image, question, history), "Looking at the image…")
            if answer is not None:
                case.update_image(image_id, chat=chat_log + [{"role": "user", "content": question},
                                                             {"role": "assistant", "content": answer}])

        chat_log = case.images[image_id].get("chat", [])
        with st.container(height=480, border=True):
            if not chat_log:
                st.caption("No questions yet. Follow-up questions remember the earlier conversation about this image.")
            for turn in chat_log:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])
        if chat_log and st.button("Clear conversation"):
            case.update_image(image_id, chat=[])
            st.rerun()
