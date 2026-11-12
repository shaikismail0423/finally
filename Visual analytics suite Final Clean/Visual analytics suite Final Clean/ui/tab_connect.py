"""Connect the Dots tab: compare several images, list what links them, and ask follow-up questions."""

import uuid

import streamlit as st

import config
from backend import analysis
from backend.cases import Case, now
from ui.common import preview, run_model

CONFIDENCE_COLORS = {"high": "green", "medium": "orange", "low": "red"}


def _thumbnails(case: Case, image_ids: list, per_row: int = 8) -> None:
    for start in range(0, len(image_ids), per_row):
        for column, (number, image_id) in zip(st.columns(per_row),
                                               enumerate(image_ids[start:start + per_row], start=start + 1)):
            column.image(preview(case, image_id, 200), caption=f"Image {number}", width="stretch")


def _show_run(case: Case, run: dict) -> None:
    st.warning("These are **observations to verify**, not conclusions. Check each one against its images.")
    if run["result"].get("overall_summary"):
        st.markdown(f"**Overall:** {run['result']['overall_summary']}")
    if not run["result"]["insights"]:
        st.info("No connections were reported for this group.")
    for insight in run["result"]["insights"]:
        supporting = [i for i in insight["image_ids"] if i in case.images]
        with st.container(border=True):
            color = CONFIDENCE_COLORS.get(insight["confidence"], "gray")
            st.markdown(f"**{insight['type'].replace('_', ' ').capitalize()}** · :{color}[{insight['confidence']}]")
            st.write(insight["description"])
            for column, image_id in zip(st.columns(6), supporting[:6]):
                column.image(preview(case, image_id, 200), caption=case.images[image_id]["filename"], width="stretch")
    with st.expander("Raw model output"):
        st.code(run.get("raw", ""), language="json")


def _followups(case: Case, run: dict) -> None:
    st.markdown("##### Ask about this group")
    for followup in run.get("followups", []):
        st.chat_message("user").markdown(followup["question"])
        st.chat_message("assistant").markdown(followup["answer"])
    with st.form(f"followup_{run['id']}", clear_on_submit=True, border=False):
        question = st.text_input("Question", placeholder="e.g. Which images contain the white van?")
        asked = st.form_submit_button("Ask")
    if asked:
        images = [case.load_image(i) for i in run["image_ids"] if i in case.images][:config.MAX_IMAGES]
        answer = run_model(lambda: analysis.ask_group(images, question), "Thinking about the group…")
        if answer is not None:
            run.setdefault("followups", []).append({"question": question, "answer": answer, "at": now()})
            case.save()
            st.rerun()


def render(case: Case) -> None:
    if len(case.images) < 2:
        st.info("Add at least two images to this case to find connections.")
        return
    picks = st.multiselect(f"Images to compare (2 to {config.MAX_IMAGES}; the order gives Image 1, Image 2, …)",
                           list(case.images), format_func=case.label, max_selections=config.MAX_IMAGES)
    if picks:
        _thumbnails(case, picks)
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    focus = c1.text_input("Focus (optional)", placeholder="e.g. vehicles and licence plates", key="connect_focus")
    if c2.button("Find connections", type="primary", disabled=len(picks) < 2, width="stretch"):
        images = {image_id: case.load_image(image_id) for image_id in picks}
        result = run_model(lambda: analysis.connect(images, focus), f"Comparing {len(picks)} images…")
        if result is not None:
            case.insights.insert(0, {"id": uuid.uuid4().hex[:8], "created_at": now(), "image_ids": picks,
                                     "focus": focus.strip(), "raw": result["raw"], "followups": [],
                                     "result": {"insights": result["insights"],
                                                "overall_summary": result["overall_summary"]}})
            case.save()
            st.session_state[f"connect_run_{case.case_id}"] = 0  # show the new run

    runs = case.insights
    if not runs:
        st.caption("No comparisons yet.")
        return
    index = st.selectbox("Comparison", range(len(runs)), key=f"connect_run_{case.case_id}",
                         format_func=lambda i: f"{runs[i]['created_at'].replace('T', ' ')} · "
                                               f"{len(runs[i]['image_ids'])} images · "
                                               f"{len(runs[i]['result']['insights'])} insights")
    _show_run(case, runs[index])
    _followups(case, runs[index])
