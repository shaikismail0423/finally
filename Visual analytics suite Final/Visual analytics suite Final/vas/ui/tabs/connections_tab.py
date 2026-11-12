"""Connect the Dots tab (4.5): multi-image insights and follow-up questions."""

import html
import uuid

import streamlit as st

from config import APP_SETTINGS, multi_image_pixels
from vas.core.case_store import now_iso
from vas.services.multi_image import ask_group_question, find_connections
from vas.services.summarization import summarize_image
from vas.ui.components import thumbnail
from vas.ui.context import AppContext

TYPE_ICONS = {"recurring_object": "🔁", "same_location": "📍", "text_or_logo": "🔤", "timeline": "🕒",
              "difference": "↔️", "other": "💡"}


def _description_for(ctx: AppContext, image_id: str) -> str:
    """Text used in 'text mode': best available summary + tags (generates a caption if needed)."""
    rec = ctx.case.images[image_id]
    summaries = rec.get("summaries", {})
    summary = next((summaries[k]["text"] for k in ("Short paragraph", "Detailed description", "One-line caption")
                    if k in summaries), None)
    if summary is None:
        image = ctx.case.load_image(image_id)
        summary = ctx.run_model(lambda: summarize_image(ctx.backend, image, "One-line caption", "",
                                                        ctx.settings.max_pixels_single),
                                spinner=f"Describing {rec['filename']}…")
        summary = summary or "(no description)"
    tags = ", ".join(rec.get("tags", [])) or "none"
    return f"{summary} Tags: {tags}."


def _image_strip(ctx: AppContext, image_ids: list, per_row: int = 8, size: int = 160) -> None:
    """Numbered thumbnails: Image 1, Image 2, ..."""
    for start in range(0, len(image_ids), per_row):
        cols = st.columns(per_row)
        for col, (n, image_id) in zip(cols, enumerate(image_ids[start:start + per_row], start=start + 1)):
            col.image(thumbnail(ctx.case, image_id, size), caption=f"Image {n}",
                      width="stretch")


def _run_analysis(ctx: AppContext, picks: list, mode: str, extra: str) -> None:
    case = ctx.case
    use_text = mode == "Text (summaries + tags)" or (mode == "Auto" and len(picks) > APP_SETTINGS.max_images_visual_multi)
    descriptions = [_description_for(ctx, i) for i in picks] if use_text else None
    images = [(i, case.load_image(i)) for i in picks]
    result = ctx.run_model(
        lambda: find_connections(ctx.backend, images, multi_image_pixels(ctx.settings, len(images)), extra, descriptions),
        spinner=f"Comparing {len(picks)} images…")
    if result is None:
        return
    case.add_insight_run({"id": uuid.uuid4().hex[:8], "created_at": now_iso(),
                          "image_ids": picks, "mode": result["mode"], "extra": extra,
                          "raw": result["raw"], "followups": [],
                          "result": {"insights": result["insights"], "overall_summary": result["overall_summary"]}})
    st.session_state["insight_run_pick"] = 0


def _show_run(ctx: AppContext, run: dict) -> None:
    case = ctx.case
    st.warning("These are **observations to verify**, not conclusions. Open the supporting images to check each one.",
               icon=":material/fact_check:")
    if run["result"].get("overall_summary"):
        st.markdown(f"**Overall:** {run['result']['overall_summary']}")
    st.caption(f"Mode: {run['mode']} · {len(run['image_ids'])} images · {run['created_at']}")
    if not run["result"]["insights"]:
        st.info("No connections were reported for this group.")
    for ins in run["result"]["insights"]:
        icon = TYPE_ICONS.get(ins["type"], "💡")
        support = [i for i in ins["image_ids"] if i in case.images]
        numbers = ", ".join(f"Image {run['image_ids'].index(i) + 1}" for i in support)
        st.markdown(f"""<div class="vas-insight {ins['confidence']}">
            <b>{icon} {html.escape(ins['type'].replace('_', ' ').title())}</b>
            <span class="vas-badge vas-{ {'high': 'good', 'medium': 'fair', 'low': 'poor'}[ins['confidence']] }">{ins['confidence']}</span><br>
            {html.escape(ins['description'])}<br><span class="vas-muted">Supporting: {numbers or 'not specified'}</span>
            </div>""", unsafe_allow_html=True)
        if support:
            cols = st.columns(min(len(support), 6) + 2)
            for col, image_id in zip(cols, support[:6]):
                col.image(thumbnail(case, image_id, 140), caption=case.images[image_id]["filename"], width="stretch")
    with st.expander("Raw model output"):
        st.code(run.get("raw", ""), language="json")


def _followups(ctx: AppContext, run: dict) -> None:
    case = ctx.case
    st.markdown("##### Ask about this group")
    for f in run.get("followups", []):
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(f["question"])
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            st.markdown(f["answer"])
    with st.form(f"followup_{run['id']}", clear_on_submit=True, border=False):
        question = st.text_input("Question", placeholder="e.g. Which images contain the white van?")
        asked = st.form_submit_button("Ask", icon=":material/send:")
    if asked:
        ids = [i for i in run["image_ids"] if i in case.images]
        if len(ids) > APP_SETTINGS.max_images_visual_multi:
            st.caption(f"Using the first {APP_SETTINGS.max_images_visual_multi} images to fit GPU memory.")
            ids = ids[:APP_SETTINGS.max_images_visual_multi]
        images = [(i, case.load_image(i)) for i in ids]
        answer = ctx.run_model(lambda: ask_group_question(ctx.backend, images, question,
                                                          multi_image_pixels(ctx.settings, len(images))),
                               spinner="Thinking about the group…")
        if answer:
            run.setdefault("followups", []).append({"question": question, "answer": answer, "at": now_iso()})
            case.save()
            st.rerun()


def render(ctx: AppContext) -> None:
    case = ctx.case
    if len(case.images) < 2:
        st.info("Add at least two images to this case to find connections.", icon=":material/hub:")
        return

    st.markdown("#### Select images to compare")
    default = [i for i in st.session_state.get("filtered_image_ids", list(case.images)) if i in case.images][:12]
    picks = st.multiselect("Images (order = Image 1, Image 2, …)", list(case.images), default=default,
                           format_func=case.label, key="multi_pick")
    if picks:
        _image_strip(ctx, picks)
    c1, c2, c3 = st.columns([1.3, 2.5, 1])
    mode = c1.selectbox("Mode", ["Auto", "Visual (images together)", "Text (summaries + tags)"],
                        help=f"Auto sends images directly for up to {APP_SETTINGS.max_images_visual_multi} images, "
                             "otherwise reasons over per-image summaries and tags (scales to 10-20+ images).")
    extra = c2.text_input("Focus (optional)", placeholder="e.g. vehicles and licence plates")
    c3.write("")
    if c3.button("Find connections", type="primary", disabled=len(picks) < 2, icon=":material/hub:", width="stretch"):
        _run_analysis(ctx, picks, mode, extra)

    if case.insights:
        runs = case.insights
        idx = st.selectbox("Insight run", range(len(runs)), key="insight_run_pick",
                           format_func=lambda i: f"{runs[i]['created_at'].replace('T', ' ')} · {len(runs[i]['image_ids'])} images · {len(runs[i]['result']['insights'])} insights")
        run = runs[min(idx or 0, len(runs) - 1)]
        _show_run(ctx, run)
        _followups(ctx, run)
    else:
        st.caption("No insight runs yet.")
