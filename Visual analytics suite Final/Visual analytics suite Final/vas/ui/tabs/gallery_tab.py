"""Case Gallery tab: overview KPIs, search/filter by tag, thumbnails, analytics and exports."""

import pandas as pd
import streamlit as st

from vas.ui.components import quality_badge_html, tag_chips_html, thumbnail
from vas.ui.context import AppContext
from vas.utils.export import case_report_markdown, dataframe_to_csv_bytes, tags_table
from vas.visualization.charts import quality_chart, tag_frequency_chart

GRID_COLUMNS = 4


def _open_image(image_id: str) -> None:
    """Button callback: make this the active image (callbacks may change widget state)."""
    st.session_state["active_image"] = image_id


def render(ctx: AppContext) -> None:
    case = ctx.case
    records = case.images

    # ---------------- KPIs ----------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Images", len(records))
    k2.metric("Tagged", sum(bool(r.get("tags")) for r in records.values()))
    k3.metric("Focus-item flags", sum(bool(r.get("focus_hits")) for r in records.values()),
              help="Images where a case focus item (e.g. weapon, currency) was detected.")
    k4.metric("Documents read", sum(bool(r.get("ocr")) for r in records.values()))
    k5.metric("Insight runs", len(case.insights))

    if not records:
        st.info("This case has no images yet. Use **Add images** in the sidebar, or **Load sample case**.",
                icon=":material/add_photo_alternate:")
        return

    # ---------------- Search & filter ----------------
    st.markdown("#### Search & filter")
    f1, f2, f3, f4 = st.columns([3, 1.2, 2.5, 1.3])
    selected_tags = f1.multiselect("Tags", case.all_tags(), placeholder="Filter by tag…")
    match_all = f2.radio("Match", ["any", "all"], horizontal=True) == "all"
    text = f3.text_input("Search text", placeholder="Filename, tag, summary or document text…")
    flagged_only = f4.toggle("Flagged only", help="Only images with focus-item hits")
    visible = case.filter_images(selected_tags, match_all, text, flagged_only)
    st.session_state["filtered_image_ids"] = visible
    st.caption(f"Showing {len(visible)} of {len(records)} images · filtered set is pre-selected in *Connect the dots*.")

    # ---------------- Thumbnail grid ----------------
    for row_start in range(0, len(visible), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, image_id in zip(cols, visible[row_start:row_start + GRID_COLUMNS]):
            rec = records[image_id]
            with col.container(border=True):
                st.image(thumbnail(case, image_id), width="stretch")
                st.markdown(f'<div class="vas-card-title" title="{rec["filename"]}">{rec["filename"]}</div>',
                            unsafe_allow_html=True)
                hits = [h["item"] for h in rec.get("focus_hits", [])]
                st.markdown(quality_badge_html(rec.get("quality", {}))
                            + (f'<span class="vas-badge vas-poor">⚑ {", ".join(hits)}</span>' if hits else ""),
                            unsafe_allow_html=True)
                st.markdown(tag_chips_html(rec.get("tags", []), hits, limit=6) or
                            '<span class="vas-muted">No tags yet</span>', unsafe_allow_html=True)
                caption = rec.get("summaries", {}).get("One-line caption", {}).get("text")
                if caption:
                    st.caption(caption)
                is_active = image_id == ctx.active_image_id
                st.button("Active" if is_active else "Open", key=f"open_{image_id}", on_click=_open_image,
                          args=(image_id,), disabled=is_active, width="stretch",
                          icon=":material/check_circle:" if is_active else ":material/open_in_new:")

    # ---------------- Analytics ----------------
    with st.expander("Case analytics", icon=":material/insights:", expanded=False):
        c1, c2 = st.columns(2)
        tags_by_image = {r["filename"]: r.get("tags", []) for r in records.values() if r.get("tags")}
        with c1:
            if tags_by_image:
                st.plotly_chart(tag_frequency_chart(tags_by_image), key="gallery_tag_freq")
            else:
                st.caption("Tag images (Tags tab) to see tag statistics.")
        with c2:
            qrows = pd.DataFrame([{"image": r["filename"], "score": r["quality"]["score"], "level": r["quality"]["level"]}
                                  for r in records.values()])
            st.plotly_chart(quality_chart(qrows), key="gallery_quality")

    # ---------------- Exports & management ----------------
    e1, e2, e3 = st.columns([1, 1, 1.4])
    e1.download_button("Tags (CSV)", dataframe_to_csv_bytes(tags_table(case, visible)),
                       file_name=f"{case.case_id}_tags.csv", mime="text/csv", icon=":material/table:", width="stretch")
    e2.download_button("Case report (Markdown)", case_report_markdown(case).encode("utf-8"),
                       file_name=f"{case.case_id}_report.md", mime="text/markdown", icon=":material/description:",
                       width="stretch")
    with e3.popover("Remove an image", icon=":material/delete:", width="stretch"):
        to_remove = st.selectbox("Image", list(records), format_func=case.label, key="remove_pick")
        if st.button("Remove permanently", type="primary", key="remove_btn"):
            case.delete_image(to_remove)
            st.rerun()
