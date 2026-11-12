"""Case Gallery tab: all images with their tags, search and filter, downloads, and removing images."""

import pandas as pd
import streamlit as st

from backend.cases import Case, now
from ui.common import csv_bytes, preview

GRID_COLUMNS = 4


def _matches(record: dict, tags: list, text: str, flagged_only: bool) -> bool:
    """Does this image pass the filters? (any selected tag, free text, focus-item flag)"""
    if tags and not set(tags) & set(record.get("tags", [])):
        return False
    if flagged_only and not record.get("focus_hits"):
        return False
    if text:
        searchable = [record["filename"], *record.get("tags", []),
                      *(summary["text"] for summary in record.get("summaries", {}).values()),
                      (record.get("ocr") or {}).get("markdown", "")]
        return text.lower() in " ".join(searchable).lower()
    return True


def _report(case: Case) -> str:
    """Markdown report of everything produced for the case."""
    lines = [f"# Case report — {case.name}", "", f"*Generated {now()} by the Visual Analytics Suite.*", "",
             "> Decision support only: all outputs are machine-generated observations to be verified by an analyst.",
             "", f"**Focus list:** {', '.join(case.focus_tags) or '—'}", ""]
    for image_id, record in case.images.items():
        lines += [f"## {record['filename']} (`{image_id}`)", ""]
        if record.get("tags"):
            lines.append(f"- **Tags:** {', '.join(record['tags'])}")
        for hit in record.get("focus_hits", []):
            lines.append(f"- **Focus item flagged:** {hit['item']} ({hit['evidence']})")
        for length, summary in record.get("summaries", {}).items():
            lines.append(f"- **Summary ({length}):** {summary['text']}")
        for result in record.get("groundings", []):
            lines.append(f"- **Highlighted '{result['target']}':** {len(result['detections'])} found")
        for turn in record.get("chat", []):
            lines.append(f"- *{'Analyst' if turn['role'] == 'user' else 'Model'}:* {turn['content']}")
        if record.get("ocr"):
            lines += ["", "**Document text**", "", record["ocr"]["markdown"]]
        lines.append("")
    if case.insights:
        lines += ["## Connect the Dots (observations to verify)", ""]
        for run in case.insights:
            for insight in run["result"]["insights"]:
                files = ", ".join(case.images[i]["filename"] for i in insight["image_ids"] if i in case.images)
                lines.append(f"- [{insight['confidence']}] {insight['description']} *(images: {files or '—'})*")
        lines.append("")
    return "\n".join(lines)


def _open(image_id: str) -> None:
    st.session_state["active_image"] = image_id


def render(case: Case) -> None:
    if not case.images:
        st.info("This case has no images yet. Add some in the sidebar, or click **Load sample case**.")
        return

    all_tags = sorted({tag for record in case.images.values() for tag in record.get("tags", [])})
    c1, c2, c3 = st.columns([3, 3, 1], vertical_alignment="bottom")
    tags = c1.multiselect("Filter by tag", all_tags, placeholder="Any tag")
    text = c2.text_input("Search", placeholder="Filename, tag, summary or document text").strip()
    flagged_only = c3.toggle("Flagged only", help="Only images where a focus item was found")
    visible = [image_id for image_id, record in case.images.items() if _matches(record, tags, text, flagged_only)]
    st.caption(f"Showing {len(visible)} of {len(case.images)} images. **Open** makes an image the active one in the other tabs.")

    for start in range(0, len(visible), GRID_COLUMNS):
        for column, image_id in zip(st.columns(GRID_COLUMNS), visible[start:start + GRID_COLUMNS]):
            record = case.images[image_id]
            with column.container(border=True):
                st.image(preview(case, image_id, 400), caption=record["filename"], width="stretch")
                hits = [hit["item"] for hit in record.get("focus_hits", [])]
                if hits:
                    st.markdown(f":red[**Flagged:** {', '.join(hits)}]")
                st.caption(", ".join(record.get("tags", [])[:8]) or "No tags yet")
                st.button("Open", key=f"open_{image_id}", on_click=_open, args=(image_id,), width="stretch")

    st.divider()
    tags_table = pd.DataFrame([{"image_id": image_id, "filename": case.images[image_id]["filename"],
                                "tags": "; ".join(case.images[image_id].get("tags", [])),
                                "focus_hits": "; ".join(h["item"] for h in case.images[image_id].get("focus_hits", []))}
                               for image_id in visible])
    d1, d2, d3 = st.columns(3)
    d1.download_button("Tags (CSV)", csv_bytes(tags_table), f"{case.case_id}_tags.csv", "text/csv", width="stretch")
    d2.download_button("Case report (Markdown)", _report(case).encode("utf-8"), f"{case.case_id}_report.md",
                       "text/markdown", width="stretch")
    with d3.popover("Remove an image", width="stretch"):
        image_id = st.selectbox("Image", list(case.images), format_func=case.label, key="remove_pick")
        if st.button("Remove permanently", type="primary"):
            case.delete_image(image_id)
            st.rerun()
