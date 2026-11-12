"""Tags tab: tag one or all images, flag case focus items, edit tags by hand, manage the focus list."""

import streamlit as st

import config
from backend import analysis
from backend.cases import Case
from ui.common import image_picker, preview, run_model

EXTRA_FOCUS_OPTIONS = ["knife", "firearm", "drugs", "syringe", "bank card", "passport", "laptop"]


def _tag_image(case: Case, image_id: str) -> bool:
    """Tag one image and save the result. Returns True if it worked."""
    image = case.load_image(image_id)
    result = run_model(lambda: analysis.tag(image, case.focus_tags), f"Tagging {case.images[image_id]['filename']}…")
    if result is None:
        return False
    # tags_version is part of the tag editor's key, so the editor shows the new tags
    version = case.images[image_id].get("tags_version", 0) + 1
    case.update_image(image_id, tags=result["tags"], tag_categories=result["categories"],
                      focus_hits=result["focus_hits"], tags_version=version)
    return True


def _focus_list(case: Case) -> None:
    with st.expander(f"Case focus list ({len(case.focus_tags)} items)"):
        st.caption("Tagging checks each item explicitly and flags images where it is clearly visible.")
        options = sorted(set(config.DEFAULT_FOCUS_TAGS) | set(case.focus_tags) | set(EXTRA_FOCUS_OPTIONS))
        chosen = st.multiselect("Focus items", options, default=case.focus_tags, accept_new_options=True,
                                placeholder="Type to add a new item…")
        if st.button("Save focus list"):
            case.data["focus_tags"] = [item.strip().lower() for item in chosen if item.strip()]
            case.save()
            st.success("Focus list saved. Re-tag images to apply it.")


def _bulk_tagging(case: Case) -> None:
    untagged = [image_id for image_id, record in case.images.items() if "tag_categories" not in record]
    c1, c2 = st.columns(2)
    if c1.button(f"Tag untagged images ({len(untagged)})", type="primary", disabled=not untagged, width="stretch"):
        targets = untagged
    elif c2.button(f"Re-tag all ({len(case.images)})", width="stretch"):
        targets = list(case.images)
    else:
        return
    progress = st.progress(0.0)
    done = 0
    for number, image_id in enumerate(targets, start=1):
        done += _tag_image(case, image_id)
        progress.progress(number / len(targets), text=f"Tagged {number}/{len(targets)}")
    st.success(f"Tagged {done} of {len(targets)} image(s). Filter by tag in the Case Gallery.")


def render(case: Case) -> None:
    _focus_list(case)
    image_id = image_picker(case, "tags")
    if not image_id:
        return
    left, right = st.columns([1, 1.2], gap="large")
    left.image(preview(case, image_id), caption=case.images[image_id]["filename"], width="stretch")

    with right:
        if st.button("Generate tags", type="primary"):
            _tag_image(case, image_id)
        record = case.images[image_id]
        for hit in record.get("focus_hits", []):
            st.error(f"**Focus item: {hit['item']}** — {hit.get('evidence') or 'visible in image'}")
        for category, values in record.get("tag_categories", {}).items():
            st.markdown(f"**{category.replace('_', ' ').capitalize()}:** {', '.join(values)}")

        options = sorted({tag for r in case.images.values() for tag in r.get("tags", [])})
        edited = st.multiselect("Tags (add, edit or remove)", options, default=record.get("tags", []),
                                accept_new_options=True, placeholder="Type a tag and press Enter…",
                                key=f"tags_{image_id}_{record.get('tags_version', 0)}")
        if st.button("Save tags"):
            case.update_image(image_id, tags=list(dict.fromkeys(t.strip().lower() for t in edited if t.strip())))
            st.success("Tags saved.")

    st.divider()
    st.markdown("#### Tag many images")
    _bulk_tagging(case)
