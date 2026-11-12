"""Tags tab (4.3): auto-tag single or many images, edit tags, manage the case focus list."""

import streamlit as st

from config import APP_SETTINGS
from vas.services.tagging import tag_image
from vas.ui.components import image_picker, image_with_quality, no_image_placeholder, tag_chips_html
from vas.ui.context import AppContext


def _run_tagging(ctx: AppContext, image_id: str) -> bool:
    """Tag one image and store the result. Returns True on success."""
    case = ctx.case
    image = case.load_image(image_id)
    result = ctx.run_model(
        lambda: tag_image(ctx.backend, image, case.focus_tags, ctx.settings.max_pixels_single),
        spinner=f"Tagging {case.images[image_id]['filename']}…")
    if result is None:
        return False
    rec = case.images[image_id]
    case.update_image(image_id, tags=result["tags"], model_tags=result["tags"], focus_hits=result["focus_hits"],
                      tag_categories=result["categories"], tags_version=rec.get("tags_version", 0) + 1)
    return True


def _focus_list_editor(ctx: AppContext) -> None:
    case = ctx.case
    with st.expander(f"Case focus list ({len(case.focus_tags)} items)", icon=":material/flag:"):
        st.caption("Items relevant to this case. Tagging checks each one explicitly and flags images where it is visible.")
        options = sorted(set(APP_SETTINGS.default_focus_tags) | set(case.focus_tags)
                         | {"knife", "firearm", "drugs", "syringe", "bank card", "passport", "laptop"})
        new_focus = st.multiselect("Focus items", options, default=case.focus_tags, accept_new_options=True,
                                   key="focus_editor", placeholder="Type to add a new item…")
        if st.button("Save focus list", key="save_focus"):
            case.set_focus_tags([t.strip().lower() for t in new_focus if t.strip()])
            st.success("Focus list saved. Re-tag images to apply it.")


def _bulk_tagging(ctx: AppContext) -> None:
    case = ctx.case
    untagged = [i for i, r in case.images.items() if not r.get("model_tags")]
    st.markdown("#### Bulk tagging")
    c1, c2 = st.columns(2)
    run_untagged = c1.button(f"Tag untagged images ({len(untagged)})", disabled=not untagged, type="primary",
                             icon=":material/auto_awesome:", width="stretch")
    run_all = c2.button(f"Re-tag all ({len(case.images)})", icon=":material/refresh:", width="stretch")
    targets = untagged if run_untagged else list(case.images) if run_all else []
    if targets:
        bar = st.progress(0.0, text="Starting…")
        done = 0
        for n, image_id in enumerate(targets, start=1):
            done += _run_tagging(ctx, image_id)
            bar.progress(n / len(targets), text=f"Tagged {n}/{len(targets)}")
        st.success(f"Tagged {done} of {len(targets)} image(s). Use *Case Gallery* to filter by tag.")


def render(ctx: AppContext) -> None:
    case = ctx.case
    _focus_list_editor(ctx)
    image_id = image_picker(ctx, "tagging")
    if not image_id:
        no_image_placeholder()
        return

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        image_with_quality(ctx, image_id)
    with right:
        rec = case.images[image_id]
        st.markdown("#### Tags for this image")
        if st.button("Generate tags", type="primary", icon=":material/sell:", key="tag_one"):
            _run_tagging(ctx, image_id)
            rec = case.images[image_id]

        hits = rec.get("focus_hits", [])
        for hit in hits:
            st.error(f"**Focus item: {hit['item']}** — {hit.get('evidence') or 'visible in image'}", icon=":material/flag:")

        if rec.get("tag_categories"):
            for category, values in rec["tag_categories"].items():
                if values:
                    st.markdown(f"<span class='vas-muted'>{category.replace('_', ' ').title()}:</span> "
                                + tag_chips_html(values, [h["item"] for h in hits], limit=20), unsafe_allow_html=True)

        st.markdown("##### Review & edit")
        options = sorted(set(case.all_tags()) | set(rec.get("tags", [])))
        edited = st.multiselect("Tags (add, edit or remove)", options, default=rec.get("tags", []),
                                accept_new_options=True, key=f"tags_edit_{image_id}_{rec.get('tags_version', 0)}",
                                placeholder="Type a tag and press Enter…")
        if st.button("Save tags", icon=":material/save:", key="save_tags"):
            clean = list(dict.fromkeys(t.strip().lower() for t in edited if t.strip()))
            case.update_image(image_id, tags=clean)
            st.success("Tags saved.")
        if not rec.get("model_tags") and not rec.get("tags"):
            st.caption("No tags yet — click **Generate tags** or add your own.")

    st.divider()
    _bulk_tagging(ctx)
