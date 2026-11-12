"""Show Me Where tab (4.2): highlight objects with labelled boxes; export the marked-up image."""

import json

import pandas as pd
import streamlit as st

from vas.core.case_store import now_iso
from vas.services.grounding import detections_from_record, ground_objects
from vas.ui.components import image_picker, image_with_quality, no_image_placeholder, verify_banner
from vas.ui.context import AppContext
from vas.utils.image_utils import image_to_bytes
from vas.visualization.drawing import draw_boxes

QUICK_TARGETS = ["vehicles", "persons", "bags or backpacks", "license plates", "text or signs", "mobile phones"]


def render(ctx: AppContext) -> None:
    image_id = image_picker(ctx, "grounding")
    if not image_id:
        no_image_placeholder()
        return
    case = ctx.case
    rec = case.images[image_id]
    image = case.load_image(image_id)

    left, right = st.columns([1.4, 1], gap="large")
    with right:
        st.markdown("#### Find and highlight")
        verify_banner()
        quick = st.pills("Quick requests", QUICK_TARGETS, key="ground_quick")
        with st.form("ground_form", border=False):
            target = st.text_input("What should I find?", value=quick or "",
                                   placeholder="e.g. red backpack, every person wearing a hat")
            submitted = st.form_submit_button("Highlight", type="primary", icon=":material/center_focus_strong:")
        if submitted:
            result = ctx.run_model(
                lambda: ground_objects(ctx.backend, image, target, ctx.settings.max_pixels_single, ctx.settings.bbox_mode),
                spinner="Locating…")
            if result is not None:
                if result["blocked"]:
                    st.warning(result["message"])
                else:
                    groundings = rec.get("groundings", [])
                    groundings.insert(0, {"target": result["target"], "created_at": now_iso(), "raw": result["raw"],
                                          "detections": [d.to_dict() for d in result["detections"]]})
                    case.update_image(image_id, groundings=groundings[:20])
                    st.session_state["ground_pick"] = 0

        groundings = rec.get("groundings", [])
        selected = None
        if groundings:
            idx = st.selectbox("Result", range(len(groundings)), key="ground_pick",
                               format_func=lambda i: f"“{groundings[i]['target']}” · {len(groundings[i]['detections'])} found · {groundings[i]['created_at'][11:16]}")
            selected = groundings[min(idx or 0, len(groundings) - 1)]
            dets = detections_from_record(selected)
            st.metric("Items found", len(dets))
            if dets:
                table = pd.DataFrame([{"#": i + 1, "label": d.label, "x1": d.box[0], "y1": d.box[1],
                                       "x2": d.box[2], "y2": d.box[3]} for i, d in enumerate(dets)])
                st.dataframe(table, hide_index=True, width="stretch")
            else:
                st.info("Nothing matching was found in this image.")
            with st.expander("Raw model output"):
                st.code(selected.get("raw", ""), language="json")

    with left:
        if selected and selected["detections"]:
            dets = detections_from_record(selected)
            marked = draw_boxes(image, [d.box for d in dets], [d.label for d in dets])
            image_with_quality(ctx, image_id, marked, caption=f"{rec['filename']} — “{selected['target']}”")
            d1, d2 = st.columns(2)
            d1.download_button("Marked-up image (PNG)", image_to_bytes(marked),
                               file_name=f"{image_id}_{selected['target'][:20].replace(' ', '_')}.png",
                               mime="image/png", icon=":material/download:", width="stretch")
            d2.download_button("Boxes (JSON)", json.dumps({"image": rec["filename"], "image_id": image_id,
                                                           "target": selected["target"],
                                                           "detections": selected["detections"]}, indent=2),
                               file_name=f"{image_id}_boxes.json", mime="application/json",
                               icon=":material/data_object:", width="stretch")
        else:
            image_with_quality(ctx, image_id, image)
