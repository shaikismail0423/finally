"""Show Me Where tab: find objects, draw labelled boxes, download the marked-up image."""

import json

import pandas as pd
import streamlit as st

from backend import analysis
from backend.cases import Case, now
from ui.common import VERIFY_NOTE, draw_boxes, image_picker, png_bytes, preview, run_model

QUICK_TARGETS = ["vehicles", "persons", "bags or backpacks", "license plates", "text or signs", "mobile phones"]


def render(case: Case) -> None:
    image_id = image_picker(case, "locate")
    if not image_id:
        return
    record = case.images[image_id]
    left, right = st.columns([1.4, 1], gap="large")

    with right:
        st.caption(VERIFY_NOTE)
        quick = st.pills("Quick requests", QUICK_TARGETS, key="locate_quick")
        with st.form("locate_form", border=False):
            target = st.text_input("What should I find?", value=quick or "",
                                   placeholder="e.g. red backpack, every person wearing a hat")
            submitted = st.form_submit_button("Highlight", type="primary")
        if submitted:
            image = case.load_image(image_id)
            result = run_model(lambda: analysis.locate(image, target), "Locating…")
            if result is not None:
                result["created_at"] = now()
                case.update_image(image_id, groundings=[result] + record.get("groundings", [])[:19])
                st.session_state[f"locate_result_{image_id}"] = 0  # show the new result

        results = record.get("groundings", [])
        selected = None
        if results:
            index = st.selectbox("Result", range(len(results)), key=f"locate_result_{image_id}",
                                 format_func=lambda i: f"“{results[i]['target']}” · {len(results[i]['detections'])} "
                                                       f"found · {results[i]['created_at'][11:16]}")
            selected = results[index]
            detections = selected["detections"]
            st.metric("Items found", len(detections))
            if detections:
                st.dataframe(pd.DataFrame([{"#": n, "label": d["label"], "x1": d["box"][0], "y1": d["box"][1],
                                            "x2": d["box"][2], "y2": d["box"][3]}
                                           for n, d in enumerate(detections, start=1)]),
                             hide_index=True, width="stretch")
            else:
                st.info("Nothing matching was found in this image.")
            with st.expander("Raw model output"):
                st.code(selected.get("raw", ""), language="json")

    with left:
        if selected and selected["detections"]:
            marked = draw_boxes(case.load_image(image_id), selected["detections"])
            st.image(marked, caption=f"{record['filename']} — “{selected['target']}”", width="stretch")
            name = f"{image_id}_{selected['target'][:20].replace(' ', '_')}"
            d1, d2 = st.columns(2)
            d1.download_button("Marked-up image (PNG)", png_bytes(marked), f"{name}.png", "image/png", width="stretch")
            d2.download_button("Boxes (JSON)", json.dumps({"image": record["filename"], "target": selected["target"],
                                                           "detections": selected["detections"]}, indent=2),
                               f"{name}.json", "application/json", width="stretch")
        else:
            st.image(preview(case, image_id), caption=record["filename"], width="stretch")
