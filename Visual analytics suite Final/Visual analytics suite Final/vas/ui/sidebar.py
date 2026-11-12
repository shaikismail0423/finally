"""
Sidebar: case workspace, image upload, active-image picker
and model settings. Returns a ready-to-use AppContext.
"""

from dataclasses import replace

import streamlit as st

from config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_MB, SAMPLE_DATA_DIR, ModelSettings
from vas.core.case_store import CaseStore
from vas.core.vlm import create_backend
from vas.ui.components import thumbnail
from vas.ui.context import AppContext
from vas.utils.image_utils import ImageValidationError

SAMPLE_CASE_NAME = "Sample case (synthetic demo data)"
RESOLUTION_PRESETS = {"Fast (~0.4 MP)": 512 * 28 * 28, "Balanced (~0.8 MP)": 1024 * 28 * 28,
                      "High detail (~1.6 MP)": 2048 * 28 * 28}


@st.cache_resource(show_spinner=False)
def get_store() -> CaseStore:
    return CaseStore()


def load_sample_case(store: CaseStore) -> str:
    """Create (or reuse) the sample case and add every image in sample_data/. Returns case id."""
    existing = [c for c in store.list_cases() if c.name == SAMPLE_CASE_NAME]
    case = existing[0] if existing else store.create_case(
        SAMPLE_CASE_NAME, "Synthetic receipts, notes and scenes generated for demonstration.")
    for path in sorted(SAMPLE_DATA_DIR.glob("*")):
        if path.suffix.lower().lstrip(".") in ALLOWED_IMAGE_TYPES:
            is_doc = any(k in path.stem for k in ("receipt", "invoice", "note", "label", "form"))
            case.add_image(path.read_bytes(), path.name, is_document=is_doc)
    return case.case_id


def _model_settings_panel() -> ModelSettings:
    """Model server and image-detail controls; values persist in session_state."""
    base: ModelSettings = st.session_state.setdefault("model_settings", ModelSettings())
    with st.expander("Model settings", icon=":material/tune:"):
        settings = replace(base)
        settings.api_base_url = st.text_input("Server URL", base.api_base_url,
                                              help="Office model server, ending in /v1 (VAS_API_BASE_URL)")
        settings.api_model_name = st.text_input("Model name", base.api_model_name,
                                                help="Exact model name on the server (VAS_API_MODEL)")
        preset_names = list(RESOLUTION_PRESETS)
        current = next((n for n, v in RESOLUTION_PRESETS.items() if v == base.max_pixels_single), preset_names[1])
        preset = st.selectbox("Image detail sent to model", preset_names, index=preset_names.index(current),
                              help="Higher detail = better small text/objects, but slower and more GPU memory.")
        settings.max_pixels_single = RESOLUTION_PRESETS[preset]
        settings.temperature = st.slider("Temperature", 0.0, 1.0, float(base.temperature), 0.05,
                                         help="Low values give more consistent, factual answers.")
        settings.bbox_mode = st.selectbox("Box coordinates", ["absolute", "relative"],
                                          index=["absolute", "relative"].index(base.bbox_mode),
                                          help="Qwen2.5-VL uses absolute pixels. Use 'relative' (0-1000) for Qwen2-VL.")
    st.session_state["model_settings"] = settings
    return settings


@st.cache_data(ttl=20, show_spinner=False)
def _health(cache_key: tuple, _backend) -> tuple:
    """Cached health check so we don't ping the server on every rerun."""
    return _backend.health_check()


def render_sidebar() -> tuple:
    """Draw the sidebar and return (AppContext, backend_ok, backend_message)."""
    store = get_store()
    with st.sidebar:
        # ---------------- Case workspace ----------------
        st.markdown("### :material/folder_open: Case workspace")
        cases = store.list_cases()
        case_ids = [c.case_id for c in cases]
        names = {c.case_id: c.name for c in cases}
        # Widgets can't be changed after they render, so buttons below store a
        # "pending" case id that is applied here on the next rerun.
        if "pending_case_id" in st.session_state:
            st.session_state["case_id"] = st.session_state.pop("pending_case_id")
        if st.session_state.get("case_id") not in case_ids:
            st.session_state["case_id"] = case_ids[0] if case_ids else None
        if case_ids:
            st.selectbox("Open case", case_ids, format_func=lambda cid: names[cid], key="case_id")
        with st.popover("New case", icon=":material/create_new_folder:", width="stretch"):
            with st.form("new_case_form", clear_on_submit=True, border=False):
                new_name = st.text_input("Case name", placeholder="e.g. Op. Harbour – 2026/114")
                new_desc = st.text_area("Description (optional)", height=70)
                if st.form_submit_button("Create case", type="primary"):
                    try:
                        st.session_state["pending_case_id"] = store.create_case(new_name, new_desc).case_id
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        if st.button("Load sample case", icon=":material/dataset:", width="stretch",
                     help="Creates a case with synthetic receipts, a note and scenes for a quick demo."):
            st.session_state["pending_case_id"] = load_sample_case(store)
            st.rerun()

        case = store.open_case(st.session_state["case_id"]) if st.session_state.get("case_id") else None

        # Image files deleted outside the app would crash every tab, so deal with them first
        missing = case.missing_images() if case else []
        if missing:
            st.error("Image file(s) missing from this case's folder: "
                     + ", ".join(case.images[i]["filename"] for i in missing)
                     + f". Put the file(s) back in `{case.images_dir}` and reload, or remove them from the case.",
                     icon=":material/broken_image:")
            if st.button("Remove missing images from case", type="primary", width="stretch"):
                for image_id in missing:
                    case.delete_image(image_id)
                st.rerun()
            st.stop()

        # ---------------- Upload & active image ----------------
        active_image_id = None
        if case:
            uploads = st.file_uploader(f"Add images to this case (max {MAX_UPLOAD_MB} MB each)",
                                       type=ALLOWED_IMAGE_TYPES, accept_multiple_files=True,
                                       key=f"uploader_{case.case_id}_{st.session_state.get('upload_round', 0)}")
            as_docs = st.checkbox("These are documents (enables skew check)", key="upload_docs")
            if uploads and st.button("Add to case", type="primary", icon=":material/upload:", width="stretch"):
                added, errors = 0, []
                for f in uploads:
                    try:
                        case.add_image(f.getvalue(), f.name, is_document=as_docs)
                        added += 1
                    except ImageValidationError as exc:
                        errors.append(f"{f.name}: {exc}")
                st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
                st.session_state["upload_msg"] = (added, errors)
                st.rerun()
            if "upload_msg" in st.session_state:
                added, errors = st.session_state.pop("upload_msg")
                if added:
                    st.success(f"Added {added} image(s).")
                for err in errors:
                    st.error(err)

            image_ids = list(case.images)
            if image_ids:
                st.markdown("### :material/image: Active image")
                if st.session_state.get("active_image") not in image_ids:
                    st.session_state["active_image"] = image_ids[0]
                st.selectbox("Used by Ask, Show me where, Tags, Summary and Documents", image_ids,
                             format_func=case.label, key="active_image")
                active_image_id = st.session_state["active_image"]
                st.image(thumbnail(case, active_image_id, 360, letterbox=False),
                         width="stretch")

        st.divider()
        settings = _model_settings_panel()
        backend = create_backend(settings)
        cache_key = (settings.api_base_url, settings.api_model_name)
        ok, msg = _health(cache_key, backend)
        if not ok:
            st.error(msg, icon=":material/link_off:")
            st.caption("Check VAS_API_BASE_URL, VAS_API_MODEL and VAS_API_KEY in .env (see README → *Troubleshooting*).")
        st.caption("Images go only to your organisation's model server. No facial recognition.")

    ctx = AppContext(store=store, settings=settings, backend=backend, case=case, active_image_id=active_image_id)
    return ctx, ok, (msg if ok else "Model offline")
