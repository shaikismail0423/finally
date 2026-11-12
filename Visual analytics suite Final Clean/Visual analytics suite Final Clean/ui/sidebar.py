"""Sidebar: open or create a case, load the sample case, add images, and show the model server status."""

import streamlit as st

import config
from backend import cases
from backend.model_client import check_connection

SAMPLE_CASE_NAME = "Sample case (synthetic demo data)"


@st.cache_data(ttl=30, show_spinner=False)
def _server_status() -> tuple:
    """Cached for 30 s so the server is not checked on every click."""
    return check_connection()


def _load_sample_case() -> str:
    """Create (or reuse) the sample case, add every image in sample_data/ and return its id."""
    existing = [case for case in cases.list_cases() if case.name == SAMPLE_CASE_NAME]
    case = existing[0] if existing else cases.create_case(SAMPLE_CASE_NAME, "Synthetic receipts, notes and scenes.")
    for path in sorted(config.SAMPLE_DIR.iterdir()):
        if path.suffix.lower().lstrip(".") in config.IMAGE_TYPES:
            case.add_image(path.read_bytes(), path.name)
    return case.case_id


def _add_images_form(case: cases.Case) -> None:
    # The uploader's key changes after every upload, which empties it for the next batch
    uploads = st.file_uploader("Add images to this case", type=config.IMAGE_TYPES, accept_multiple_files=True,
                               key=f"upload_{case.case_id}_{st.session_state.get('upload_round', 0)}")
    if uploads and st.button("Add to case", type="primary", width="stretch"):
        messages = []
        for upload in uploads:
            try:
                case.add_image(upload.getvalue(), upload.name)
            except ValueError as exc:
                messages.append(str(exc))
        st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
        st.session_state["upload_errors"] = messages
        st.rerun()
    for message in st.session_state.pop("upload_errors", []):
        st.error(message)


def render_sidebar():
    """Draw the sidebar and return the open case (None if there are no cases yet)."""
    with st.sidebar:
        st.markdown("### Case")
        names = {case.case_id: case.name for case in cases.list_cases()}
        # A widget's value can't be changed after it is drawn, so the buttons below store the case
        # to open in "pending_case_id", and it is applied here at the start of the next run.
        if "pending_case_id" in st.session_state:
            st.session_state["case_id"] = st.session_state.pop("pending_case_id")
        if st.session_state.get("case_id") not in names:
            st.session_state["case_id"] = next(iter(names), None)
        if names:
            st.selectbox("Open case", list(names), format_func=names.get, key="case_id")

        with st.popover("New case", width="stretch"):
            with st.form("new_case", clear_on_submit=True, border=False):
                name = st.text_input("Case name", placeholder="e.g. Op. Harbour – 2026/114")
                description = st.text_area("Description (optional)", height=70)
                if st.form_submit_button("Create case", type="primary"):
                    try:
                        st.session_state["pending_case_id"] = cases.create_case(name, description).case_id
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        if st.button("Load sample case", width="stretch"):
            st.session_state["pending_case_id"] = _load_sample_case()
            st.rerun()

        case = cases.open_case(st.session_state["case_id"]) if st.session_state.get("case_id") else None
        if case:
            missing = case.missing_images()
            if missing:  # files deleted outside the app would break every tab, so deal with them first
                st.error("Image file(s) missing from the case folder: "
                         + ", ".join(case.images[image_id]["filename"] for image_id in missing)
                         + f". Put them back in `{case.folder / 'images'}` and reload, or remove them from the case.")
                if st.button("Remove missing images from case", type="primary", width="stretch"):
                    for image_id in missing:
                        case.delete_image(image_id)
                    st.rerun()
                st.stop()
            _add_images_form(case)

        st.divider()
        ok, message = _server_status()
        if ok:
            st.success(message)
        else:
            st.error(message)
            st.caption("Check VAS_API_BASE_URL, VAS_API_MODEL and VAS_API_KEY in .env, then restart the app.")
    return case
