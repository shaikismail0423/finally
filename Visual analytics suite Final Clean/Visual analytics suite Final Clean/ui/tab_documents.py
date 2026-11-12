"""Documents & OCR tab: read a document image as structured text, tables and key details."""

import pandas as pd
import streamlit as st

from backend import analysis
from backend.cases import Case, now
from backend.parsing import markdown_tables
from ui.common import VERIFY_NOTE, csv_bytes, image_picker, preview, run_model


def _show(ocr: dict, stem: str) -> None:
    if ocr.get("quality_issues"):
        st.warning("**Double-check this document:** " + " · ".join(ocr["quality_issues"]))
    st.caption(f"Document type: **{ocr.get('document_type', 'unknown')}** · read {ocr['created_at'].replace('T', ' ')}")

    structured, plain, tables_tab, details = st.tabs(["Structured text", "Plain text", "Tables", "Key details"])
    with structured:
        with st.container(height=460, border=True):
            st.markdown(ocr["markdown"])
    with plain:
        st.code(ocr["markdown"], language="markdown", wrap_lines=True, height=460)
    with tables_tab:
        tables = markdown_tables(ocr["markdown"])
        if not tables:
            st.caption("No tables found in this document.")
        for number, table in enumerate(tables, start=1):
            st.markdown(f"**Table {number}**")
            st.dataframe(table, hide_index=True, width="stretch")
            st.download_button(f"Table {number} (CSV)", csv_bytes(table), f"{stem}_table{number}.csv", "text/csv",
                               key=f"table_{stem}_{number}")
    with details:
        rows = [{"field": field.replace("_", " "), "value": value}
                for field, values in ocr.get("fields", {}).items() for value in values]
        if rows:
            table = pd.DataFrame(rows)
            st.dataframe(table, hide_index=True, width="stretch")
            st.download_button("Key details (CSV)", csv_bytes(table), f"{stem}_details.csv", "text/csv")
        else:
            st.caption("No key details extracted.")

    d1, d2 = st.columns(2)
    d1.download_button("Text (.md)", ocr["markdown"].encode("utf-8"), f"{stem}.md", "text/markdown", width="stretch")
    d2.download_button("Text (.txt)", ocr["markdown"].encode("utf-8"), f"{stem}.txt", "text/plain", width="stretch")


def render(case: Case) -> None:
    image_id = image_picker(case, "documents")
    if not image_id:
        return
    left, right = st.columns([1, 1.25], gap="large")
    left.image(preview(case, image_id), caption=case.images[image_id]["filename"], width="stretch")

    with right:
        st.caption(VERIFY_NOTE)
        with_fields = st.checkbox("Also extract key details", value=True,
                                  help="Names, dates, amounts, addresses, phone numbers, references, emails")
        if st.button("Extract text", type="primary"):
            image = case.load_image(image_id)
            result = run_model(lambda: analysis.read_document(image, with_fields), "Reading the document…")
            if result is not None:
                case.update_image(image_id, ocr={**result, "created_at": now()})
        ocr = case.images[image_id].get("ocr")
        if ocr:
            _show(ocr, case.images[image_id]["filename"].rsplit(".", 1)[0])
        else:
            st.caption("Works on receipts, invoices, forms, handwritten notes, statements, screenshots and labels.")
