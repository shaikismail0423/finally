"""Documents & OCR tab (4.6): structured text, tables, key fields, quality flags and exports."""

import streamlit as st

from vas.core.case_store import now_iso
from vas.services.document_ocr import extract_document, fields_table, parse_markdown_tables
from vas.ui.components import image_picker, image_with_quality, no_image_placeholder, verify_banner
from vas.ui.context import AppContext
from vas.utils.export import dataframe_to_csv_bytes


def _extract(ctx: AppContext, image_id: str, with_fields: bool, high_res: bool) -> None:
    case = ctx.case
    if not case.images[image_id].get("is_document"):
        case.set_document_flag(image_id, True)  # also adds the skew check to quality
    image = case.load_image(image_id)
    max_pixels = ctx.settings.max_pixels_single * (2 if high_res else 1)
    result = ctx.run_model(
        lambda: extract_document(ctx.backend, image, max_pixels, with_fields),
        spinner=f"Reading {case.images[image_id]['filename']}…")
    if result is None:
        return
    result.update({"created_at": now_iso()})
    case.update_image(image_id, ocr=result)


def _show_result(ctx: AppContext, image_id: str, ocr: dict) -> None:
    rec = ctx.case.images[image_id]
    stem = rec["filename"].rsplit(".", 1)[0]
    issues = list(ocr.get("quality_issues", [])) + rec.get("quality", {}).get("flags", [])
    if issues:
        st.warning("**Double-check this document:** " + " · ".join(dict.fromkeys(issues)), icon=":material/warning:")
    st.caption(f"Document type: **{ocr.get('document_type', 'unknown')}** · read {ocr['created_at'].replace('T', ' ')}")

    t_struct, t_raw, t_tables, t_fields = st.tabs(["Structured text", "Plain text", "Tables", "Key details"])
    with t_struct:
        with st.container(height=460, border=True):
            st.markdown(ocr["markdown"])
    with t_raw:
        st.code(ocr["markdown"], language="markdown", wrap_lines=True, height=460)
    with t_tables:
        tables = parse_markdown_tables(ocr["markdown"])
        if not tables:
            st.caption("No tables detected in this document.")
        for n, df in enumerate(tables, start=1):
            st.markdown(f"**Table {n}**")
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button(f"Table {n} (CSV)", dataframe_to_csv_bytes(df), f"{stem}_table{n}.csv", "text/csv",
                               key=f"tbl_{image_id}_{n}", icon=":material/table:")
    with t_fields:
        table = fields_table(ocr.get("fields", {}), ocr.get("regex_fields", {}))
        if table.empty:
            st.caption("No key details found.")
        else:
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"source": st.column_config.TextColumn(
                             help="model + regex = found by both (higher trust); regex only = pattern match the model missed")})
            st.download_button("Key details (CSV)", dataframe_to_csv_bytes(table), f"{stem}_fields.csv", "text/csv",
                               key=f"fld_{image_id}", icon=":material/table:")

    d1, d2 = st.columns(2)
    d1.download_button("Text (.md)", ocr["markdown"].encode("utf-8"), f"{stem}.md", "text/markdown",
                       icon=":material/download:", width="stretch")
    d2.download_button("Text (.txt)", ocr["markdown"].encode("utf-8"), f"{stem}.txt", "text/plain",
                       icon=":material/download:", width="stretch")


def render(ctx: AppContext) -> None:
    image_id = image_picker(ctx, "document")
    if not image_id:
        no_image_placeholder()
        return
    rec = ctx.case.images[image_id]
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        image_with_quality(ctx, image_id)
    with right:
        st.markdown("#### Read this document")
        verify_banner()
        o1, o2 = st.columns(2)
        with_fields = o1.checkbox("Extract key details", value=True, help="Names, dates, amounts, addresses, phones, references")
        high_res = o2.checkbox("High resolution", value=False, help="Sends 2× more pixels — better for small print, slower.")
        if st.button("Extract text", type="primary", icon=":material/document_scanner:"):
            _extract(ctx, image_id, with_fields, high_res)
            rec = ctx.case.images[image_id]
        if rec.get("ocr"):
            _show_result(ctx, image_id, rec["ocr"])
        else:
            st.caption("Works on receipts, invoices, IDs, forms, handwritten notes, statements, chat screenshots and labels.")
