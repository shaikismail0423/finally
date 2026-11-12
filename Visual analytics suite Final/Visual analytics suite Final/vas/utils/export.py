"""Builders for downloadable files: CSV tables and a Markdown case report."""

from typing import List

import pandas as pd

from vas.core.case_store import Case, now_iso


def tags_table(case: Case, image_ids: List[str] = None) -> pd.DataFrame:
    """One row per image with its tags and focus-item hits."""
    rows = []
    for image_id in image_ids or list(case.images):
        rec = case.images[image_id]
        rows.append({"image_id": image_id, "filename": rec["filename"], "tags": "; ".join(rec.get("tags", [])),
                     "focus_hits": "; ".join(h["item"] for h in rec.get("focus_hits", [])),
                     "quality": rec.get("quality", {}).get("level", "")})
    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV with a UTF-8 BOM so Excel opens non-ASCII text correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


def case_report_markdown(case: Case) -> str:
    """Full Markdown report of everything produced for a case (for case notes)."""
    lines = [f"# Case report — {case.name}", "",
             f"*Generated {now_iso()} · Visual Analytics Suite (Qwen2.5-VL)*", "",
             "> **Decision support only.** All outputs are machine-generated observations that must be "
             "verified by an analyst before use in a case.", ""]
    if case.data.get("description"):
        lines += [case.data["description"], ""]
    lines += [f"**Images:** {len(case.images)}  ", f"**Focus list:** {', '.join(case.focus_tags) or '—'}", ""]

    for image_id, rec in case.images.items():
        lines += [f"## {rec['filename']}  (`{image_id}`)", ""]
        q = rec.get("quality", {})
        lines.append(f"- **Quality:** {q.get('level', '?')} ({q.get('score', '?')}/100)"
                     + (f" — {'; '.join(q.get('flags', []))}" if q.get("flags") else ""))
        if rec.get("tags"):
            lines.append(f"- **Tags:** {', '.join(rec['tags'])}")
        if rec.get("focus_hits"):
            lines.append("- **Focus items flagged:** " + "; ".join(f"{h['item']} ({h['evidence']})" for h in rec["focus_hits"]))
        for length, summary in rec.get("summaries", {}).items():
            lines.append(f"- **Summary ({length}):** {summary['text']}")
        for g in rec.get("groundings", []):
            lines.append(f"- **Highlighted '{g['target']}':** {len(g['detections'])} found")
        if rec.get("chat"):
            lines += ["", "**Questions & answers**", ""]
            for turn in rec["chat"]:
                who = "Analyst" if turn["role"] == "user" else "System"
                lines.append(f"- *{who}:* {turn['content']}")
        if rec.get("ocr"):
            lines += ["", f"**Document text** ({rec['ocr'].get('document_type', '')})", "", rec["ocr"]["markdown"]]
        lines.append("")

    if case.insights:
        lines += ["## Multi-image insights (observations to verify)", ""]
        for run in case.insights:
            files = ", ".join(case.images[i]["filename"] for i in run["image_ids"] if i in case.images)
            lines += [f"### Run {run['created_at']} — {files}", ""]
            for ins in run["result"]["insights"]:
                support = ", ".join(case.images[i]["filename"] for i in ins["image_ids"] if i in case.images)
                lines.append(f"- [{ins['confidence']}] **{ins['type']}**: {ins['description']} *(images: {support or '—'})*")
            lines.append("")
    return "\n".join(lines)
