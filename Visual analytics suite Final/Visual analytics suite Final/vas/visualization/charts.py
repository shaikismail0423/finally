"""
Plotly charts used across the UI. Each function takes plain data and returns
a figure, so charts can be reused or tested without Streamlit.
"""

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PRIMARY = "#2563EB"
GOOD, WARN, BAD = "#16A34A", "#F59E0B", "#DC2626"


def _style(fig: go.Figure, height: int = 320) -> go.Figure:
    """Consistent, clean styling for every chart."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      title_font=dict(size=15))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def tag_frequency_chart(tags_by_image: Dict[str, List[str]], top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of the most common tags in a case."""
    counts: Dict[str, int] = {}
    for tags in tags_by_image.values():
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    df = pd.DataFrame(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n], columns=["tag", "images"])
    fig = px.bar(df.iloc[::-1], x="images", y="tag", orientation="h", title="Most common tags",
                 color_discrete_sequence=[PRIMARY])
    return _style(fig, height=max(260, 24 * len(df) + 80))


def quality_chart(quality_rows: pd.DataFrame) -> go.Figure:
    """Bar chart of image-quality scores, coloured by level."""
    colors = {"Good": GOOD, "Fair": WARN, "Poor": BAD}
    fig = px.bar(quality_rows, x="image", y="score", color="level", color_discrete_map=colors,
                 title="Image quality score (0-100)", range_y=[0, 100])
    fig.update_xaxes(tickangle=-30)
    return _style(fig)
