"""Custom CSS for a clean, professional look (works in light and dark themes)."""

import streamlit as st

_CSS = """
<style>
/* Layout */
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}

/* Header */
.vas-header {display:flex; align-items:center; justify-content:space-between; gap:1rem;
             padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 0.8rem;
             background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 55%, #0EA5E9 100%); color: #fff;}
.vas-header h1 {font-size: 1.55rem; margin: 0; color:#fff; font-weight: 700; letter-spacing: -0.01em;}
.vas-header p {margin: 0.15rem 0 0 0; opacity: 0.9; font-size: 0.92rem;}
.vas-pill {display:inline-block; padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.8rem;
           font-weight: 600; background: rgba(255,255,255,0.18); color:#fff; white-space: nowrap;}
.vas-pill.ok::before {content:"●  "; color:#4ADE80;}
.vas-pill.bad::before {content:"●  "; color:#F87171;}

/* Badges & chips */
.vas-badge {display:inline-block; padding: 0.12rem 0.55rem; border-radius: 6px; font-size: 0.75rem;
            font-weight: 600; margin-right: 0.3rem;}
.vas-good {background: rgba(22,163,74,0.15); color: #15803D;}
.vas-fair {background: rgba(245,158,11,0.18); color: #B45309;}
.vas-poor {background: rgba(220,38,38,0.15); color: #B91C1C;}
.vas-info {background: rgba(37,99,235,0.12); color: #1D4ED8;}
.vas-chip {display:inline-block; padding: 0.1rem 0.5rem; margin: 0.12rem 0.15rem 0.12rem 0; border-radius: 999px;
           font-size: 0.74rem; background: rgba(100,116,139,0.14); border: 1px solid rgba(100,116,139,0.25);}
.vas-chip.focus {background: rgba(220,38,38,0.12); border-color: rgba(220,38,38,0.4); color:#B91C1C; font-weight:600;}

/* Cards */
.vas-card-title {font-weight: 600; font-size: 0.86rem; margin: 0.35rem 0 0.15rem 0; overflow:hidden;
                 text-overflow: ellipsis; white-space: nowrap;}
.vas-insight {border-left: 4px solid #2563EB; padding: 0.6rem 0.9rem; margin: 0.5rem 0;
              background: rgba(37,99,235,0.06); border-radius: 0 8px 8px 0;}
.vas-insight.low {border-left-color: #F59E0B; background: rgba(245,158,11,0.07);}
.vas-insight.high {border-left-color: #16A34A; background: rgba(22,163,74,0.07);}
.vas-muted {opacity: 0.7; font-size: 0.82rem;}

/* KPI metrics */
div[data-testid="stMetric"] {background: rgba(100,116,139,0.07); border: 1px solid rgba(100,116,139,0.18);
                             padding: 0.6rem 0.9rem; border-radius: 10px;}
/* Tabs */
button[data-baseweb="tab"] p {font-size: 0.93rem; font-weight: 600;}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
