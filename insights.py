"""
insights.py
-----------
All EDA computations, chart helpers, and auto-generated text insights
for the Market Insights page.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st
from collections import Counter

log = logging.getLogger(__name__)

# ── colour palette ────────────────────────────────────────────────────────
COLORS = [
    "#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0",
    "#06D6A0", "#FFD166", "#EF476F", "#118AB2", "#073B4C",
]


# ════════════════════════════════════════════════════════════════════════════
# Individual chart functions
# ════════════════════════════════════════════════════════════════════════════

def chart_top_cities(df: pd.DataFrame, n: int = 15):
    """Bar chart: top cities by internship count (excluding Remote/Unknown/Multiple)."""
    city_counts = df[
        ~df["city"].isin(["Unknown", "Multiple", "Remote/WFH"])
    ]["city"].value_counts().head(n)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(city_counts.index[::-1], city_counts.values[::-1], color=COLORS[:n])
    ax.set_xlabel("Number of Internships")
    ax.set_title(f"Top {n} Cities by Internship Count")
    for bar, val in zip(bars, city_counts.values[::-1]):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    plt.tight_layout()
    return fig, city_counts


def chart_remote_vs_onsite(df: pd.DataFrame):
    """Pie chart: Remote vs On-site share."""
    remote_count = df["is_remote"].sum()
    onsite_count = len(df) - remote_count
    labels = [f"Remote/WFH\n({remote_count:,})", f"On-site\n({onsite_count:,})"]
    sizes = [remote_count, onsite_count]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=["#4CC9F0", "#F72585"], startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("Remote vs On-site Distribution")
    plt.tight_layout()
    return fig, {"Remote": remote_count, "On-site": onsite_count}


def chart_top_profiles(df: pd.DataFrame, n: int = 15):
    """Horizontal bar: top internship profiles / roles."""
    profile_counts = df["profile"].dropna().value_counts().head(n)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(profile_counts.index[::-1], profile_counts.values[::-1], color=COLORS[:n])
    ax.set_xlabel("Number of Internships")
    ax.set_title(f"Top {n} Internship Profiles")
    plt.tight_layout()
    return fig, profile_counts


def chart_top_skills(df: pd.DataFrame, n: int = 20):
    """Bar chart: top skills from skills_tokens."""
    all_skills: list = []
    for tokens in df["skills_tokens"]:
        if isinstance(tokens, list):
            all_skills.extend(tokens)

    counts = Counter(all_skills)
    # remove very generic tokens
    for drop in ["", "and", "or", "the", "a", "an", "of", "in", "to"]:
        counts.pop(drop, None)

    top = pd.Series(dict(counts.most_common(n)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top)), top.values, color=COLORS * 2)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Top {n} Skills in Demand")
    plt.tight_layout()
    return fig, top


def chart_stipend_type(df: pd.DataFrame):
    """Bar chart: stipend type distribution."""
    type_counts = df["stipend_type"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(type_counts.index, type_counts.values, color=COLORS[:len(type_counts)])
    ax.set_ylabel("Count")
    ax.set_title("Stipend Type Distribution")
    for i, (k, v) in enumerate(type_counts.items()):
        ax.text(i, v + 10, f"{v:,}", ha="center", fontsize=9)
    plt.tight_layout()
    return fig, type_counts


def chart_duration_distribution(df: pd.DataFrame):
    """Histogram: duration in months."""
    dur = df["duration_months"].dropna()
    dur = dur[dur <= 24]   # clip outliers

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dur, bins=24, color="#4361EE", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Duration (months)")
    ax.set_ylabel("Count")
    ax.set_title("Duration Distribution")
    ax.axvline(dur.median(), color="#F72585", linestyle="--", label=f"Median: {dur.median():.1f} mo")
    ax.legend()
    plt.tight_layout()
    return fig, dur


def chart_city_stipend(df: pd.DataFrame, n: int = 12):
    """Box-style bar: median stipend by city (if enough data)."""
    sub = df[
        df["stipend_max"].notna() &
        df["stipend_type"].isin(["fixed", "range"]) &
        ~df["city"].isin(["Unknown", "Multiple", "Remote/WFH"])
    ]
    if len(sub) < 50:
        return None, None

    city_med = (
        sub.groupby("city")["stipend_max"]
        .agg(["median", "count"])
        .query("count >= 5")
        .sort_values("median", ascending=False)
        .head(n)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(city_med.index[::-1], city_med["median"][::-1], color=COLORS[:n])
    ax.set_xlabel("Median Stipend (₹)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    ax.set_title(f"Median Stipend by City (Top {n})")
    plt.tight_layout()
    return fig, city_med


# ════════════════════════════════════════════════════════════════════════════
# Auto-generated insights text
# ════════════════════════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame) -> list:
    """Return list of insight bullet strings."""
    bullets = []

    total = len(df)
    bullets.append(f"📋 **Total internships in dataset:** {total:,}")

    # remote share
    remote_pct = df["is_remote"].mean() * 100
    bullets.append(f"🌐 **{remote_pct:.1f}%** of internships are Remote/WFH.")

    # top city
    top_city_series = df[~df["city"].isin(["Unknown", "Multiple", "Remote/WFH"])]["city"].value_counts()
    if not top_city_series.empty:
        tc, tc_cnt = top_city_series.index[0], top_city_series.iloc[0]
        bullets.append(f"📍 **{tc}** has the most on-site internships ({tc_cnt:,} listings).")

    # top profile
    top_profile = df["profile"].value_counts()
    if not top_profile.empty:
        tp, tp_cnt = top_profile.index[0], top_profile.iloc[0]
        bullets.append(f"💼 Most common role: **{tp}** ({tp_cnt:,} listings).")

    # top skill
    all_skills: list = []
    for tokens in df["skills_tokens"]:
        if isinstance(tokens, list):
            all_skills.extend(tokens)
    if all_skills:
        top_skill = Counter(all_skills).most_common(1)[0]
        bullets.append(f"🛠️ Most sought-after skill: **{top_skill[0]}** (appears {top_skill[1]:,} times).")

    # stipend stats
    paid = df[df["stipend_type"].isin(["fixed", "range"]) & df["stipend_max"].notna()]
    if len(paid) > 10:
        median_stip = paid["stipend_max"].median()
        mean_stip = paid["stipend_max"].mean()
        bullets.append(
            f"💰 Among paid internships — Median stipend: **₹{median_stip:,.0f}**, "
            f"Mean: **₹{mean_stip:,.0f}** per month."
        )

    unpaid_pct = (df["stipend_type"] == "unpaid").mean() * 100
    bullets.append(f"🚫 **{unpaid_pct:.1f}%** of listings are explicitly unpaid.")

    unknown_stip_pct = (df["stipend_type"] == "unknown").mean() * 100
    bullets.append(f"❓ **{unknown_stip_pct:.1f}%** of listings have unspecified/unparseable stipends.")

    # duration
    dur = df["duration_months"].dropna()
    if len(dur) > 10:
        bullets.append(
            f"⏱ Typical duration: **{dur.median():.1f} months** (median), "
            f"range {dur.min():.1f} – {dur.max():.1f} months."
        )

    # multi-city
    multi_pct = df["is_multi_city"].mean() * 100
    bullets.append(f"🗺️ **{multi_pct:.1f}%** of internships are open to multiple cities.")

    return bullets
