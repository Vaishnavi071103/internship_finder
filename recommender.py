"""
recommender.py
--------------
TF-IDF content-based recommender with:
  - Hard-constraint filtering (location, stipend, duration)
  - Soft scoring (TF-IDF similarity + stipend + deadline)
  - Per-result explainability
"""

import re
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# TF-IDF model (call build_tfidf once; then reuse)
# ════════════════════════════════════════════════════════════════════════════

def build_tfidf(df: pd.DataFrame):
    """
    Fit TF-IDF on text_blob column.
    Returns (vectorizer, tfidf_matrix).
    Cache this in Streamlit with @st.cache_resource.
    """
    log.info("Fitting TF-IDF on text_blob …")
    texts = df["text_blob"].fillna("").tolist()
    vec = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(texts)
    log.info(f"TF-IDF fitted: {matrix.shape[0]} docs × {matrix.shape[1]} features")
    return vec, matrix


# ════════════════════════════════════════════════════════════════════════════
# Filtering helpers
# ════════════════════════════════════════════════════════════════════════════

def filter_internships(
    df: pd.DataFrame,
    location_mode: str,      # "Any" | "Remote/WFH only" | "City"
    city_query: str,         # used when location_mode == "City"
    min_stipend: float,
    dur_min: float,
    dur_max: float,
    include_unpaid: bool,
) -> pd.DataFrame:
    """
    Apply hard constraints and return filtered DataFrame.
    Missing values are generally allowed (benefit of the doubt).
    """
    mask = pd.Series([True] * len(df), index=df.index)

    # ── Location filter ──────────────────────────────────────────────────
    if location_mode == "Remote/WFH only":
        mask &= df["is_remote"] == True
    elif location_mode == "City" and city_query.strip():
        cq = city_query.strip().lower()
        city_mask = (
            df["city"].str.lower().str.contains(cq, na=False) |
            df["location"].str.lower().str.contains(cq, na=False) |
            (df["is_multi_city"] == True)   # multi-city listings might include the city
        )
        mask &= city_mask

    # ── Duration filter ──────────────────────────────────────────────────
    dur_known = df["duration_months"].notna()
    in_range = df["duration_months"].between(dur_min, dur_max)
    mask &= (~dur_known | in_range)   # allow rows where duration unknown

    # ── Stipend filter ───────────────────────────────────────────────────
    if min_stipend > 0:
        # unpaid
        if not include_unpaid:
            mask &= df["stipend_type"] != "unpaid"
        # known types must meet threshold
        known_types = df["stipend_type"].isin(["fixed", "range", "performance_based"])
        stipend_ok = df["stipend_max"].fillna(0) >= min_stipend
        # unknown stipend type → allow
        mask &= (~known_types | stipend_ok)
    else:
        if not include_unpaid:
            mask &= df["stipend_type"] != "unpaid"

    filtered = df[mask].copy()
    log.info(f"After filtering: {len(filtered):,} / {len(df):,} rows remain.")
    return filtered


# ════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ════════════════════════════════════════════════════════════════════════════

def _stipend_score(filtered: pd.DataFrame) -> pd.Series:
    """Normalize stipend_max to [0,1]; unknown → median."""
    s = filtered["stipend_max"].copy()
    median_val = s.median()
    if np.isnan(median_val):
        median_val = 0.0
    s = s.fillna(median_val)
    max_val = s.max()
    if max_val == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s / max_val).clip(0, 1)


def _deadline_score(filtered: pd.DataFrame, today: pd.Timestamp) -> pd.Series:
    """Nearer future deadlines → higher score; missing → 0.5 (neutral)."""
    dl = filtered["apply_by_dt"].copy()
    # days until deadline
    days = (dl - today).dt.days
    # clip to [0, 180]; normalize inverted so 0 days = score 1, 180 days = 0
    days_clipped = days.clip(lower=0, upper=180)
    score = 1 - (days_clipped / 180)
    score = score.where(dl.notna(), other=0.5)
    return score.clip(0, 1)


# ════════════════════════════════════════════════════════════════════════════
# Main recommendation function
# ════════════════════════════════════════════════════════════════════════════

def get_recommendations(
    df: pd.DataFrame,
    vec: TfidfVectorizer,
    tfidf_matrix,           # sparse matrix for the FULL df (same index order)
    user_skills: str,
    location_mode: str,
    city_query: str,
    min_stipend: float,
    dur_min: float,
    dur_max: float,
    include_unpaid: bool,
    sort_mode: str,         # "Best match" | "Highest stipend" | "Closest deadline"
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Filter → score → explain → return top_n rows with extra columns.
    """

    # ── 1. Filter ────────────────────────────────────────────────────────
    filtered = filter_internships(
        df, location_mode, city_query,
        min_stipend, dur_min, dur_max, include_unpaid
    )

    if filtered.empty:
        return pd.DataFrame()   # caller handles empty case

    # ── 2. TF-IDF similarity ─────────────────────────────────────────────
    query = user_skills.strip() if user_skills.strip() else "internship"
    query_vec = vec.transform([query])

    # slice tfidf_matrix to filtered rows
    filtered_indices = filtered.index.tolist()
    # map df integer positions
    all_indices = list(df.index)
    pos_map = {idx: pos for pos, idx in enumerate(all_indices)}
    positions = [pos_map[i] for i in filtered_indices]

    filtered_matrix = tfidf_matrix[positions]
    sim_scores = cosine_similarity(query_vec, filtered_matrix).flatten()
    filtered = filtered.copy()
    filtered["similarity_score"] = sim_scores

    # ── 3. Stipend + deadline scores ─────────────────────────────────────
    today = pd.Timestamp.now().normalize()
    filtered["_stip_score"] = _stipend_score(filtered).values
    filtered["_dl_score"] = _deadline_score(filtered, today).values

    # ── 4. Final composite score ─────────────────────────────────────────
    filtered["final_score"] = (
        0.75 * filtered["similarity_score"]
        + 0.15 * filtered["_stip_score"]
        + 0.10 * filtered["_dl_score"]
    )

    # ── 5. Sort ──────────────────────────────────────────────────────────
    if sort_mode == "Highest stipend":
        filtered = filtered.sort_values("stipend_max", ascending=False, na_position="last")
    elif sort_mode == "Closest deadline":
        filtered = filtered.sort_values("apply_by_dt", ascending=True, na_position="last")
    else:  # Best match
        filtered = filtered.sort_values("final_score", ascending=False)

    top = filtered.head(top_n).copy()

    # ── 6. Explainability ────────────────────────────────────────────────
    user_tokens = set(re.split(r"[,\s]+", user_skills.lower()))
    user_tokens = {t.strip() for t in user_tokens if t.strip()}

    def explain(row):
        matched = user_tokens & set(row.get("skills_tokens") or [])
        parts = []
        if matched:
            parts.append(f"✅ Matched skills: {', '.join(sorted(matched))}")
        else:
            parts.append("⚠️ No direct skill match (matched via job description)")

        if row.get("is_remote"):
            parts.append("🌐 Remote/WFH position")
        elif row.get("city") and row["city"] not in ("Unknown", "Multiple"):
            parts.append(f"📍 City: {row['city']}")

        stype = row.get("stipend_type", "unknown")
        if stype == "unpaid":
            parts.append("💰 Unpaid internship")
        elif stype in ("fixed", "range") and pd.notna(row.get("stipend_max")):
            parts.append(f"💰 Stipend up to ₹{int(row['stipend_max']):,}")
        elif stype == "performance_based":
            parts.append("💰 Performance-based stipend")

        if pd.notna(row.get("duration_months")):
            parts.append(f"⏱ Duration: {row['duration_months']} months")

        sim = row.get("similarity_score", 0)
        final = row.get("final_score", 0)
        parts.append(f"📊 Similarity: {sim:.3f} | Final score: {final:.3f}")

        return " | ".join(parts)

    top["matched_skills"] = top.apply(
        lambda r: sorted(user_tokens & set(r.get("skills_tokens") or [])), axis=1
    )
    top["why_recommended"] = top.apply(explain, axis=1)

    return top


# ════════════════════════════════════════════════════════════════════════════
# Top cities helper (for sidebar dropdown)
# ════════════════════════════════════════════════════════════════════════════

def get_top_cities(df: pd.DataFrame, n: int = 30) -> list:
    """Return top n non-remote, non-multiple cities."""
    cities = df[
        (~df["is_remote"]) &
        (~df["is_multi_city"]) &
        (df["city"].notna()) &
        (~df["city"].isin(["Unknown", "Multiple", "Remote/WFH"]))
    ]["city"].value_counts().head(n).index.tolist()
    return sorted(cities)
