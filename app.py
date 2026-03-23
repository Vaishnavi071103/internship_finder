"""
app.py
------
Streamlit entry-point for "Internship Matcher (India 2025)".

Run with:
    streamlit run app.py --server.address 0.0.0.0 --server.port 3000

Pages:
  1. Matcher       – Explainable recommender
  2. Market Insights – EDA charts + auto insights
  3. About         – Project info
"""

import logging
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── project modules ───────────────────────────────────────────────────────
from preprocess import load_and_preprocess
from recommender import build_tfidf, get_recommendations, get_top_cities
from insights import (
    chart_top_cities, chart_remote_vs_onsite, chart_top_profiles,
    chart_top_skills, chart_stipend_type, chart_duration_distribution,
    chart_city_stipend, generate_insights,
)

# ════════════════════════════════════════════════════════════════════════════
# Page config (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Internship Matcher India 2025",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .card {
        background: #f8f9fc;
        border-left: 4px solid #4361EE;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .card h4 { margin: 0 0 4px 0; color: #1a1a2e; }
    .card .meta { font-size: 0.85rem; color: #555; }
    .card .why { font-size: 0.82rem; color: #4361EE; margin-top: 8px; }
    .score-badge {
        background: #4361EE; color: white;
        border-radius: 4px; padding: 2px 8px; font-size: 0.78rem;
    }
    .tag {
        display: inline-block;
        background: #e8ecff; color: #3a0ca3;
        border-radius: 12px; padding: 2px 10px;
        font-size: 0.78rem; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Cached data loaders
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading & preprocessing dataset …")
def load_data():
    # Adding fixing Changes as gpt said
    return load_and_preprocess()


@st.cache_resource(show_spinner="Building TF-IDF model …")
def load_tfidf(df_hash):
    """Cache TF-IDF model – df_hash forces rebuild when data changes."""
    return build_tfidf(st.session_state["_df"])


# ════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/internship.png", width=64)
    st.title("Internship Matcher")
    st.caption("India 2025 — ~8.4k listings")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🔍 Matcher", "📊 Market Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

# ════════════════════════════════════════════════════════════════════════════
# Load data once
# ════════════════════════════════════════════════════════════════════════════

try:
    df = load_data()
    st.session_state["_df"] = df   # needed by load_tfidf
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – MATCHER
# ════════════════════════════════════════════════════════════════════════════

if page == "🔍 Matcher":
    st.title("🔍 Internship Matcher")
    st.markdown("Find internships that match your skills and preferences. Results include **explainability** for every recommendation.")

    # ── Sidebar inputs ───────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("Your Preferences")

        user_skills = st.text_input(
            "Skills (comma-separated) *",
            placeholder="e.g. Python, Machine Learning, SQL",
            help="Required. List your skills separated by commas.",
        )

        loc_mode = st.selectbox(
            "Location Preference",
            ["Any", "Remote/WFH only", "Specific City"],
        )

        city_query = ""
        if loc_mode == "Specific City":
            top_cities = get_top_cities(df, 30)
            city_choice = st.selectbox("Select a city", ["(type below)"] + top_cities)
            custom_city = st.text_input("Or type city name", placeholder="e.g. Pune")
            city_query = custom_city.strip() if custom_city.strip() else (
                city_choice if city_choice != "(type below)" else ""
            )

        min_stipend = st.number_input(
            "Minimum Stipend (₹/month)", min_value=0, step=1000, value=0,
        )

        dur_min, dur_max = st.slider(
            "Duration (months)", min_value=1, max_value=12, value=(1, 6),
        )

        include_unpaid = st.checkbox("Include unpaid internships", value=False)

        sort_mode = st.radio(
            "Sort by",
            ["Best match", "Highest stipend", "Closest deadline"],
            index=0,
        )

        run_btn = st.button("🔎 Find Internships", type="primary", use_container_width=True)

    # ── Build/load TF-IDF ────────────────────────────────────────────────
    vec, tfidf_matrix = load_tfidf(len(df))

    # ── Run recommender ──────────────────────────────────────────────────
    if run_btn or user_skills:
        if not user_skills.strip():
            st.warning("⚠️ Please enter at least one skill to get recommendations.")
            st.stop()

        with st.spinner("Finding best matches …"):
            results = get_recommendations(
                df=df,
                vec=vec,
                tfidf_matrix=tfidf_matrix,
                user_skills=user_skills,
                location_mode=loc_mode if loc_mode != "Specific City" else "City",
                city_query=city_query,
                min_stipend=float(min_stipend),
                dur_min=float(dur_min),
                dur_max=float(dur_max),
                include_unpaid=include_unpaid,
                sort_mode=sort_mode,
                top_n=10,
            )

        if results.empty:
            st.error("😕 No internships matched your filters.")
            st.info(
                "**Suggestions to get results:**\n"
                "- Broaden location to 'Any'\n"
                "- Lower the minimum stipend\n"
                "- Widen the duration range\n"
                "- Check 'Include unpaid internships'"
            )
        else:
            st.success(f"✅ Showing top {len(results)} recommendations for: **{user_skills}**")

            # ── Download button ──────────────────────────────────────────
            download_cols = [
                "profile", "company", "location", "stipend", "stipend_min",
                "stipend_max", "stipend_type", "duration", "duration_months",
                "apply_by_date", "offer", "why_recommended",
                "similarity_score", "final_score",
            ]
            available_cols = [c for c in download_cols if c in results.columns]
            csv_bytes = results[available_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_bytes,
                file_name="internship_recommendations.csv",
                mime="text/csv",
            )

            st.divider()

            # ── Result cards ─────────────────────────────────────────────
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                profile = row.get("profile") or "Unknown Profile"
                company = row.get("company") or "Unknown Company"
                location = row.get("location") or "—"
                stipend_raw = row.get("stipend") or "—"
                duration_raw = row.get("duration") or "—"
                apply_by = row.get("apply_by_date") or "—"
                offer = row.get("offer") or ""
                why = row.get("why_recommended", "")
                matched = row.get("matched_skills", [])
                score = row.get("final_score", 0.0)

                # parsed extras
                dur_m = row.get("duration_months")
                dur_str = f"{dur_m:.1f} mo" if pd.notna(dur_m) else "—"
                stip_max = row.get("stipend_max")
                stip_str = f"₹{int(stip_max):,}" if pd.notna(stip_max) else "—"

                matched_html = "".join(f'<span class="tag">{s}</span>' for s in matched)

                st.markdown(f"""
<div class="card">
  <h4>#{rank} {profile}
    <span class="score-badge" style="float:right">Score: {score:.3f}</span>
  </h4>
  <div class="meta">
    🏢 <b>{company}</b> &nbsp;|&nbsp;
    📍 {location} &nbsp;|&nbsp;
    💰 {stipend_raw} ({stip_str}) &nbsp;|&nbsp;
    ⏱ {duration_raw} ({dur_str}) &nbsp;|&nbsp;
    📅 Apply by: {apply_by}
  </div>
  {'<div class="meta" style="margin-top:4px">📜 ' + offer[:120] + ('…' if len(str(offer)) > 120 else '') + '</div>' if offer else ''}
  {'<div style="margin-top:6px">' + matched_html + '</div>' if matched_html else ''}
  <div class="why">{why}</div>
</div>
""", unsafe_allow_html=True)

    else:
        st.info("👈 Enter your skills in the sidebar and click **Find Internships** to get started.")

        # Quick stats strip
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Listings", f"{len(df):,}")
        c2.metric("Remote/WFH", f"{df['is_remote'].sum():,}")
        c3.metric("Cities Covered", f"{df['city'].nunique():,}")
        c4.metric("Unique Profiles", f"{df['profile'].nunique():,}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – MARKET INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

elif page == "📊 Market Insights":
    st.title("📊 Market Insights — India Internship Landscape 2025")
    st.markdown("Data-driven view of the internship market based on ~8,400 listings.")

    # ── Auto-generated insights ──────────────────────────────────────────
    st.subheader("🔑 Key Insights")
    insights = generate_insights(df)
    for bullet in insights:
        st.markdown(f"- {bullet}")

    st.divider()

    # ── Charts grid ──────────────────────────────────────────────────────

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏙️ Top Cities by Internship Count")
        fig, _ = chart_top_cities(df)
        st.pyplot(fig)

    with col2:
        st.subheader("🌐 Remote vs On-site")
        fig, _ = chart_remote_vs_onsite(df)
        st.pyplot(fig)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("💼 Top Internship Profiles")
        fig, _ = chart_top_profiles(df)
        st.pyplot(fig)

    with col4:
        st.subheader("🛠️ Top Skills in Demand")
        fig, _ = chart_top_skills(df)
        st.pyplot(fig)

    st.divider()

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("💰 Stipend Type Distribution")
        fig, _ = chart_stipend_type(df)
        st.pyplot(fig)

    with col6:
        st.subheader("⏱ Duration Distribution")
        fig, _ = chart_duration_distribution(df)
        st.pyplot(fig)

    # ── Optional: city-wise stipend ──────────────────────────────────────
    st.divider()
    st.subheader("🏙️ Median Stipend by City")
    fig_cs, city_stip_data = chart_city_stipend(df)
    if fig_cs is not None:
        st.pyplot(fig_cs)
        st.caption("Only cities with ≥5 parsed stipend entries shown.")
    else:
        st.info("Not enough parsed stipend data to show city-wise summary.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – ABOUT
# ════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About — Internship Matcher India 2025")

    st.markdown("""
### What is this app?

**Internship Matcher (India 2025)** is an explainable recommendation system for Indian internship seekers.
It processes a dataset of ~8,400 internship listings and matches them to your skills using
TF-IDF content-based filtering with transparent explanations for every result.

---

### File Structure

```
project/
├── app.py              ← Streamlit UI & page routing
├── preprocess.py       ← CSV loading, cleaning, parsing pipeline
├── recommender.py      ← TF-IDF model, filter, rank, explain
├── insights.py         ← EDA charts & auto-generated insights
├── requirements.txt    ← Python dependencies
├── README.md           ← Setup & usage guide
└── data/
    ├── merged_internships_dataset.csv   ← Raw input (you provide)
    └── internships_processed.csv        ← Auto-generated cache
```

---

### How the Recommender Works

1. **Preprocessing** — Stipend text, duration strings, dates, and locations are parsed into structured fields.
2. **TF-IDF Model** — Built on a `text_blob` combining profile + skills + perks + education + offer.
3. **Filtering** — Hard constraints: location mode, stipend threshold, duration range.
4. **Scoring** — `score = 0.75×similarity + 0.15×stipend_score + 0.10×deadline_score`
5. **Explainability** — Each result shows matched skills, constraints satisfied, and scores.

---

### How to Run (Replit / Local)

```bash
# 1. Place dataset
cp /path/to/merged_internships_dataset.csv data/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run app
streamlit run app.py --server.address 0.0.0.0 --server.port 3000
```

On first run, `data/internships_processed.csv` is generated and cached for faster subsequent starts.

---

### Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Data | Pandas, NumPy |
| ML / NLP | scikit-learn (TF-IDF, cosine similarity) |
| Date parsing | python-dateutil |
| Charts | Matplotlib |

---

### Dataset Schema (Raw CSV)

`internship_id`, `date_time`, `profile`, `company`, `location`, `start_date`,
`stipend`, `duration`, `apply_by_date`, `offer`, `education`, `skills`, `perks`
    """)

    st.info("Built with ❤️ for Indian internship seekers — powered by Python & Streamlit.")
