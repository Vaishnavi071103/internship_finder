
# 🎓 Internship Matcher — India 2025
Checkout the live app here:  
[Open the Website](https://internshipfinder-fsistxjp6qdzqkfvdymcyn.streamlit.app/)

A Streamlit web app that matches you to internships from a ~8,400-listing Indian dataset using an explainable TF-IDF recommender.

---

## 📁 Setup

### 1. Place the dataset

```
data/merged_internships_dataset.csv
```

Create the `data/` folder if it doesn't exist. The raw CSV must have these columns:

`internship_id`, `date_time`, `profile`, `company`, `location`, `start_date`,
`stipend`, `duration`, `apply_by_date`, `offer`, `education`, `skills`, `perks`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 3000
```

On first run, the app preprocesses the raw CSV and saves a cache at `data/internships_processed.csv`. Subsequent startups load the cache directly (much faster).

---

## 📂 File Structure

```
project/
├── app.py              ← Streamlit UI & page routing
├── preprocess.py       ← CSV loading, cleaning, parsing pipeline
├── recommender.py      ← TF-IDF model, filter, rank, explain
├── insights.py         ← EDA charts & auto-generated insights
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── data/
    ├── merged_internships_dataset.csv   ← Raw input (you provide)
    └── internships_processed.csv        ← Auto-generated cache
```

---

## 🔍 Pages

| Page | Description |
|---|---|
| **Matcher** | Enter skills → get top 10 explainable recommendations |
| **Market Insights** | 7 charts + 10+ auto-generated insights about the dataset |
| **About** | Project overview, tech stack, how it works |

---

## ⚙️ How the Recommender Works

1. **Preprocessing** parses stipend, duration, dates, and location into structured fields.
2. A **TF-IDF model** is built on a `text_blob` field (profile + skills + perks + education).
3. **Hard filters** apply: location mode, stipend threshold, duration range.
4. **Scoring**: `0.75 × TF-IDF similarity + 0.15 × stipend_score + 0.10 × deadline_score`
5. **Explainability**: each result shows matched skills, why it was recommended, and scores.

---

## 🛠 Tech Stack

- **Streamlit** — UI
- **Pandas / NumPy** — data processing
- **scikit-learn** — TF-IDF + cosine similarity
- **python-dateutil** — date parsing
- **Matplotlib** — charts

---

## 💡 Tips

- If the first run is slow, wait — preprocessing 8k rows takes ~10–20 seconds.
- To force reprocess (e.g. after updating the CSV): delete `data/internships_processed.csv`.
- The TF-IDF model is cached in Streamlit's `@st.cache_resource` so it rebuilds only once per session.

  ```Test```
