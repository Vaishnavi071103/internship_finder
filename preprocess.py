"""
preprocess.py
-------------
Handles loading raw CSV, cleaning, parsing structured fields,
and saving the processed cache CSV.

Run standalone to regenerate cache:
    python preprocess.py
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from dateutil import parser as date_parser

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RAW_PATH = "data/merged_internships_dataset.csv"
PROCESSED_PATH = "data/internships_processed.csv"

# ── tokens that count as "missing" ──────────────────────────────────────────
MISSING_TOKENS = {
    "n/a", "na", "not specified", "not available", "none", "null",
    "-", "--", "---", "unknown", "not mentioned", "not applicable", "",
}


# ════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ════════════════════════════════════════════════════════════════════════════

def normalize_missing(val):
    """Return np.nan for any missing-like string, else the stripped string."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in MISSING_TOKENS:
        return np.nan
    return s


def clean_number(s):
    """Strip commas and extract first float from a string."""
    s = re.sub(r"[,\s]", "", str(s))
    m = re.search(r"[\d.]+", s)
    return float(m.group()) if m else None


# ════════════════════════════════════════════════════════════════════════════
# Stipend parser
# ════════════════════════════════════════════════════════════════════════════

def parse_stipend(raw):
    """
    Returns dict with keys:
        stipend_min, stipend_max, stipend_type, stipend_period
    """
    result = {
        "stipend_min": np.nan,
        "stipend_max": np.nan,
        "stipend_type": "unknown",
        "stipend_period": "unknown",
    }
    if pd.isna(raw):
        return result

    s = str(raw).strip().lower()

    # unpaid
    if "unpaid" in s:
        result.update(stipend_min=0.0, stipend_max=0.0, stipend_type="unpaid")
        return result

    # performance-based
    if "performance" in s:
        result["stipend_type"] = "performance_based"
        # still try to extract numbers below

    # period detection
    if "month" in s or "mon" in s:
        result["stipend_period"] = "month"
    elif "week" in s:
        result["stipend_period"] = "week"
    elif "lump" in s or "one time" in s or "onetime" in s:
        result["stipend_period"] = "lump_sum"

    # extract numbers
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", s.replace(",", ""))
    nums = []
    for n in numbers:
        try:
            nums.append(float(n.replace(",", "")))
        except ValueError:
            pass

    # filter out obviously wrong values (e.g. year numbers like 2024)
    nums = [n for n in nums if n < 500000]

    if len(nums) == 0:
        if result["stipend_type"] != "performance_based":
            result["stipend_type"] = "unknown"
    elif len(nums) == 1:
        result["stipend_min"] = nums[0]
        result["stipend_max"] = nums[0]
        if result["stipend_type"] != "performance_based":
            result["stipend_type"] = "fixed"
    else:
        result["stipend_min"] = min(nums)
        result["stipend_max"] = max(nums)
        if result["stipend_type"] != "performance_based":
            result["stipend_type"] = "range"

    return result


# ════════════════════════════════════════════════════════════════════════════
# Duration parser
# ════════════════════════════════════════════════════════════════════════════

def parse_duration(raw):
    """Returns duration_months as float, or np.nan."""
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip().lower()
    try:
        # months
        m = re.search(r"([\d.]+)\s*month", s)
        if m:
            return float(m.group(1))
        # weeks
        m = re.search(r"([\d.]+)\s*week", s)
        if m:
            return round(float(m.group(1)) / 4.345, 2)
        # days
        m = re.search(r"([\d.]+)\s*day", s)
        if m:
            return round(float(m.group(1)) / 30, 2)
        # bare number — assume months
        m = re.search(r"^([\d.]+)$", s.strip())
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return np.nan


# ════════════════════════════════════════════════════════════════════════════
# Date parser
# ════════════════════════════════════════════════════════════════════════════

def safe_parse_date(val):
    """Parse a date string to datetime, returns NaT on failure."""
    if pd.isna(val):
        return pd.NaT
    try:
        return pd.to_datetime(val, errors="coerce", infer_datetime_format=True)
    except Exception:
        return pd.NaT


# ════════════════════════════════════════════════════════════════════════════
# Location parser
# ════════════════════════════════════════════════════════════════════════════

REMOTE_KEYWORDS = {"work from home", "wfh", "remote", "work from home (wfh)"}

def parse_location(raw):
    """
    Returns dict:
        is_remote (bool), city (str), is_multi_city (bool)
    """
    result = {"is_remote": False, "city": "Unknown", "is_multi_city": False}
    if pd.isna(raw):
        return result

    s = str(raw).strip()
    sl = s.lower()

    # remote check
    for kw in REMOTE_KEYWORDS:
        if kw in sl:
            result["is_remote"] = True
            result["city"] = "Remote/WFH"
            return result

    # multi-city detection
    separators = re.split(r"[,/|&+]", s)
    separators = [c.strip() for c in separators if c.strip()]

    multi_keywords = {"multiple", "pan india", "various", "all india", "anywhere"}
    if any(mk in sl for mk in multi_keywords):
        result["is_multi_city"] = True
        result["city"] = "Multiple"
        return result

    if len(separators) > 1:
        result["is_multi_city"] = True
        result["city"] = "Multiple"
        return result

    # single city — clean up common suffixes
    city = re.sub(r"\s*\(.*?\)", "", s).strip()  # remove parentheses
    city = re.sub(r"\s*(district|state|india).*", "", city, flags=re.IGNORECASE).strip()
    result["city"] = city if city else "Unknown"
    return result


# ════════════════════════════════════════════════════════════════════════════
# Skills tokenizer
# ════════════════════════════════════════════════════════════════════════════

def tokenize_skills(raw):
    """Returns list of lowercase skill tokens."""
    if pd.isna(raw):
        return []
    tokens = re.split(r"[,|;/]", str(raw))
    return [t.strip().lower() for t in tokens if t.strip()]


# ════════════════════════════════════════════════════════════════════════════
# Main preprocessing function
# ════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(force=False):
    """
    Load processed CSV if available, else run full pipeline.
    Set force=True to always reprocess.
    """
    if not force and os.path.exists(PROCESSED_PATH):
        log.info(f"Loading cached processed data from {PROCESSED_PATH}")
        df = pd.read_csv(PROCESSED_PATH, low_memory=False)
        
        # restore list column from string
        df["skills_tokens"] = df["skills_tokens"].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else []
        )
        
        # ── Datetime Fix: Convert strings back to datetime objects ──
        df["apply_by_dt"] = pd.to_datetime(df["apply_by_dt"], errors="coerce")
        df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
        df["posted_dt"] = pd.to_datetime(df["posted_dt"], errors="coerce")

        log.info(f"Loaded {len(df):,} rows from cache.")
        return df

    log.info(f"Loading raw data from {RAW_PATH}")
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at '{RAW_PATH}'. "
            "Please place merged_internships_dataset.csv in the data/ folder."
        )

    df = pd.read_csv(RAW_PATH, low_memory=False)
    log.info(f"Raw rows loaded: {len(df):,} | Columns: {list(df.columns)}")

    # ── Step 1: Normalize missing-like tokens ────────────────────────────
    log.info("Step 1: Normalizing missing tokens …")
    text_cols = ["profile", "company", "location", "start_date", "stipend",
                 "duration", "apply_by_date", "offer", "education", "skills",
                 "perks", "date_time"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_missing)

    # ── Step 2: Parse stipend ─────────────────────────────────────────────
    log.info("Step 2: Parsing stipend …")
    stipend_parsed = df["stipend"].apply(parse_stipend)
    df["stipend_min"] = stipend_parsed.apply(lambda x: x["stipend_min"])
    df["stipend_max"] = stipend_parsed.apply(lambda x: x["stipend_max"])
    df["stipend_type"] = stipend_parsed.apply(lambda x: x["stipend_type"])
    df["stipend_period"] = stipend_parsed.apply(lambda x: x["stipend_period"])

    # ── Step 3: Parse duration ────────────────────────────────────────────
    log.info("Step 3: Parsing duration …")
    df["duration_months"] = df["duration"].apply(parse_duration)

    # ── Step 4: Parse dates ───────────────────────────────────────────────
    log.info("Step 4: Parsing dates …")
    df["apply_by_dt"] = pd.to_datetime(df["apply_by_date"], errors="coerce", infer_datetime_format=True)
    df["posted_dt"] = pd.to_datetime(df["date_time"], errors="coerce", infer_datetime_format=True)

    # start_date: detect "immediate"
    df["start_immediate"] = df["start_date"].apply(
        lambda x: bool(re.search(r"immediate", str(x), re.IGNORECASE)) if pd.notna(x) else False
    )
    df["start_dt"] = df["start_date"].apply(
        lambda x: pd.NaT if (pd.isna(x) or re.search(r"immediate", str(x), re.IGNORECASE))
        else pd.to_datetime(x, errors="coerce", infer_datetime_format=True)
    )

    # ── Step 5: Normalize location ────────────────────────────────────────
    log.info("Step 5: Parsing location …")
    loc_parsed = df["location"].apply(parse_location)
    df["is_remote"] = loc_parsed.apply(lambda x: x["is_remote"])
    df["city"] = loc_parsed.apply(lambda x: x["city"])
    df["is_multi_city"] = loc_parsed.apply(lambda x: x["is_multi_city"])

    # ── Step 6: Skills tokens + text blob ────────────────────────────────
    log.info("Step 6: Building skills tokens and text_blob …")
    df["skills_tokens"] = df["skills"].apply(tokenize_skills)

    def make_blob(row):
        parts = [
            str(row.get("profile", "") or ""),
            str(row.get("skills", "") or ""),
            str(row.get("perks", "") or ""),
            str(row.get("education", "") or ""),
            str(row.get("offer", "") or ""),
            str(row.get("company", "") or ""),
        ]
        return " ".join(p for p in parts if p and p.lower() != "nan")

    df["text_blob"] = df.apply(make_blob, axis=1)

    # ── Save processed CSV ────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    # skills_tokens saved as string repr (list)
    df_save = df.copy()
    df_save["skills_tokens"] = df_save["skills_tokens"].apply(str)
    df_save.to_csv(PROCESSED_PATH, index=False)
    log.info(f"Saved processed data to {PROCESSED_PATH} ({len(df):,} rows).")

    return df


if __name__ == "__main__":
    df = load_and_preprocess(force=True)
    print(df.head(3))
    print(df.dtypes)