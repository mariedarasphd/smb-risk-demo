# -------------------------------------------------
# app.py – Streamlit demo (no caching, robust CSV load, debug info)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import pathlib
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -------------------------------------------------
# 0️⃣  Custom CSS + logo (Tiffany blue background)
# -------------------------------------------------
CUSTOM_CSS = """
body {
    background-color: #0ABAB5;      /* Tiffany blue */
    color: #ffffff;                /* White text */
}
[data-testid="stSidebar"] {
    background-color: #0ABAB5;
}
section[data-testid="stHeader"] {
    background-color: #0ABAB5;
}
footer {
    background-color: #0ABAB5;
}
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
}
.logo-img {
    max-height: 60px;
    margin-right: 12px;
}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -------------------------------------------------
# 0️⃣‑B  Show the logo (sidebar)
# -------------------------------------------------
logo_path = pathlib.Path(__file__).parent / "logo.png"
st.sidebar.image(str(logo_path), width=120)

# -------------------------------------------------
# DEBUG INFO – show what the container sees
# -------------------------------------------------
st.subheader("🔎 Debug info (remove later)")
cwd = pathlib.Path.cwd()
st.write(f"**Current working directory:** `{cwd}`")
st.write("**Files in repo root:**", sorted([p.name for p in cwd.iterdir() if p.is_file()]))

# -------------------------------------------------
# 1️⃣  Load the sample CSV (no caching, robust date handling)
# -------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Reads the CSV that lives next to this script.
    Returns a fresh copy that can be mutated safely.
    """
    data_path = pathlib.Path(__file__).parent / "sample_flagged.csv"

    # ---- sanity check: does the file exist? ----
    if not data_path.is_file():
        raise FileNotFoundError(
            f"❌ CSV not found at {data_path}. "
            "Make sure the file is named exactly 'sample_flagged.csv' "
            "and lives in the repository root."
        )

    # ---- read the CSV (use the safe Python engine) ----
    try:
        df = pd.read_csv(
            data_path,
            engine="python",          # fallback engine – works on any CSV
            encoding="utf-8",        # most CSVs are UTF‑8
        )
    except Exception as exc:
        st.error(
            f"❌ **Failed to read CSV** at `{data_path}`.\n"
            f"**Pandas error:** `{type(exc).__name__}` – {exc}"
        )
        raise

    # -------------------------------------------------
    # 2️⃣  Convert any *existing* date columns to datetime
    # -------------------------------------------------
    expected_date_cols = ["order_date_time", "synthetic_date", "Survey_response_Date"]
    existing_date_cols = [col for col in expected_date_cols if col in df.columns]

    if existing_date_cols:
        try:
            # Automatic inference with dayfirst=True (handles "10-08-2023 15:52")
            df[existing_date_cols] = df[existing_date_cols].apply(
                lambda col: pd.to_datetime(col, dayfirst=True, errors="coerce")
            )
            # If you prefer an explicit format, uncomment the block below:
            # fmt = "%d-%m-%Y %H:%M"
            # df[existing_date_cols] = df[existing_date_cols].apply(
            #     lambda col: pd.to_datetime(col, format=fmt, errors="coerce")
            # )
        except Exception as exc:
            st.warning(
                f"⚠️ Could not parse dates for columns {existing_date_cols}: {exc}. "
                "They will remain as strings."
            )

    # Return a fresh copy so downstream code can safely mutate it
    return df.copy()

# Load the data (debug panel already shows the file list)
df = load_data()

# -------------------------------------------------
# 2️⃣  UI layout
# -------------------------------------------------
st.set_page_config(
    page_title="SMB Risk Dashboard",
    layout="wide",
    page_icon="🔍"
)

st.title("🔎 SMB Customer‑Sentiment + Transaction Risk Dashboard")
st.markdown(
    """
    A lightweight demo that joins **customer remarks** with **synthetic transaction data**, 
    runs **sentiment analysis**, and highlights high‑value, negative‑sentiment cases.
    """
)

# ---- Sidebar filters -------------------------------------------------
st.sidebar.header("🔧 Filters")
price_min = st.sidebar.slider(
    "Minimum Transaction Amount ($)",
    min_value=0,
    max_value=int(df["Item_price"].max()),
    value=200,
    step=50,
)

sentiment_max = st.sidebar.slider(
    "Maximum Sentiment (more negative → lower)",
    min_value=-1.0,
    max_value=0.0,
    value=-0.4,
    step=0.05,
)

channel_opts = st.sidebar.multiselect(
    "Channel",
    options=df["channel_name"].dropna().unique(),
    default=df["channel_name"].dropna().unique().tolist(),
)

# ---- Apply filters ----------------------------------------------------
filtered = df[
    (df["Item_price"] > price_min)
    & (df["sentiment_score"] < sentiment_max)
    & (df["channel_name"].isin(channel_opts))
]

st.subheader(f"📊 {len(filtered)} flagged rows (out of {len(df)} total)")

# ---- Table ------------------------------------------------------------
cols_to_show = [
    "Unique id",
    "Order_id",
    "order_date_time",
    "Customer Remarks",
    "sentiment_score",
    "Item_price",
    "synthetic_amount",
    "synthetic_merchant",
    "channel_name",
    "CSAT Score",
]

st.dataframe(
    filtered[cols_to_show].reset_index(drop=True),
    height=400,
    use_container_width=True,
)

# ---- Quick metrics ----------------------------------------------------
st.subheader("💡 Quick Insights")
col1, col2, col3 = st.columns(3)

col1.metric("Avg. Item Price", f"${filtered['Item_price'].mean():,.0f}")
col2.metric("Avg. Synthetic Amount", f"${filtered['synthetic_amount'].mean():,.0f}")
col3.metric("Mean Sentiment", f"{filtered['sentiment_score'].mean():.2f}")

# ---- Scatter chart ----------------------------------------------------
st.subheader("📈 Amount vs. Sentiment")
chart_data = filtered[
    ["Item_price", "synthetic_amount", "sentiment_score"]
].rename(
    columns={
        "Item_price": "Real Amount",
        "synthetic_amount": "Synthetic Amount",
        "sentiment_score": "Sentiment",
    }
)
st.scatter_chart(chart_data)

# ---- Download button -------------------------------------------------
csv_bytes = filtered.to_csv(index=False).encode()
st.download_button(
    label="💾 Download filtered rows (CSV)",
    data=csv_bytes,
    file_name="flagged_filtered.csv",
    mime="text/csv",
)
