import streamlit as st
import pandas as pd
import pathlib
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

# Path to the logo file (must be in the repo root)
logo_path = pathlib.Path(__file__).parent / "logo.png"

# Show logo in the sidebar (you can also use st.image for main‑page placement)
st.sidebar.image(str(logo_path), width=120)

# -------------------------------------------------
# 1️⃣  Load the sample CSV (cached for speed)
# -------------------------------------------------
@st.cache_data(ttl=86400)   # cache for 24 h
def load_data():
    data_path = pathlib.Path(__file__).parent / "sample_flagged.csv"
    df = pd.read_csv(
        data_path,
        parse_dates=["order_date_time",
                     "synthetic_date",
                     "Survey_response_Date"]
    )
    return df

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
st.markdown("""
A lightweight demo that joins **customer remarks** with **synthetic transaction data**, runs **sentiment analysis**, and highlights high‑value, negative‑sentiment cases.
""")

# ---- Sidebar filters -------------------------------------------------
st.sidebar.header("🔧 Filters")
price_min = st.sidebar.slider(
    "Minimum Transaction Amount ($)",
    min_value=0,
    max_value=int(df["Item_price"].max()),
    value=200,
    step=50
)

sentiment_max = st.sidebar.slider(
    "Maximum Sentiment (more negative → lower)",
    min_value=-1.0,
    max_value=0.0,
    value=-0.4,
    step=0.05
)

channel_opts = st.sidebar.multiselect(
    "Channel",
    options=df["channel_name"].dropna().unique(),
    default=df["channel_name"].dropna().unique().tolist()
)

# ---- Apply filters ----------------------------------------------------
filtered = df[
    (df["Item_price"] > price_min) &
    (df["sentiment_score"] < sentiment_max) &
    (df["channel_name"].isin(channel_opts))
]

st.subheader(f"📊 {len(filtered)} flagged rows (out of {len(df)} total)")

# ---- Table ------------------------------------------------------------
cols_to_show = [
    "Unique id", "Order_id", "order_date_time",
    "Customer Remarks", "sentiment_score", "Item_price",
    "synthetic_amount", "synthetic_merchant",
    "channel_name", "CSAT Score"
]

st.dataframe(
    filtered[cols_to_show].reset_index(drop=True),
    height=400,
    use_container_width=True
)

# ---- Quick metrics ----------------------------------------------------
st.subheader("💡 Quick Insights")
col1, col2, col3 = st.columns(3)

col1.metric("Avg. Item Price", f"${filtered['Item_price'].mean():,.0f}")
col2.metric("Avg. Synthetic Amount", f"${filtered['synthetic_amount'].mean():,.0f}")
col3.metric("Mean Sentiment", f"{filtered['sentiment_score'].mean():.2f}")

# ---- Scatter chart ----------------------------------------------------
st.subheader("📈 Amount vs. Sentiment")
chart_data = filtered[["Item_price", "synthetic_amount", "sentiment_score"]]
st.scatter_chart(
    chart_data.rename(columns={
        "Item_price": "Real Amount",
        "synthetic_amount": "Synthetic Amount",
        "sentiment_score": "Sentiment"
    })
)

# ---- Download button -------------------------------------------------
csv_bytes = filtered.to_csv(index=False).encode()
st.download_button(
    label="💾 Download filtered rows (CSV)",
    data=csv_bytes,
    file_name="flagged_filtered.csv",
    mime="text/csv"
)
