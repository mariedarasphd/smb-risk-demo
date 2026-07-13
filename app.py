# -------------------------------------------------
# app.py – Streamlit demo (pinned deps, ultra-safe)
# -------------------------------------------------
import sys
import streamlit as st

# Check basic imports first (catch segfault sources early)
try:
    import pandas as pd
except Exception as e:
    st.error(f"Failed to import pandas: {e}")
    sys.exit(1)

import pathlib

# ----------------------------------------------------------------------
# 0️⃣  Page configuration – must be the first Streamlit call
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SMB Risk Dashboard",
    layout="wide",
    page_icon="🔍"
)

# ----------------------------------------------------------------------
# 1️⃣  Custom CSS + logo (Tiffany blue background)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 0️⃣‑B  Show the logo (sidebar)
# ----------------------------------------------------------------------
logo_path = pathlib.Path(__file__).parent / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=120)

# ----------------------------------------------------------------------
# 2️⃣  Load the sample CSV (cached to avoid re‑reading on every interaction)
# ----------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Read `sample_flagged.csv` located next to this script."""
    data_path = pathlib.Path(__file__).parent / "sample_flagged.csv"

    if not data_path.is_file():
        raise FileNotFoundError(
            f"CSV not found at {data_path}. "
            "Make sure the file is named exactly 'sample_flagged.csv' "
            "and lives in the repository root."
        )

    df = pd.read_csv(
        data_path,
        engine="python",   # works on any CSV dialect
        encoding="utf-8",
    )

    return df.copy()


# Load once (cached)
try:
    df = load_data()
    if df.empty:
        st.error("CSV file is empty.")
        st.stop()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ----------------------------------------------------------------------
# 3️⃣  UI layout – filters, tables, metrics, chart, download
# ----------------------------------------------------------------------
st.title("🔎 SMB Customer‑Sentiment + Transaction Risk Dashboard")
st.markdown(
    """
    A lightweight demo that joins **customer remarks** with **synthetic transaction data**, 
    highlighting high‑value, negative‑sentiment cases.
    """
)

# ---- Sidebar filters -------------------------------------------------
st.sidebar.header("🔧 Filters")

# Safely get column values
price_col = "Item_price" if "Item_price" in df.columns else None
sentiment_col = "sentiment_score" if "sentiment_score" in df.columns else None
channel_col = "channel_name" if "channel_name" in df.columns else None

if price_col:
    price_min = st.sidebar.slider(
        "Minimum Transaction Amount ($)",
        min_value=0,
        max_value=int(df[price_col].max()) if not df[price_col].empty else 10000,
        value=200,
        step=50,
    )
else:
    price_min = 0
    st.warning("No 'Item_price' column found in CSV.")

if sentiment_col:
    sentiment_max = st.sidebar.slider(
        "Maximum Sentiment (more negative → lower)",
        min_value=float(df[sentiment_col].min()) if not df[sentiment_col].empty else -1.0,
        max_value=0.0,
        value=-0.4,
        step=0.05,
    )
else:
    sentiment_max = -0.4
    st.warning("No 'sentiment_score' column found in CSV.")

if channel_col:
    channel_opts = st.sidebar.multiselect(
        "Channel",
        options=df[channel_col].dropna().unique(),
        default=list(df[channel_col].dropna().unique()),
    )
else:
    channel_opts = []
    st.warning("No 'channel_name' column found in CSV.")

# ---- Apply filters ----------------------------------------------------
if all([price_col, sentiment_col, channel_col]):
    filtered = df[
        (df[price_col] > price_min)
        & (df[sentiment_col] < sentiment_max)
        & (df[channel_col].isin(channel_opts))
    ]
elif price_col and sentiment_col:
    filtered = df[
        (df[price_col] > price_min)
        & (df[sentiment_col] < sentiment_max)
    ]
else:
    filtered = df.copy()

st.subheader(f"📊 {len(filtered)} flagged rows (out of {len(df)} total)")

# ---- Table ------------------------------------------------------------
cols_to_show = [
    col for col in [
        "Unique id", "Order_id", "order_date_time", "Customer Remarks",
        "sentiment_score", "Item_price", "synthetic_amount",
        "synthetic_merchant", "channel_name", "CSAT Score",
    ] if col in df.columns
]

if not filtered.empty and cols_to_show:
    st.dataframe(
        filtered[cols_to_show].reset_index(drop=True),
        height=400,
    )

    # ---- Quick metrics ------------------------------------------------
    st.subheader("💡 Quick Insights")
    col1, col2, col3 = st.columns(3)

    if price_col:
        avg_price = filtered[price_col].mean()
        col1.metric("Avg. Item Price", f"${avg_price:,.0f}" if pd.notna(avg_price) else "N/A")
    
    if "synthetic_amount" in filtered.columns:
        avg_amount = filtered['synthetic_amount'].mean()
        col2.metric("Avg. Synthetic Amount", f"${avg_amount:,.0f}" if pd.notna(avg_amount) else "N/A")
    
    if sentiment_col:
        avg_sentiment = filtered[sentiment_col].mean()
        col3.metric("Mean Sentiment", f"{avg_sentiment:.2f}" if pd.notna(avg_sentiment) else "N/A")

    # ---- Scatter chart ------------------------------------------------
    if price_col and sentiment_col and "Item_price" in df.columns and "sentiment_score" in df.columns:
        st.subheader("📈 Amount vs. Sentiment")
        chart_data = filtered[
            ["Item_price", "synthetic_amount" if "synthetic_amount" in filtered.columns else "Item_price", "sentiment_score"]
        ].iloc[:, :3].rename(
            columns={
                "Item_price": "Real Amount",
                "synthetic_amount" if "synthetic_amount" in filtered.columns else "Item_price": "Synthetic Amount",
                "sentiment_score": "Sentiment",
            }
        ).dropna(axis=1, how='all')
        
        if not chart_data.empty:
            st.scatter_chart(chart_data)

    # ---- Download button ---------------------------------------------
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        label="💾 Download filtered rows (CSV)",
        data=csv_bytes,
        file_name="flagged_filtered.csv",
        mime="text/csv",
    )
else:
    st.warning("No rows match the current filters or data is unavailable.")
