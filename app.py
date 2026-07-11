# -------------------------------------------------
# app.py – Streamlit demo (lightweight, no NLTK)
# -------------------------------------------------
import streamlit as st
import pandas as pd
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

    # -------------------------------------------------
    # Convert any *existing* date columns to datetime
    # -------------------------------------------------
    expected_date_cols = ["order_date_time", "synthetic_date", "Survey_response_Date"]
    existing_date_cols = [c for c in expected_date_cols if c in df.columns]

    if existing_date_cols:
        try:
            df[existing_date_cols] = df[existing_date_cols].apply(
                lambda col: pd.to_datetime(col, dayfirst=False, errors="coerce")
            )
        except Exception as exc:
            st.warning(
                f"Could not parse dates for columns {existing_date_cols}: {exc}. "
                "They will remain as strings."
            )

    return df.copy()


# Load once (cached)
try:
    df = load_data()
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
price_min = st.sidebar.slider(
    "Minimum Transaction Amount ($)",
    min_value=0,
    max_value=int(df["Item_price"].max()) if not df.empty else 10000,
    value=200,
    step=50,
)

sentiment_max = st.sidebar.slider(
    "Maximum Sentiment (more negative → lower)",
    min_value=float(df["sentiment_score"].min()) if not df.empty and "sentiment_score" in df.columns else -1.0,
    max_value=0.0,
    value=-0.4,
    step=0.05,
)

channel_opts = st.sidebar.multiselect(
    "Channel",
    options=df["channel_name"].dropna().unique() if not df.empty and "channel_name" in df.columns else [],
    default=list(df["channel_name"].dropna().unique()) if not df.empty and "channel_name" in df.columns else [],
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

if not filtered.empty:
    st.dataframe(
        filtered[cols_to_show].reset_index(drop=True),
        height=400,
    )

    # ---- Quick metrics ------------------------------------------------
    st.subheader("💡 Quick Insights")
    col1, col2, col3 = st.columns(3)

    avg_price = filtered['Item_price'].mean()
    avg_amount = filtered['synthetic_amount'].mean()
    avg_sentiment = filtered['sentiment_score'].mean()

    col1.metric("Avg. Item Price", f"${avg_price:,.0f}" if pd.notna(avg_price) else "N/A")
    col2.metric("Avg. Synthetic Amount", f"${avg_amount:,.0f}" if pd.notna(avg_amount) else "N/A")
    col3.metric("Mean Sentiment", f"{avg_sentiment:.2f}" if pd.notna(avg_sentiment) else "N/A")

    # ---- Scatter chart ------------------------------------------------
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

    # ---- Download button ---------------------------------------------
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        label="💾 Download filtered rows (CSV)",
        data=csv_bytes,
        file_name="flagged_filtered.csv",
        mime="text/csv",
    )
else:
    st.warning("No rows match the current filters. Try adjusting the sliders or channel selection.")
