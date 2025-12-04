import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

CSV_FOLDER = "csv_output"
SUMMARY_FILE = os.path.join(CSV_FOLDER, "summary_metrics.csv")
MONTHLY_FOLDER = os.path.join(CSV_FOLDER, "monthly_top5")
SECTOR_FILE = "sector_map.csv"

st.set_page_config(page_title="Nifty 50 Stock Dashboard", layout="wide")

# ---------------- LOADERS ----------------
@st.cache_data
def load_symbol(symbol: str) -> pd.DataFrame:
    fpath = os.path.join(CSV_FOLDER, f"{symbol}.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df.columns = df.columns.astype(str).str.strip()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date")
        return df
    return pd.DataFrame()


@st.cache_data
def load_summary() -> pd.DataFrame:
    if os.path.exists(SUMMARY_FILE):
        df = pd.read_csv(SUMMARY_FILE)
        df.columns = df.columns.astype(str).str.strip()
        return df
    return pd.DataFrame()


@st.cache_data
def load_monthly_tables() -> dict:
    if not os.path.isdir(MONTHLY_FOLDER):
        return {}
    out = {}
    for f in os.listdir(MONTHLY_FOLDER):
        if f.endswith(".csv"):
            ym = f.replace(".csv", "")
            df = pd.read_csv(os.path.join(MONTHLY_FOLDER, f))
            out[ym] = df
    return out


@st.cache_data
def load_sector_map() -> pd.DataFrame:
    if os.path.exists(SECTOR_FILE):
        df = pd.read_csv(SECTOR_FILE)
        df.columns = df.columns.astype(str).str.strip()
        return df
    return pd.DataFrame()


# ---------------- DATA LOAD ----------------
summary = load_summary()
monthly_tables = load_monthly_tables()
sector_map = load_sector_map()

st.title("📊 Nifty 50 Stock Performance Dashboard")

if summary.empty:
    st.error("Summary metrics not found. Run: python data_pipeline.py")
    st.stop()

# ---------------- SAFE COLUMN HANDLING ----------------
required_cols = ["Symbol", "Yearly_Return", "Volatility", "Avg_Price", "Avg_Volume"]
for col in required_cols:
    if col not in summary.columns:
        if col == "Yearly_Return" and "Close_First" in summary.columns and "Close_Last" in summary.columns:
            summary["Yearly_Return"] = (summary["Close_Last"] - summary["Close_First"]) / summary["Close_First"]
        else:
            summary[col] = 0.0

# Normalize Symbol
summary["Symbol"] = summary["Symbol"].astype(str).str.strip().str.upper()

# ---------------- MARKET SUMMARY ----------------
total_stocks = summary.shape[0]
green_stocks = int((summary["Yearly_Return"] > 0).sum())
red_stocks = total_stocks - green_stocks
avg_price_all = summary["Avg_Price"].mean()
avg_vol_all = summary["Avg_Volume"].mean()

st.subheader("📌 Market Overview")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total Stocks", total_stocks)
col_b.metric("Green Stocks", green_stocks)
col_c.metric("Red Stocks", red_stocks)
col_d.metric("Avg Price (All)", f"{avg_price_all:.2f}")

# ---------------- SIDEBAR ----------------
symbols = sorted(summary["Symbol"].dropna().unique())
symbol = st.sidebar.selectbox("Choose Symbol", symbols)

df = load_symbol(symbol)

st.sidebar.write("### Stock Metrics")
if symbol in summary["Symbol"].values:
    row = summary[summary["Symbol"] == symbol].iloc[0]
    st.sidebar.metric("Yearly Return", f"{row['Yearly_Return'] * 100:.2f}%")
    st.sidebar.metric("Volatility", f"{row['Volatility']:.4f}")
    st.sidebar.metric("Avg Price", f"{row['Avg_Price']:.2f}")
    st.sidebar.metric("Avg Volume", f"{row['Avg_Volume']:.0f}")

# ---------------- MAIN DISPLAY: Price & Volume ----------------
st.header(f"📈 {symbol} — Price and Volume Trends")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Close Price Over Time")
    if not df.empty and "Close" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Close"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        st.pyplot(fig)
    else:
        st.info("No Close Price data for this symbol.")

with col2:
    st.subheader("Volume Over Time")
    if not df.empty and "Volume" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Volume"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume")
        st.pyplot(fig)
    else:
        st.info("No Volume data for this symbol.")

# ---------------- DAILY RETURNS & CUMULATIVE RETURN ----------------
st.subheader("📉 Daily and Cumulative Returns")
if not df.empty and "Close" in df.columns:
    df = df.sort_values("Date")
    df["Daily_Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    c1, c2 = st.columns(2)
    with c1:
        st.write("Daily Returns")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Daily_Return"])
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_ylabel("Daily Return")
        st.pyplot(fig)
    with c2:
        st.write("Cumulative Return")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Cumulative_Return"])
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_ylabel("Cumulative Return")
        st.pyplot(fig)
else:
    st.info("Cannot compute returns for this symbol.")

# ---------------- TOP 10 GAINERS / LOSERS ----------------
st.subheader("🏅 Top 10 Gainers and Losers (Yearly)")
top10 = summary.sort_values("Yearly_Return", ascending=False).head(10)
bottom10 = summary.sort_values("Yearly_Return").head(10)

c_top, c_bottom = st.columns(2)
with c_top:
    st.write("Top 10 Green Stocks")
    st.dataframe(top10[["Symbol", "Yearly_Return", "Avg_Price", "Avg_Volume"]], use_container_width=True)
with c_bottom:
    st.write("Top 10 Loss Stocks")
    st.dataframe(bottom10[["Symbol", "Yearly_Return", "Avg_Price", "Avg_Volume"]], use_container_width=True)

# ---------------- VOLATILITY TOP 10 ----------------
st.subheader("⚡ Top 10 Most Volatile Stocks")
vol_top = summary.sort_values("Volatility", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=vol_top, x="Symbol", y="Volatility", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# ---------------- CUMULATIVE RETURN: TOP 5 STOCKS ----------------
st.subheader("📈 Cumulative Return: Top 5 Stocks")
top5_syms = summary.sort_values("Yearly_Return", ascending=False).head(5)["Symbol"].tolist()
fig, ax = plt.subplots(figsize=(10, 4))

for sym in top5_syms:
    df_sym = load_symbol(sym)
    if df_sym.empty or "Close" not in df_sym.columns:
        continue
    df_sym = df_sym.sort_values("Date")
    df_sym["Daily_Return"] = df_sym["Close"].pct_change()
    df_sym["Cumulative_Return"] = (1 + df_sym["Daily_Return"]).cumprod() - 1
    ax.plot(df_sym["Date"], df_sym["Cumulative_Return"], label=sym)

ax.set_ylabel("Cumulative Return")
ax.axhline(0, color="gray", linestyle="--")
ax.legend()
st.pyplot(fig)

# ---------------- SECTOR-WISE PERFORMANCE ----------------
st.subheader("🏭 Sector-wise Performance")
if not sector_map.empty and "Symbol" in sector_map.columns and "Sector" in sector_map.columns:
    sector_map["Symbol"] = sector_map["Symbol"].astype(str).str.split(": ").str[-1].str.strip().str.upper()
    sector_summary = pd.merge(summary, sector_map[["Symbol", "Sector"]], on="Symbol", how="left")
    sector_summary["Sector"] = sector_summary["Sector"].fillna("Unknown")

    sector_metrics = sector_summary.groupby("Sector").agg(
        Avg_Yearly_Return=("Yearly_Return", "mean"),
        Avg_Volatility=("Volatility", "mean"),
        Count_Stocks=("Symbol", "count")
    ).reset_index().sort_values("Avg_Yearly_Return", ascending=False)

    st.write("Sector Performance Metrics")
    st.dataframe(sector_metrics, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sector_metrics, x="Sector", y="Avg_Yearly_Return", palette="viridis", ax=ax)
    ax.set_ylabel("Average Yearly Return")
    ax.set_xlabel("Sector")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)
else:
    st.info("Sector map missing or incomplete.")

# ---------------- CORRELATION HEATMAP ----------------
st.subheader("🔗 Stock Price Correlation Heatmap")
symbol_frames = []
for sym in symbols:
    df_sym = load_symbol(sym)
    if df_sym.empty or "Close" not in df_sym.columns:
        continue
    df_sym = df_sym[["Date", "Close"]].copy()
    df_sym = df_sym.rename(columns={"Close": sym})
    symbol_frames.append(df_sym)

if symbol_frames:
    merged_close = symbol_frames[0]
    for df_sym in symbol_frames[1:]:
        merged_close = pd.merge(merged_close, df_sym, on="Date", how="outer")
    merged_close = merged_close.set_index("Date").sort_index()
    corr = merged_close.pct_change().corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
else:
    st.info("Not enough data to compute correlation.")

# ---------------- MONTHLY TOP 5 GAINERS / LOSERS ----------------
st.subheader("📅 Monthly Top 5 Gainers and Losers")
available_months = sorted(monthly_tables.keys())
if available_months:
    selected_month = st.selectbox("Select Month", available_months)
    dfm = monthly_tables[selected_month]
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("Top 5 Gainers / Losers Table")
        st.dataframe(dfm, use_container_width=True)
else:
    st.info("No monthly CSVs found. Ensure pipeline created monthly_top5 files.")

# ---------------- FOOTER ----------------
st.write("---")
st.write("Built with ❤️ using Python, Pandas, SQL, Power BI, and Streamlit")
