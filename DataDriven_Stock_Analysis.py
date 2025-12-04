import os
import argparse
import yaml
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------ CONFIG ------------
DATA_YAML_FOLDER = "dataset_yaml"
CSV_OUTPUT_FOLDER = "csv_output"
PLOTS_FOLDER = "plots"
SECTOR_CSV = "sector_map.csv"
# --------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_folders():
    os.makedirs(CSV_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(CSV_OUTPUT_FOLDER, "monthly_top5"), exist_ok=True)


# ---------- YAML READER ----------
def read_yaml_files(root_folder: str) -> dict:
    symbol_data = {}

    if not os.path.isdir(root_folder):
        logging.warning("YAML folder not found.")
        return {}

    for root, _, files in os.walk(root_folder):
        for f in files:
            if not f.endswith((".yaml", ".yml")):
                continue
            fpath = os.path.join(root, f)

            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
            except Exception as e:
                logging.warning(f"Bad YAML {f}: {e}")
                continue

            if not isinstance(data, list):
                logging.warning(f"YAML is not list: {f}")
                continue

            for rec in data:
                if not isinstance(rec, dict):
                    continue

                symbol = rec.get("Ticker") or rec.get("Symbol")
                if not symbol:
                    symbol = os.path.splitext(f)[0]

                entry = {
                    "Symbol": symbol,
                    "Date": rec.get("date"),
                    "Open": rec.get("open"),
                    "High": rec.get("high"),
                    "Low": rec.get("low"),
                    "Close": rec.get("close"),
                    "Volume": rec.get("volume"),
                }

                symbol_data.setdefault(symbol, []).append(entry)

    logging.info(f"YAML extraction complete. Symbols found: {len(symbol_data)}")
    return symbol_data


# ---------- NORMALIZATION ----------
def normalize_record_fields(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()

    if "Date" not in df.columns:
        raise KeyError(f"'Date' column missing. Columns: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------- CSV SAVE ----------
def save_symbol_csvs(symbol_data: dict):
    ensure_folders()
    for symbol, recs in symbol_data.items():
        df = pd.DataFrame(recs)
        if df.empty:
            continue
        try:
            df = normalize_record_fields(df)
        except Exception as e:
            logging.warning(f"Skipping {symbol} due to error: {e}")
            continue
        out = os.path.join(CSV_OUTPUT_FOLDER, f"{symbol}.csv")
        df.to_csv(out, index=False)
    logging.info("CSV saved.")


# ---------- LOAD CSV ----------
def load_all_csvs(folder: str = CSV_OUTPUT_FOLDER) -> dict:
    data = {}
    if not os.path.isdir(folder):
        logging.warning(f"CSV folder not found: {folder}")
        return data

    for f in os.listdir(folder):
        if not f.endswith(".csv"):
            continue
        symbol = f.replace(".csv", "")
        fpath = os.path.join(folder, f)
        df = pd.read_csv(fpath)
        df.columns = df.columns.astype(str).str.strip()
        if "Date" not in df.columns:
            logging.warning(f"File {f} missing Date column, skipping.")
            continue
        try:
            df = normalize_record_fields(df)
        except Exception as e:
            logging.warning(f"Error normalizing {f}: {e}")
            continue
        df["Symbol"] = symbol
        data[symbol] = df

    logging.info("Loaded CSVs.")
    return data


# ---------- METRICS ----------
def compute_metrics(all_data_dict: dict) -> pd.DataFrame:
    rows = []
    for sym, df in all_data_dict.items():
        if df.shape[0] < 2:
            continue

        df = df.sort_values("Date").reset_index(drop=True)
        df["Daily_Return"] = df["Close"].pct_change()

        first_close = df["Close"].iloc[0]
        last_close = df["Close"].iloc[-1]
        if pd.isna(first_close) or pd.isna(last_close) or first_close == 0:
            continue

        yearly_ret = (last_close - first_close) / first_close

        rows.append(
            {
                "Symbol": sym,
                "Yearly_Return": yearly_ret,
                "Avg_Price": df["Close"].mean(),
                "Avg_Volume": df["Volume"].mean(),
                "Volatility": df["Daily_Return"].std(),
                "Last_Close": last_close,
                "First_Date": df["Date"].iloc[0],
                "Last_Date": df["Date"].iloc[-1],
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        logging.warning("No metrics computed.")
        return metrics

    metrics["Is_Green"] = metrics["Yearly_Return"] > 0
    return metrics.sort_values("Yearly_Return", ascending=False)


# ---------- MONTHLY TOP5 ----------
def monthly_top5(all_data_dict: dict) -> dict:
    lst = []
    for sym, df in all_data_dict.items():
        df2 = df.copy()
        df2["Symbol"] = sym
        lst.append(df2)

    if not lst:
        return {}

    df_all = pd.concat(lst, ignore_index=True)
    df_all["YM"] = df_all["Date"].dt.to_period("M")

    results = {}
    for ym, gp in df_all.groupby("YM"):
        recs = []
        for sym, g2 in gp.groupby("Symbol"):
            g2 = g2.sort_values("Date")
            if g2.shape[0] < 2:
                continue
            first_close = g2["Close"].iloc[0]
            last_close = g2["Close"].iloc[-1]
            if pd.isna(first_close) or pd.isna(last_close) or first_close == 0:
                continue
            ret = (last_close - first_close) / first_close
            recs.append((sym, ret))

        if not recs:
            continue

        dfm = pd.DataFrame(recs, columns=["Symbol", "Monthly_Return"])
        results[str(ym)] = {
            "top5": dfm.sort_values("Monthly_Return", ascending=False).head(5),
            "bottom5": dfm.sort_values("Monthly_Return").head(5),
        }

    return results


# ---------- PLOTS ----------
def plot_volatility(summary: pd.DataFrame):
    if summary.empty:
        logging.warning("Summary empty, skip volatility plot.")
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=summary.sort_values("Volatility", ascending=False).head(10),
        x="Symbol",
        y="Volatility",
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, "volatility_top10.png"))
    plt.close()


# ---------- MAIN PIPELINE ----------
def run_pipeline(extract_yaml: bool = True, do_plots: bool = True):
    ensure_folders()

    if extract_yaml:
        data = read_yaml_files(DATA_YAML_FOLDER)
        save_symbol_csvs(data)

    all_data = load_all_csvs()
    if not all_data:
        logging.error("No CSV data loaded; check dataset_yaml / csv_output.")
        return

    summary = compute_metrics(all_data)

    if not summary.empty:
        total = summary.shape[0]
        green = summary["Is_Green"].sum()
        red = total - green
        avg_price = summary["Avg_Price"].mean()
        avg_volume = summary["Avg_Volume"].mean()
        logging.info(f"Total stocks: {total}, Green: {green}, Red: {red}")
        logging.info(f"Average price: {avg_price:.2f}, Average volume: {avg_volume:.0f}")

    summary_path = os.path.join(CSV_OUTPUT_FOLDER, "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)
    logging.info(f"Summary metrics saved to {summary_path}")

    if do_plots:
        plot_volatility(summary)

    mres = monthly_top5(all_data)
    monthly_folder = os.path.join(CSV_OUTPUT_FOLDER, "monthly_top5")
    os.makedirs(monthly_folder, exist_ok=True)
    for m, rec in mres.items():
        rec["top5"].to_csv(os.path.join(monthly_folder, f"{m}_top5.csv"), index=False)
        rec["bottom5"].to_csv(os.path.join(monthly_folder, f"{m}_bottom5.csv"), index=False)

    print("\nPIPELINE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        extract_yaml=not args.no_extract,
        do_plots=not args.no_plots,
    )
