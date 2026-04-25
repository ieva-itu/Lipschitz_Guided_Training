
import pandas as pd
from pathlib import Path

'''
def load_single_series(csv_path: str,
                       item_col: str = "StockCode",
                       item_id: str = "85123A",
                       date_col: str = "InvoiceDate",
                       sales_col: str = "Quantity"):
    """
    Loads the Online Retail II cleaned CSV and returns a daily demand series
    for one StockCode item.

    Produces output:
        index: daily dates (D frequency)
        column: 'y' (sales)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col])

    # Filter for selected item
    df_item = df[df[item_col] == item_id]

    # Aggregate quantity by date
    df_daily = df_item.groupby(date_col)[sales_col].sum().sort_index()

    # Ensure daily frequency; missing days → 0 demand
    df_daily = df_daily.asfreq("D").fillna(0.0)

    # Rename into expected time-series format
    series = df_daily.to_frame(name="y")
    series.index.name = "date"

    return series
'''

import pandas as pd


def load_single_series(
    csv_path: str,
    item_col: str = "StockCode",
    item_id: str = "sku_1",
    qty_col: str = "Quantity",
    date_col: str = "InvoiceDate",
):
    """
    Load a single SKU time series from the UCI Online Retail-style CSV.

    Expected columns in CSV:
      - item_col (e.g. 'StockCode')
      - qty_col  (e.g. 'Quantity')
      - date_col (e.g. 'InvoiceDate')

    Returns a DataFrame with columns:
      - 'date' : pandas.Timestamp (daily)
      - 'qty'  : aggregated demand per day (float)
    """

    df = pd.read_csv(csv_path)

    # Filter to the chosen SKU; if item_id not present, fall back to most frequent
    if item_id not in df[item_col].astype(str).unique():
        most_freq = (
            df[item_col].astype(str)
            .value_counts()
            .idxmax()
        )
        print(
            f"[load_single_series] item_id={item_id} not found; "
            f"falling back to most frequent {item_col}={most_freq}"
        )
        item_id = most_freq

    sku_df = df[df[item_col].astype(str) == str(item_id)].copy()

    # Parse datetime
    sku_df[date_col] = pd.to_datetime(sku_df[date_col])

    # Aggregate to daily demand
    daily = (
        sku_df
        .groupby(sku_df[date_col].dt.date)[qty_col]
        .sum()
        .reset_index()
        .rename(columns={date_col: "date", qty_col: "qty"})
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


def train_val_split(series, val_ratio=0.2):
    n = len(series)
    n_val = int(n * val_ratio)
    return series.iloc[:-n_val], series.iloc[-n_val:]


if __name__ == "__main__":
    s = load_single_series(
        "data/raw/retail_demand.csv",
        item_col="StockCode",
        item_id="85123A",         # ← CHANGE THIS to your chosen top-selling StockCode
        date_col="InvoiceDate",
        sales_col="Quantity",
    )
    print(s.head())
    print(s.tail())

