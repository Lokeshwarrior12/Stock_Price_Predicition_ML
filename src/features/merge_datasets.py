# src/features/merge_datasets.py

import pandas as pd
import os
from utils.config import CONFIG


def merge_datasets(symbol: str):
    tech_path = f"{CONFIG['save_paths']['raw_data']}{symbol}_technical_data.csv"
    fund_path = f"{CONFIG['save_paths']['raw_data']}{symbol}_fundamentals.csv"

    df_tech = pd.read_csv(tech_path)
    df_fund = pd.read_csv(fund_path)

    df_tech["Date"] = pd.to_datetime(df_tech["Date"])
    df_fund["date"] = pd.to_datetime(df_fund["date"])

    # Merge fundamentals into technical data by nearest date (quarterly -> daily)
    df_merged = pd.merge_asof(
        df_tech.sort_values("Date"),
        df_fund.sort_values("date"),
        left_on="Date", right_on="date",
        direction="backward"
    )

    df_merged.drop(columns=["date"], inplace=True)
    df_merged.fillna(method="ffill", inplace=True)

    save_path = f"{CONFIG['save_paths']['processed_data']}{symbol}_merged_features.csv"
    os.makedirs(CONFIG["save_paths"]["processed_data"], exist_ok=True)
    df_merged.to_csv(save_path, index=False)
    print(f"[SUCCESS] Merged dataset saved to {save_path}")

    return df_merged


if __name__ == "__main__":
    symbol = CONFIG["stock_symbol"]
    merge_datasets(symbol)
