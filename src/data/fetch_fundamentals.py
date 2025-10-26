# src/data/fetch_fundamentals.py

import os
import requests
import pandas as pd
from utils.config import CONFIG


def fetch_fundamentals(symbol: str, api_key: str, base_url: str, quarters: int = 8):
    """Fetch quarterly key metrics for a given stock symbol."""
    url = f"{base_url}/key-metrics/{symbol}?period=quarter&limit={quarters}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
    data = response.json()
    df = pd.DataFrame(data)
    return df


def select_metrics(df: pd.DataFrame, metrics: list):
    """Select only the desired metrics and clean data."""
    cols = ["date"] + [m for m in metrics if m in df.columns]
    df = df[cols]
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_fundamentals(df: pd.DataFrame, path: str, symbol: str):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{symbol}_fundamentals.csv")
    df.to_csv(file_path, index=False)
    print(f"[INFO] Fundamentals saved to {file_path}")


if __name__ == "__main__":
    symbol = CONFIG["stock_symbol"]
    cfg = CONFIG["fundamentals"]

    print(f"[INFO] Fetching fundamentals for {symbol}...")
    df = fetch_fundamentals(symbol, cfg["api_key"], cfg["base_url"], cfg["quarters"])
    df = select_metrics(df, cfg["metrics"])

    save_fundamentals(df, CONFIG["save_paths"]["raw_data"], symbol)
    print("[SUCCESS] Fundamentals fetched and saved ✅")
