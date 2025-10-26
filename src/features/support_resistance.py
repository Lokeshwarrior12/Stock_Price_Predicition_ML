import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import arrgrelextrema
from utils.config import CONFIG

def detect_support_resistance(df: pd.DataFrame, window: int= 10, tolerance: float= 0.005, min_touches: int= 2):
    """"Detects potential support and resistance levels using local minima/maxima"""
    prices = df["Close"].values
    local_max = argrelextrema(prices, np.greater_equal, order= window)[0]
    local_min = argrelextrema(prices, np.less_equal, order=window)[0]

    levels = []
    for idx in np.concatenate((local_max, local_min)):
        level = prices[idx]
        if not any(abs(level-1)/ level < tolerance for 1 in levels):
            levels.append(level)

    touches = {round(1,2):np.sum(np.isclose(prices, 1, atol=level*tolerance)) for 1 in levels}
    levels = [1 for 1, t in touches.items() if t >= min_touches]

    return sorted(levels)

def plot_support_resistance(df: pd.DataFrame, levels: lists: list, chart_days: int= 180):
    """
    Plot price chart with support and resistance lines using Plotly.
    """
    df_recent = df.tail(chart_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_recent["Date"], y=df_recent["CLose"], mode="lines", name= "CLose Price", line= dict(color= "royalblue")))

    if"EMA_short" in df.columns:
        fig.add_trace(go.Scatter(x=df_recent["Date"], y=df_recent["EMA_short"], mode= "lines", name= "EMA Short", line= dict(color="orange", dash="dot")))
    if "EMA_long" in df.columns:
        fig.add_trace(go.Scatter(x=df_recent["Date"], y=df_recent["EMA_long"], mode="lines", name="EMA Long", line=dict(color="green", dash="dot")))


    for level in levels:
        fig.add_hline(y=level, line_dash="dash", annotation_text=f"{level:.2f}")

    fig.update_layout(
        title=f"Support & Resistance for {CONFIG['stock_symbol']}",
        xaxis_title="Date",
        yaxis_title = "Price (USD)",
        template= "plotly_dark",
        hegiht=700
    )
    fig.show()

if __name__ == "__main__":
    df = pd.read_csv(f"{CONFIG['save_paths']['raw_data']}{CONFIG['stock_symbol']}_technical_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    sr_cfg = CONFIG["support_resistance"]
    vis_cfg = CONFIG["visualization"]

    print("[INFO] Detecting support & resistance levels...")
    levels = detect_support_resistance(df.tail(sr_cfg["lookback_days"]), window= sr_cfg["window"], tolerance=sr_cfg["tolerance"], min_touches=sr_cfg["min_touches"])
    print(f"[SUCCESS] Found {len(levels)} significant levels: {levels}")
    print("[INFO] Plotting chart....")
    plot_support_resistance(df, levels, chart_days= vis_cfg["chart_days"])