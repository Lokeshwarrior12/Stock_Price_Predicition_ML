# app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from model.train_hybrid_model import HybridModel
from genai.market_explainer import generate_prompt, generate_analysis
from utils.config import CONFIG


# -----------------------
# Load Model and Data
# -----------------------

@st.cache_resource
def load_model(model_path, input_dim_seq, input_dim_fund):
    cfg = CONFIG["hybrid_model"]
    model = HybridModel(input_dim_seq, input_dim_fund,
                        hidden_dim=cfg["hidden_dim"],
                        num_layers=cfg["num_layers"],
                        dropout=cfg["dropout"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_data(symbol):
    df = pd.read_csv(f"data/processed/{symbol}_merged_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)
    return df


# -----------------------
# Plotting Utilities
# -----------------------

def plot_predictions(df, preds, lookback_days=180):
    df_recent = df.tail(lookback_days)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_recent["Date"], y=df_recent["Close"], mode="lines",
                             name="Actual Price", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=df_recent["Date"].iloc[-len(preds):], y=preds,
                             mode="lines", name="Predicted Price", line=dict(color="orange")))
    fig.update_layout(template="plotly_dark", height=600,
                      title="Predicted vs Actual Price Trend")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------
# Streamlit App Layout
# -----------------------

def main():
    st.set_page_config(page_title="AI Stock Predictor", layout="wide")
    st.title("📊 AI-Powered Stock Prediction Dashboard")

    symbol = CONFIG["stock_symbol"]

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 1, 20, 5)
    lookback_window = st.sidebar.slider("Lookback Window", 30, 120, 60)

    st.sidebar.markdown("---")
    st.sidebar.write("Using Model:")
    st.sidebar.code("HybridModel (Technical + Fundamentals)")

    # Load data
    df = load_data(symbol)
    st.subheader(f"Stock: {symbol}")

    # Prepare features
    tech_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "EMA_short", "EMA_long", "MACD", "ATR"]
    fund_cols = [c for c in df.columns if c not in tech_cols + ["Date", "Target"] and df[c].dtype != 'O']

    input_dim_seq = len(tech_cols)
    input_dim_fund = len(fund_cols)

    model = load_model(CONFIG["hybrid_model"]["model_save_path"], input_dim_seq, input_dim_fund)

    # Prepare tensors
    df_scaled = df.copy()
    X_seq = torch.tensor(df_scaled[tech_cols].tail(lookback_window).values, dtype=torch.float32).unsqueeze(0)
    X_fund = torch.tensor(df_scaled[fund_cols].iloc[-1].values, dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        pred_price = model(X_seq, X_fund).item()
    current_price = df["Close"].iloc[-1]

    delta = pred_price - current_price
    direction = "📈 Up" if delta > 0 else "📉 Down"
    color = "green" if delta > 0 else "red"

    st.metric(label="Predicted Movement", value=f"${pred_price:.2f}", delta=f"{delta:.2f}", delta_color=color)
    st.markdown(f"**Prediction Direction:** {direction}")

    # Chart
    preds = np.linspace(current_price, pred_price, num=lookback_window)
    plot_predictions(df, preds)

    # GenAI Analysis
    st.subheader("🤖 AI Market Analysis")

    indicators = {
        "RSI": round(df["RSI"].iloc[-1], 2),
        "MACD": round(df["MACD"].iloc[-1], 2),
        "EMA Trend": "Short EMA above Long EMA" if df["EMA_short"].iloc[-1] > df["EMA_long"].iloc[-1] else "Short EMA below Long EMA",
        "ATR": round(df["ATR"].iloc[-1], 2)
    }

    fundamentals = {k: round(df[k].iloc[-1], 2) for k in fund_cols[:5]}  # summarize top 5 metrics

    if st.button("🧠 Generate AI Explanation"):
        with st.spinner("Analyzing with GenAI..."):
            prompt = generate_prompt(symbol, current_price, pred_price, indicators, fundamentals)
            analysis = generate_analysis(prompt)
            st.success("Analysis Ready!")
            st.markdown(f"**{analysis}**")

    st.markdown("---")
    st.caption("Built by Lokeshwar • Powered by PyTorch, Streamlit, and GPT")

if __name__ == "__main__":
    main()
