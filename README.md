# Stock_Price_Predicition_ML
# 🧠 AI Stock Price Prediction Dashboard

This project predicts future stock prices using a **hybrid deep learning model** that combines **technical indicators** and **fundamental data**.  
The model is served with a **Streamlit dashboard** for interactive visualization and AI-powered analysis.

## 🚀 Features

- 📊 Predict short-term stock price movement (e.g., AAPL)
- ⚙️ Combines LSTM-based technical analysis + fundamentals
- 🧮 Interactive charts with Plotly
- 🤖 AI market explanation powered by LLMs
- 📈 Customizable lookback and forecast horizons


## 🧩 Installation

### 1️⃣ Clone or download the repository

bash
git clone https://github.com/<yourusername>/Stock-Price_Prediction.git
cd Stock-Price_Prediction

pip install -r requirements.txt

## Run the App
🖥️ Option 1 (Recommended — Python 3.13)
py -3.13 -m streamlit run app.py

## ⚙️ Configuration
Edit the config file at:
src/utils/config.py

Typical content:
CONFIG = {
    "stock_symbol": "AAPL",
    "hybrid_model": {
        "model_save_path": "models/hybrid_model.pth",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3
    }
}


## 🧾 Data Requirements
Place your processed CSV under:
data/processed/{SYMBOL}_merged_features.csv

### Example:
data/processed/AAPL_merged_features.csv

Each CSV should contain columns like:
Date, Open, High, Low, Close, Volume, RSI, EMA_short, EMA_long, MACD, MACD_signal, BB_high, ATR, ...


## 🤝 Contributing
Pull requests are welcome!
If you find a bug or want to suggest a feature, open an issue on GitHub.

## 🧑‍💻 Author
Lokeshwar
Built with ❤️ using PyTorch, Streamlit, and GPT-based analytics.
