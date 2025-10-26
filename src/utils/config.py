CONFIG = {
    "stock_symbol": "AAPL",
    "data":{
        "start_date": "2018-01-01", 
        "end_data": "2025-10-01",
        "interval": "1d",
    },
    "technical_indicators": {
        "rsi_period": 14,
        "ema_short": 12,
        "ema_long": 26,
        "macd_signal": 9,
        "bollinger_window": 20,
        "bollinger_std_dev": 2,
        "atr_period": 14,
    },
    "save_paths": {
        "raw_data": "data/raw/",
        "processed_data": "data/processed/",
    }
    "support_resistance": {
        "window": 10,
        "tolerance": 0.0005,
        "lookback_days": 365,
        "min_touches": 2
    },
    "visualization": {
        "chart_days": 180
    }
    "dateset":{
        "lookback_window": 60,
        "forecast_horizon": 5,
        "train_split": 0.8,
        "target_type": "close_price"
    },
    "scaling":{
        "method": "standard"
    }
    "training": {
        "epochs": 50,
        "batch_size" : 32,
        "learning_rate": 0.001,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "model_save_path": "modeld/checkpoints/lstm_model.pth"
    }
    "fundamentals":{
        "api_key": "YOUR_FMP_API_KEY",
        "base_url": "https://financialmodelingprep.com/api/v3",
        "metrics":[
            "peRatio", "priceToBookRatio", "returnOnEquityTTM", "netProfitMarginTTM",
            "revenuePerShareTTM", "eps", "debtToEquityTTM", "currentRatioTTM"
        ],
        "quaters": 8
    }
    "hybrid_model":{
        "epochs": 50,
        "batch_size": 32,
        "learning": 0.001,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "model_save_path": "models/checkpoints/hybrid_model.pth"
    }
    "genai":{
        "use_openai": True,
        "openai_model": "gpt-4-turbo",
        "summary_lenght": "short",
        "api_key_env": "OPENAI_API_KEY"
    }
    "ensemble": {
        "use_stacking": True,
        "weights": [0.6, 0.4],
        "models": {
            "lstm": "models/checkpoints/lstm_model.pth",
            "xgb": "models/checkpoints/xgb_model.json"
        }
    }
    "sentiment": {
        "symbol": "AAPL",
        "days_back": 90,
        "api_key": "YOUR_NEWS_API_KEY",  # from https://newsapi.org
        "api_url": "https://newsapi.org/v2/everything",
        "model_name": "yiyanghkust/finbert-tone"
    }
}