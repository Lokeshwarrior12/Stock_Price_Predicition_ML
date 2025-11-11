# src/model/train_xgb_model.py

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.config import CONFIG


def load_tabular_data(symbol: str):
    """
    Load processed dataset (merged features: technical + fundamentals + sentiment)
    and separate target variable.
    """
    data_path = os.path.join(CONFIG["save_paths"]["processed_data"], f"{symbol}_merged_features.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Target" not in df.columns:
        raise ValueError("Processed data must contain a 'Target' column for supervised learning")

    # Drop non-numeric or unnecessary columns
    # Keep only numeric columns and fill missing values instead of dropping everything
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.median(), inplace=True)


    X = df.drop(columns=["Target"])
    y = df["Target"]
    return X, y


def train_xgb_model(X, y, params, save_path):
    """
    Train an XGBoost regression model on tabular (fundamental/sentiment) features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.get("test_size", 0.2), random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("[INFO] Training XGBoost model...")
    model = xgb.train(
        params=params["train_params"],
        dtrain=dtrain,
        num_boost_round=params["train_params"].get("num_boost_round", 300),
        evals=[(dtest, "Test")],
        early_stopping_rounds=params.get("early_stopping_rounds", 30),
        verbose_eval=50,
    )

    print("[INFO] Evaluating model...")
    preds = model.predict(dtest)


    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        direction_acc = np.mean(
            np.sign(preds[1:] - preds[:-1]) == np.sign(y_test.values[1:] - y_test.values[:-1])
        )

        print(f"[RESULT] RMSE={rmse:.4f} | MAE={mae:.4f} | Directional Accuracy={direction_acc:.3f}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_model(save_path)
        print(f"[SUCCESS] Model saved to {save_path}")

        return model, rmse, mae, direction_acc



if __name__ == "__main__":
    cfg = CONFIG["xgboost"]
    symbol = CONFIG["stock_symbol"]

    print(f"[INFO] Loading processed data for {symbol}...")
    X, y = load_tabular_data(symbol)

    model, rmse, mae, acc = train_xgb_model(X, y, cfg, cfg["model_save_path"])
    print("[DONE] XGBoost training complete ✅")
