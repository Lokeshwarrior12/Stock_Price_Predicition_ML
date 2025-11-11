# src/model/train_hybrid_model.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.utils.config import CONFIG


# ----------------------------
# Hybrid Model Architecture
# ----------------------------

class HybridModel(nn.Module):
    def __init__(self, input_dim_seq, input_dim_fund, hidden_dim=128, num_layers=2, dropout=0.2):
        super(HybridModel, self).__init__()

        # LSTM branch for technical (time-series)
        self.lstm = nn.LSTM(input_dim_seq, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True)

        # Feedforward branch for fundamentals
        self.fund_net = nn.Sequential(
            nn.Linear(input_dim_fund, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Combined layer
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_seq, x_fund):
        # LSTM branch
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out[:, -1, :]  # last timestep output

        # Fundamentals branch
        fund_out = self.fund_net(x_fund)

        # Concatenate both representations
        combined = torch.cat((lstm_out, fund_out), dim=1)
        out = self.fc_combined(combined)
        return out


# ----------------------------
# Data Preparation
# ----------------------------

def prepare_data(symbol: str, lookback_window=60, forecast_horizon=5, train_split=0.8):
    df = pd.read_csv(f"data/processed/{symbol}_merged_features.csv")
    df.dropna(how="all", inplace=True)

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # expected technical columns (lower-case)
    tech_cols = ["open", "high", "low", "close", "volume",
                 "rsi", "ema_short", "ema_long", "macd", "atr"]

    # keep only those that actually exist
    tech_cols = [c for c in tech_cols if c in df.columns]
    if not tech_cols:
        raise ValueError("No technical columns found in merged dataset. "
                         "Check column names in data/processed CSV.")

    # remaining numeric, non-technical columns for fundamentals
    fund_cols = [c for c in df.select_dtypes(include=["number"]).columns
                 if c not in tech_cols + ["target"]]

    print(f"[DEBUG] Using {len(tech_cols)} technical features and {len(fund_cols)} fundamental features")

    scaler_tech = StandardScaler()
    scaler_fund = StandardScaler()
    df_tech = scaler_tech.fit_transform(df[tech_cols])
    df_fund = scaler_fund.fit_transform(df[fund_cols])


    # Target (forecast next-day price)
    # Target (forecast next-day price)
    if "close" in df.columns:
        y = df["close"].shift(-forecast_horizon).dropna().values
    elif "Close" in df.columns:
        y = df["Close"].shift(-forecast_horizon).dropna().values
    else:
        raise KeyError("Neither 'close' nor 'Close' column found in dataset for target creation.")

    df_tech = df_tech[:len(y), :]
    df_fund = df_fund[:len(y), :]

    # Create sequential inputs for technicals
    X_seq, X_fund, Y = [], [], []
    print(f"[DEBUG] Total samples in dataset: {len(df)}")
    print(f"[DEBUG] y length: {len(y)}")
    print(f"[DEBUG] Lookback window: {lookback_window}")
    print(f"[DEBUG] Forecast horizon: {forecast_horizon}")

    for i in range(len(y) - lookback_window):
        X_seq.append(df_tech[i:i + lookback_window])
        X_fund.append(df_fund[i + lookback_window])
        Y.append(y[i + lookback_window])

    X_seq, X_fund, Y = np.array(X_seq), np.array(X_fund), np.array(Y)

    # Train/test split
    split_idx = int(len(X_seq) * train_split)
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    X_train_fund, X_test_fund = X_fund[:split_idx], X_fund[split_idx:]
    y_train, y_test = Y[:split_idx], Y[split_idx:]

    return X_train_seq, X_test_seq, X_train_fund, X_test_fund, y_train, y_test


# ----------------------------
# Training Loop
# ----------------------------

def train_hybrid(model, loaders, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_seq, X_fund, y in loaders:
        X_seq, X_fund, y = X_seq.to(device), X_fund.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X_seq, X_fund)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loaders)


def evaluate(model, X_seq, X_fund, y, device):
    model.eval()
    with torch.no_grad():
        preds = model(X_seq.to(device), X_fund.to(device)).cpu().numpy().flatten()
    y_true = y.numpy().flatten()
    rmse = np.sqrt(np.mean((y_true - preds) ** 2))
    mae = np.mean(np.abs(y_true - preds))
    return rmse, mae


# ----------------------------
# Main Training Execution
# ----------------------------

if __name__ == "__main__":
    cfg = CONFIG["hybrid_model"]
    symbol = CONFIG["stock_symbol"]

    print("[INFO] Loading and preparing merged dataset...")
    X_train_seq, X_test_seq, X_train_fund, X_test_fund, y_train, y_test = prepare_data(
    symbol,
    lookback_window=10,   # smaller so you have data
    forecast_horizon=1
)

    # Convert to tensors
    X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32)
    X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32)
    X_train_fund = torch.tensor(X_train_fund, dtype=torch.float32)
    X_test_fund = torch.tensor(X_test_fund, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print("\n[DEBUG] Shape summary:")
    print("X_train_seq:", X_train_seq.shape)
    print("X_train_fund:", X_train_fund.shape)
    print("y_train:", y_train.shape)


    # Model setup
    input_dim_seq = X_train_seq.shape[2]
    input_dim_fund = X_train_fund.shape[1]
    model = HybridModel(input_dim_seq, input_dim_fund, hidden_dim=cfg["hidden_dim"],
                        num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # Dataloader
    train_data = TensorDataset(X_train_seq, X_train_fund, y_train)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)

    print("[INFO] Training hybrid model...")
    for epoch in range(cfg["epochs"]):
        loss = train_hybrid(model, train_loader, criterion, optimizer, device)
        rmse, mae = evaluate(model, X_test_seq, X_test_fund, y_test, device)
        print(f"Epoch [{epoch+1}/{cfg['epochs']}]: Loss={loss:.6f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    os.makedirs(os.path.dirname(cfg["model_save_path"]), exist_ok=True)
    torch.save(model.state_dict(), cfg["model_save_path"])
    print(f"[SUCCESS] Hybrid model saved to {cfg['model_save_path']}")
