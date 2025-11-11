# src/model/hybrid_ensemble.py

import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.model.train_lstm_model import LSTMModel
from src.utils.config import CONFIG


# -----------------------------------------------
# 🧩 1. Ensemble Class
# -----------------------------------------------

class HybridEnsemble:
    """
    Combines predictions from LSTM, XGBoost, and optionally Transformer or other models.
    Supports simple weighted averaging or stacking.
    """

    def __init__(self, weights=None, use_stacking=False):
        """
        Args:
            weights (list): Model weights for averaging (sum=1).
            use_stacking (bool): Whether to train a meta-learner (Linear Regression).
        """
        self.weights = weights or [0.5, 0.5]  # default 50-50 for LSTM & XGB
        self.use_stacking = use_stacking
        self.meta_model = LinearRegression() if use_stacking else None

        # Placeholder for sub-models
        self.lstm_model = None
        self.xgb_model = None

    # -----------------------------------------------
    # 🧠 Load Sub-models
    # -----------------------------------------------
    def load_lstm(self, path, input_dim):
        cfg = CONFIG["training"]
        self.lstm_model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"]
        )
        self.lstm_model.load_state_dict(torch.load(path, map_location="cpu"))
        self.lstm_model.eval()

    def load_xgboost(self, path):
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(path)

    # -----------------------------------------------
    # 🔮 Predict
    # -----------------------------------------------
    def predict(self, X_seq, X_tabular):
        preds = []

        # LSTM predictions
        if self.lstm_model:
            X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
            with torch.no_grad():
                lstm_preds = self.lstm_model(X_seq_tensor).cpu().numpy().flatten()
            preds.append(lstm_preds)

        # XGBoost predictions
        # XGBoost predictions (with safe feature handling)
        if self.xgb_model:
            try:
                expected_names = self.xgb_model.feature_names
                if expected_names is not None and len(expected_names) == X_tabular.shape[1]:
                    import pandas as pd
                    X_tabular_df = pd.DataFrame(X_tabular, columns=expected_names)
                    dmatrix = xgb.DMatrix(X_tabular_df)
                else:
                    dmatrix = xgb.DMatrix(X_tabular, feature_names=None)
            except Exception:
                dmatrix = xgb.DMatrix(X_tabular)

            xgb_preds = self.xgb_model.predict(dmatrix, validate_features=False)
            preds.append(xgb_preds)


        # Weighted average ensemble
        if not self.use_stacking:
            preds = np.array(preds)
            final_pred = np.average(preds, axis=0, weights=self.weights)
        else:
            # Stacking ensemble: train a linear regressor on predictions
            stacked_preds = np.vstack(preds).T  # shape: (samples, models)
            final_pred = self.meta_model.predict(stacked_preds)

        return final_pred

    # -----------------------------------------------
    # 🧮 Training Meta Learner (Stacking)
    # -----------------------------------------------
    def train_meta_learner(self, X_seq_train, X_tabular_train, y_train):
        """Train linear meta-learner using outputs from base models."""
        base_preds = []

        # Base model predictions
        if self.lstm_model:
            X_seq_tensor = torch.tensor(X_seq_train, dtype=torch.float32)
            with torch.no_grad():
                lstm_train_pred = self.lstm_model(X_seq_tensor).cpu().numpy().flatten()
            base_preds.append(lstm_train_pred)

        if self.xgb_model:
            xgb_train_pred = self.xgb_model.predict(xgb.DMatrix(X_tabular_train))
            base_preds.append(xgb_train_pred)

        stacked_train = np.vstack(base_preds).T
        self.meta_model.fit(stacked_train, y_train)
        print("[INFO] Meta-learner (stacking) trained successfully ✅")

    # -----------------------------------------------
    # 📈 Evaluation
    # -----------------------------------------------
    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            rmse = mean_squared_error(y_true, y_pred) ** 0.5

        direction_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
        print(f"[RESULT] MAE={mae:.4f}, RMSE={rmse:.4f}, Directional Accuracy={direction_acc:.3f}")
        return mae, rmse, direction_acc


# -----------------------------------------------
# ⚙️ 2. Example Usage
# -----------------------------------------------
if __name__ == "__main__":
    # Simulated demo (replace with real preprocessed data)
    X_seq = np.random.rand(100, 30, 13)   # (samples, timesteps, features)
    X_tab = np.random.rand(100, 20)       # tabular data (fundamentals/sentiment)
    y_true = np.random.rand(100)

    ensemble = HybridEnsemble(weights=[0.6, 0.4])
    ensemble.load_lstm("models/checkpoints/lstm_model.pth", input_dim=13)
    ensemble.load_xgboost("models/checkpoints/xgb_model.json")

    y_pred = ensemble.predict(X_seq, X_tab)
    ensemble.evaluate(y_true, y_pred)
