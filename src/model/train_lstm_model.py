import os
import numpy as np
import torch
import torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.config import CONFIG

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers= num_layers,
            dropout = dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x ):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_data():
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")


    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_trian, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(X_test, dtype= torch.float32)
    y_test = torch.tensor(y_test, dtype= torch.float32).unsqueeze(1)
    
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


def evaluate_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        y_true = y_test.numpy()
        mae = mean_absolute_error(y_true, preds)
        rmse = mean_squared_error(y_true, preds, squared= False)
        direction_acc = np.mean(np.sign(preds[1:] - preds[:-1]) == np.sign(y_true[1:] - y_true[:-1]))
    return mae, rmse, direction_acc


if __name__ == "__main__":
    cfg = CONFIG["training"]
    print("[INFO] Loading dataset...")
    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = LSTModel(
        input_dim = input_dim,
        hidden_dim= cfg["hidden_dim"]
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= cfg["learning_rate"])

    train_data = TensorDateset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)

    print("[INFO] Starting training...")
    for epoch in range(cfg["epochs"]):
        loss = train_model(model, train_loader, criterion, optimizer, device)
        mae, rmse, acc = evaluate_model(model, X_test, y_test, device)
        
        print(f"Epoch [ {epoch+1}/{cfg['epochs']}]:"f"Train Loss={loss:.6f} | Test RMSE={rmse:.4f} | MAE={mae..4f} | DirACC={acc:.3f}")

        os.makedirs(os.path.dirname(cfg["model_save_path"]), exist_ok=True)
        torch.save(model.state_dict(), cfg['model_save_path'])
        print(f"[SUCCESS] Model saved to {cfg['model_save_path']}")
        