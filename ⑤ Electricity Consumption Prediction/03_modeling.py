# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb
import os

os.makedirs('models', exist_ok=True)

df = pd.read_csv('preprocessed_data.csv')

# Features and target
features = [
    '생산량', '기온', '풍속', '습도', '강수량', '공장인원', 
    'Peak_lag1', 'Peak_lag2', 'Peak_lag24', 'Peak_lag48', 
    'Peak_MA3', 'Peak_MA6', 'Peak_MA12', 'Peak_MA24', 
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'Financial_Risk_Index'
]

X_df = df[features]
y_df = df[['Peak']]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaled_X = scaler_X.fit_transform(X_df)
scaled_y = scaler_y.fit_transform(y_df)

def create_sequences(feat_data, target_data, seq_length=24, pred_length=24):
    X, y = [], []
    for i in range(len(feat_data) - seq_length - pred_length + 1):
        X.append(feat_data[i:i+seq_length])
        y.append(target_data[i+seq_length:i+seq_length+pred_length])
    return np.array(X), np.array(y)

seq_len = 24
pred_len = 24
X_seq, y_seq = create_sequences(scaled_X, scaled_y, seq_len, pred_len)

train_size = int(len(X_seq) * 0.8)
X_train, y_train = X_seq[:train_size], y_seq[:train_size]
X_test, y_test = X_seq[train_size:], y_seq[train_size:]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

input_dim = len(features)
hidden_dim = 128
model = LSTMModel(input_dim, hidden_dim, pred_len, num_layers=2)
criterion = nn.MSELoss() # Standard MSE to optimize RMSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

print("Starting LSTM training...")
epochs = 50
best_loss = float('inf')
patience = 7
patience_counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_X)
        loss = criterion(out, batch_y.squeeze(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss/len(train_loader)
    
    # Validation step to use early stopping
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            out = model(batch_X)
            loss = criterion(out, batch_y.squeeze(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss/len(test_loader)
    scheduler.step(avg_val_loss)
    
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), 'models/lstm_best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model for evaluation
model.load_state_dict(torch.load('models/lstm_best_model.pth', weights_only=True))

model.eval()
test_preds = []
test_actuals = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        out = model(batch_X)
        test_preds.append(out.numpy())
        test_actuals.append(batch_y.squeeze(-1).numpy())

test_preds = np.concatenate(test_preds)
test_actuals = np.concatenate(test_actuals)

test_preds_inv = scaler_y.inverse_transform(test_preds)
test_actuals_inv = scaler_y.inverse_transform(test_actuals)

mse = np.mean((test_preds_inv - test_actuals_inv)**2)
rmse = np.sqrt(mse)
r2_lstm = r2_score(test_actuals_inv, test_preds_inv)
print(f"LSTM Test RMSE: {rmse:.4f}")
print(f"LSTM Test R^2: {r2_lstm:.4f}")

# Hybrid Ensemble (LightGBM on Residuals)
print("\nTraining LightGBM on residuals for t+1 (1-hour ahead)...")

train_preds = []
train_actuals = []
train_loader_seq = DataLoader(train_dataset, batch_size=64, shuffle=False)
model.eval()
with torch.no_grad():
    for batch_X, batch_y in train_loader_seq:
        out = model(batch_X)
        train_preds.append(out.numpy())
        train_actuals.append(batch_y.squeeze(-1).numpy())

train_preds = np.concatenate(train_preds)
train_actuals = np.concatenate(train_actuals)
train_residuals = train_actuals - train_preds

lgbm_X_train = X_train[:, -1, :] 
lgbm_y_train = train_residuals[:, 0] 

# Use Optuna-like or more aggressive LGBM params
lgbm = lgb.LGBMRegressor(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=7,
    num_leaves=64,
    random_state=42
)
lgbm.fit(lgbm_X_train, lgbm_y_train)

lgbm_X_test = X_test[:, -1, :]
lgbm_preds = lgbm.predict(lgbm_X_test)

final_preds_t1 = test_preds[:, 0] + lgbm_preds
final_preds_t1_inv = scaler_y.inverse_transform(final_preds_t1.reshape(-1,1))
actuals_t1_inv = scaler_y.inverse_transform(test_actuals[:, 0].reshape(-1,1))

rmse_hybrid = np.sqrt(np.mean((final_preds_t1_inv - actuals_t1_inv)**2))
r2_hybrid = r2_score(actuals_t1_inv, final_preds_t1_inv)
print(f"Hybrid Model Test RMSE for t+1: {rmse_hybrid:.4f}")
print(f"Hybrid Model Test R^2 for t+1: {r2_hybrid:.4f}")

lgbm.booster_.save_model('models/lgbm_residual.txt')
print("Modeling complete. Models saved to 'models/'")
