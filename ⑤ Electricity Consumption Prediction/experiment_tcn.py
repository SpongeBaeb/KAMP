import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== Grandmaster Exp 3: TCN (Temporal Convolutional Network) ===")
df = pd.read_csv('preprocessed_data.csv')

# Drop outliers for stability
df = df.dropna().reset_index(drop=True)
df = df[df['is_inrush'] == 0].reset_index(drop=True)

# We will use raw features for TCN to extract its own lag patterns
raw_features = ['Peak', '기온', '풍속', '습도', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
data = df[raw_features].values

# Create sliding windows for TCN (seq_len = 168)
SEQ_LEN = 168
X_seq, y_seq = [], []
for i in range(len(data) - SEQ_LEN):
    X_seq.append(data[i:i+SEQ_LEN, :])
    y_seq.append(data[i+SEQ_LEN, 0]) # Target is Peak

X_seq = np.array(X_seq) # (N, seq_len, num_features)
y_seq = np.array(y_seq) # (N,)

# TCN Model
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3):
        super(TCN, self).__init__()
        # Conv1d expects (batch, channels, seq_len)
        self.conv1 = nn.Conv1d(num_inputs, num_channels, kernel_size, padding=1, dilation=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size, padding=2, dilation=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(num_channels, num_channels, kernel_size, padding=24, dilation=24)
        self.relu3 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x).squeeze(1)

tscv = TimeSeriesSplit(n_splits=5)
device = torch.device('cpu')
rmse_scores = []

# TCN is slow on CPU, we will do 5-Fold but train for max 25 epochs
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_seq)):
    X_tr_full, y_tr_full = X_seq[train_idx], y_seq[train_idx]
    X_te, actual = X_seq[test_idx], y_seq[test_idx]
    
    val_sz = max(int(len(X_tr_full)*0.1), 1)
    X_tr, y_tr = X_tr_full[:-val_sz], y_tr_full[:-val_sz]
    X_val, y_val = X_tr_full[-val_sz:], y_tr_full[-val_sz:]
    
    # Scale across features
    scalers = [StandardScaler() for _ in range(X_seq.shape[2])]
    X_tr_s = np.zeros_like(X_tr)
    X_val_s = np.zeros_like(X_val)
    X_te_s = np.zeros_like(X_te)
    
    for f in range(X_seq.shape[2]):
        X_tr_s[:, :, f] = scalers[f].fit_transform(X_tr[:, :, f])
        X_val_s[:, :, f] = scalers[f].transform(X_val[:, :, f])
        X_te_s[:, :, f] = scalers[f].transform(X_te[:, :, f])
    
    tr_dataset = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val))
    
    tr_loader = DataLoader(tr_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model = TCN(num_inputs=len(raw_features), num_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(15): # Keep it short for CPU
        model.train()
        for bx, by in tr_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                pred = model(bx)
                val_loss += criterion(pred, by).item() * len(bx)
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        p_tcn = model(torch.FloatTensor(X_te_s)).numpy()
        
    rmse = np.sqrt(mean_squared_error(actual, p_tcn))
    print(f'Fold {fold+1}: TCN RMSE={rmse:.4f}')
    rmse_scores.append(rmse)

avg = np.mean(rmse_scores)
print(f'\nFinal Average RMSE (TCN Standalone): {avg:.4f}')
