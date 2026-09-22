import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== Experiment 6-3: PyTorch Deep Learning ===")
df = pd.read_csv('preprocessed_data.csv')

lags_to_add = list(range(1, 25)) + [48, 72, 96, 168]
for i in lags_to_add: df[f'Peak_lag_{i}'] = df['Peak'].shift(i)
for w in [3, 6, 12, 24, 48, 168]:
    df[f'Peak_roll_mean_{w}'] = df['Peak'].shift(1).rolling(window=w).mean()
    df[f'Peak_roll_std_{w}'] = df['Peak'].shift(1).rolling(window=w).std()

df['Peak_roll_max_24'] = df['Peak'].shift(1).rolling(window=24).max()
df['Peak_roll_min_24'] = df['Peak'].shift(1).rolling(window=24).min()
df['Peak_roll_max_168'] = df['Peak'].shift(1).rolling(window=168).max()
df['Peak_roll_min_168'] = df['Peak'].shift(1).rolling(window=168).min()
df['Peak_diff_1'] = df['Peak_lag_1'] - df['Peak_lag_2']
df['Peak_diff_24'] = df['Peak_lag_1'] - df['Peak_lag_24']
df['Peak_diff_168'] = df['Peak_lag_1'] - df['Peak_lag_168']
df['THI'] = df['기온'] - 0.55 * (1 - df['습도']/100.0) * (df['기온'] - 14.5)
for span in [3, 6, 12, 24, 168]:
    df[f'Peak_ewm_{span}'] = df['Peak'].shift(1).ewm(span=span).mean()

df = df.dropna().reset_index(drop=True)
df = df[df['is_inrush'] == 0].reset_index(drop=True)

y_abs = df['Peak'].copy()
y_diff = df['Peak'] - df['Peak_lag_1']
X = df.drop(columns=['Peak','날짜','시간','15분','30분','45분','60분','평균',
                      'is_inrush','Peak_MA3','Peak_MA6','Peak_MA12','Peak_MA24'], errors='ignore')
for c in X.select_dtypes(['category']).columns: X[c] = X[c].cat.codes

selector = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
selector.fit(X, y_diff)
imp = pd.Series(selector.feature_importances_, index=X.columns)
top30 = imp.nlargest(30).index.tolist()
X = X[top30]

# PyTorch Model
class TimeSeriesMLP(nn.Module):
    def __init__(self, input_dim):
        super(TimeSeriesMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

tscv = TimeSeriesSplit(n_splits=5)
device = torch.device('cpu')

rmse_scores_pt = []
rmse_scores_blend = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr_full, y_tr_full = X.iloc[train_idx], y_diff.iloc[train_idx]
    X_te, actual = X.iloc[test_idx], y_abs.iloc[test_idx]
    
    # Validation split for early stopping
    val_sz = max(int(len(X_tr_full)*0.1), 1)
    X_tr, y_tr = X_tr_full.iloc[:-val_sz], y_tr_full.iloc[:-val_sz]
    X_val, y_val = X_tr_full.iloc[-val_sz:], y_tr_full.iloc[-val_sz:]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)
    
    # PyTorch DataLoaders
    tr_dataset = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr.values))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val.values))
    
    tr_loader = DataLoader(tr_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train MLP
    model = TimeSeriesMLP(X_tr_s.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience, patience_counter = 10, 0
    
    for epoch in range(100):
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
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        p_pt_te = model(torch.FloatTensor(X_te_s)).numpy()
        
    # Baseline (ET + LGB)
    seeds_5 = [42, 123, 456, 789, 2024]
    preds_et = []
    for s in seeds_5:
        m_et = ExtraTreesRegressor(n_estimators=500, random_state=s, n_jobs=-1, min_samples_leaf=3)
        m_et.fit(X_tr_full, y_tr_full)
        preds_et.append(m_et.predict(X_te))
    p_et_te = np.mean(preds_et, axis=0)

    m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.036, num_leaves=64, max_depth=12, verbosity=-1, random_state=42)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgb_te = m_lgb.predict(X_te)
    
    p_base_te = 0.90 * p_et_te + 0.10 * p_lgb_te
    
    # Test RMSE (PyTorch Standalone)
    pt_preds_abs = X_te['Peak_lag_1'].values + p_pt_te
    rmse_pt = np.sqrt(mean_squared_error(actual, pt_preds_abs))
    rmse_scores_pt.append(rmse_pt)
    
    # Test RMSE (Blend Base + PyTorch)
    blend_preds_abs = X_te['Peak_lag_1'].values + (0.90 * p_base_te + 0.10 * p_pt_te)
    rmse_blend = np.sqrt(mean_squared_error(actual, blend_preds_abs))
    rmse_scores_blend.append(rmse_blend)
    
    print(f'Fold {fold+1}: PyTorch RMSE={rmse_pt:.4f} | Blend RMSE={rmse_blend:.4f}')

print(f'\nFinal Average PyTorch RMSE: {np.mean(rmse_scores_pt):.4f}')
print(f'Final Average Blend RMSE: {np.mean(rmse_scores_blend):.4f}')
