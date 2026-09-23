import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def evaluate_daily_counts(df_raw, true_defects):
    daily_pred = df_raw.groupby('working time')['is_anomaly'].sum().reset_index()
    daily_pred.columns = ['date', 'pred_defects']
    
    # Merge with true defects
    # Note: true_defects has dates as strings or objects, let's cast
    daily_pred['date'] = daily_pred['date'].astype(str)
    true_defects['date'] = true_defects['date'].astype(str)
    
    merged = pd.merge(daily_pred, true_defects, on='date', how='left').fillna(0)
    
    # We only evaluate on days that have true labels (the 9 days)
    # Actually, true_defects has 8 rows? Wait, 2020-03-24 to 2020-04-07 is 8 days?
    # 24, 25, 26, 30, 31, 02, 03, 07 -> 8 days!
    # What about 2020-03-27? The raw data has 1648 rows for 2020-03-27, but result sheet had NO rows for 03-27.
    # Does that mean 0 defects on 03-27? YES. fillna(0) handles this.
    
    rmse = np.sqrt(mean_squared_error(merged['true_defects'], merged['pred_defects']))
    # Add a small epsilon to variance to avoid division by zero in correlation
    if merged['pred_defects'].std() == 0:
        corr = 0
    else:
        corr, _ = pearsonr(merged['true_defects'], merged['pred_defects'])
        
    return rmse, corr, merged

def train_isolation_forest(df, features, true_defects):
    print("\n--- Training Isolation Forest ---")
    best_rmse = float('inf')
    best_corr = -1
    best_model = None
    best_df = None
    
    # Search space for contamination
    contaminations = [0.002, 0.003, 0.004, 0.005, 0.01]
    
    for c in contaminations:
        clf = IsolationForest(contamination=c, random_state=42, n_estimators=200)
        # Predict: 1 for normal, -1 for anomaly
        preds = clf.fit_predict(df[features])
        
        # Convert to 0 for normal, 1 for anomaly
        df['is_anomaly'] = (preds == -1).astype(int)
        
        rmse, corr, merged = evaluate_daily_counts(df, true_defects)
        print(f"Contamination: {c:.4f} | RMSE: {rmse:.2f} | Corr: {corr:.3f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_corr = corr
            best_model = clf
            best_df = df.copy()
            
    return best_model, best_df, best_rmse

# PyTorch Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(df, features, true_defects):
    print("\n--- Training Autoencoder ---")
    X = torch.tensor(df[features].values, dtype=torch.float32)
    dataset = TensorDataset(X, X)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = Autoencoder(len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    model.train()
    for epoch in range(50):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            
    # Inference
    model.eval()
    with torch.no_grad():
        reconstructed = model(X)
        mse = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
        
    best_rmse = float('inf')
    best_df = None
    
    # Threshold based on quantiles to match expected defect rate
    quantiles = [0.99, 0.995, 0.996, 0.997, 0.998]
    for q in quantiles:
        threshold = np.quantile(mse, q)
        df['is_anomaly'] = (mse > threshold).astype(int)
        rmse, corr, merged = evaluate_daily_counts(df, true_defects)
        print(f"Quantile: {q:.4f} | Threshold: {threshold:.2f} | RMSE: {rmse:.2f} | Corr: {corr:.3f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_df = df.copy()
            
    return best_df, best_rmse

if __name__ == "__main__":
    df = pd.read_csv('preprocessed_raw.csv')
    true_defects = pd.read_csv('true_daily_defects.csv')
    
    scaled_features = [c for c in df.columns if c.endswith('_scaled')]
    
    # 1. Isolation Forest
    _, df_iforest, rmse_iforest = train_isolation_forest(df, scaled_features, true_defects)
    
    # 2. Autoencoder
    df_ae, rmse_ae = train_autoencoder(df, scaled_features, true_defects)
    
    print("\n--- Best Model Selection ---")
    if rmse_iforest <= rmse_ae:
        print("Isolation Forest selected!")
        best_df = df_iforest
    else:
        print("Autoencoder selected!")
        best_df = df_ae
        
    # Save the pseudo-labeled data for XAI
    best_df.to_csv('pseudo_labeled_data.csv', index=False)
    print("Saved pseudo-labeled data to pseudo_labeled_data.csv")
