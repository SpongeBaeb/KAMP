import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import time

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------
# 1. Data Prep & Feature Engineering (From Phase 2-X)
# ---------------------------------------------------------
def load_and_engineer_data():
    df = pd.read_csv(ROOT.parent / "①Molding" / "moldset_labeled_rg3.csv")
    df.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
    df = df.sort_values('idx').reset_index(drop=True)
    
    feats = [c for c in df.columns if c not in ('idx', 'PassOrFail')]
    
    df['Virtual_ID'] = df.groupby(feats, sort=False).cumcount()
    df['Cumulative_Shots'] = df.index
    
    for col in feats:
        if df[col].nunique() > 5:
            df[f'{col}_roll5_mean'] = df[col].rolling(5, min_periods=1).mean()
            df[f'{col}_lag1'] = df[col].shift(1).bfill()
            
    df['Defects_in_last_10'] = df['PassOrFail'].shift(1).rolling(10, min_periods=1).sum().fillna(0)
    
    X = df[[c for c in df.columns if c not in ('idx', 'PassOrFail')]]
    
    top_5 = ['Max_Injection_Pressure', 'Barrel_Temperature_1', 'Cycle_Time', 'Max_Switch_Over_Pressure', 'Cushion_Position']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_feats = poly.fit_transform(df[top_5])
    poly_df = pd.DataFrame(poly_feats, columns=poly.get_feature_names_out(top_5))
    poly_df = poly_df.drop(columns=top_5)
    X = pd.concat([X, poly_df], axis=1)
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    pca = PCA(n_components=5, random_state=42)
    pca_feats = pca.fit_transform(X_scaled)
    for i in range(5):
        X[f'PCA_{i}'] = pca_feats[:, i]
        
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_dist = kmeans.fit_transform(X_scaled)
    for i in range(3):
        X[f'KMeans_dist_{i}'] = kmeans_dist[:, i]
        
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Feature Selection
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, df['PassOrFail'])
    selector = SelectFromModel(rf, prefit=True, threshold='median')
    X_sel = pd.DataFrame(selector.transform(X), columns=X.columns[selector.get_support()])
    
    if 'Virtual_ID' not in X_sel.columns:
        X_sel['Virtual_ID'] = X['Virtual_ID'].values
        
    return X_sel, df['PassOrFail']

# ---------------------------------------------------------
# 2. Optuna Tuning for CatBoost
# ---------------------------------------------------------
def optimize_catboost(X, y):
    print("\n[Step 1] Optuna Optimization for CatBoost...")
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'auto_class_weights': 'Balanced',
            'verbose': 0,
            'random_state': 42
        }
        
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        aucs = []
        for train_idx, val_idx in cv.split(X, y):
            model = CatBoostClassifier(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]), early_stopping_rounds=20, verbose=0)
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], preds))
            
        return np.mean(aucs)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print(f"  -> Best CatBoost ROC-AUC: {study.best_value:.4f}")
    return study.best_params

# ---------------------------------------------------------
# 3. Simple LSTM Model (PyTorch) for Time-Series
# ---------------------------------------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq=1, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

from sklearn.base import BaseEstimator, ClassifierMixin

class PyTorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=1):
        self.input_size = input_size
        self.model = SimpleLSTM(input_size)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_sc = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_sc).unsqueeze(1) # sequence length = 1
        y_tensor = torch.FloatTensor(np.array(y)).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
        for epoch in range(20):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                weight = torch.where(batch_y == 1, torch.tensor(20.0), torch.tensor(1.0))
                loss = (loss * weight).mean()
                loss.backward()
                optimizer.step()
        self.classes_ = np.unique(y)
        return self
        
    def predict_proba(self, X):
        X_sc = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_sc).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()
        
        # sklearn expects (N, 2) array for proba
        res = np.zeros((len(X), 2))
        res[:, 1] = preds.flatten()
        res[:, 0] = 1 - res[:, 1]
        return res

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

# ---------------------------------------------------------
# 4. Final Stacking and Evaluation
# ---------------------------------------------------------
def evaluate_stacking(X, y, cat_params):
    print("\n[Step 2] Stacking Ensemble & Evaluation...")
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    cat_params['auto_class_weights'] = 'Balanced'
    cat_params['verbose'] = 0
    cat_params['random_state'] = 42
    
    estimators = [
        ('cat', CatBoostClassifier(**cat_params)),
        ('xgb', XGBClassifier(scale_pos_weight=20, n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(class_weight='balanced', n_estimators=100, random_state=42, verbose=-1)),
        ('lstm', PyTorchSklearnWrapper(input_size=X.shape[1]))
    ]
    
    # LogisticRegression as meta-learner
    stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'))
    
    aucs = []
    # To accumulate predictions for threshold tuning
    all_y_true = []
    all_y_prob = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        stack_clf.fit(X_train, y_train)
        probs = stack_clf.predict_proba(X_test)[:, 1]
        
        aucs.append(roc_auc_score(y_test, probs))
        all_y_true.extend(y_test)
        all_y_prob.extend(probs)
        
    mean_auc = np.mean(aucs)
    print(f"  -> Stacking ROC-AUC: {mean_auc:.4f}")
    
    return all_y_true, all_y_prob, mean_auc

# ---------------------------------------------------------
# 5. Threshold Optimization
# ---------------------------------------------------------
def optimize_threshold(y_true, y_prob):
    print("\n[Step 3] Threshold Optimization (Cost-Sensitive)...")
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Maximize F1
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_pred = (np.array(y_prob) >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f"  -> Best Threshold: {best_thresh:.4f}")
    print(f"  -> F1-Score: {f1_scores[best_idx]:.4f}")
    print(f"  -> Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  -> Recall: {recalls[best_idx]:.4f} | Precision: {precisions[best_idx]:.4f}")

def main():
    print("=== Phase 3: The Final Frontier ===")
    start_time = time.time()
    
    X, y = load_and_engineer_data()
    print(f"Data Prepared. Shape: {X.shape}, Positives: {y.sum()}")
    
    best_cat_params = optimize_catboost(X, y)
    
    y_true, y_prob, final_auc = evaluate_stacking(X, y, best_cat_params)
    
    optimize_threshold(y_true, y_prob)
    
    print(f"\n=== All Completed in {time.time()-start_time:.1f} seconds ===")

if __name__ == "__main__":
    main()
