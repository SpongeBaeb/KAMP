"""Deep exploration of unlabeled data to understand its relationship to labeled data."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT.parent / "\u2460Molding"

# Load both datasets
df_labeled = pd.read_csv(data_dir / "moldset_labeled_rg3.csv")
df_labeled.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
df_labeled = df_labeled.sort_values('idx').reset_index(drop=True)

df_unlabeled = pd.read_csv(data_dir / "moldset_unlabeled_rg3.csv")
df_unlabeled.rename(columns={"Unnamed: 0": "idx"}, inplace=True)

sensor_cols = [c for c in df_labeled.columns if c not in ('idx', 'PassOrFail', 'Clamp_Open_Position')]
y = df_labeled['PassOrFail']

print("=" * 60)
print("EXPERIMENT 1: Isolation Forest trained on UNLABELED data")
print("=" * 60)
print(f"Training on {len(df_unlabeled)} unlabeled samples...")

# Train anomaly detector on unlabeled data (assumed mostly normal)
iso = IsolationForest(n_estimators=500, contamination=0.02, random_state=42)
iso.fit(df_unlabeled[sensor_cols])

# Score the labeled data
anomaly_scores = -iso.score_samples(df_labeled[sensor_cols])  # higher = more anomalous
auc = roc_auc_score(y, anomaly_scores)
print(f"  ROC-AUC (IsolationForest anomaly score): {auc:.4f}")

print("\n" + "=" * 60)
print("EXPERIMENT 2: LOF trained on UNLABELED data")
print("=" * 60)

lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.02)
lof.fit(df_unlabeled[sensor_cols])
lof_scores = -lof.score_samples(df_labeled[sensor_cols])
auc_lof = roc_auc_score(y, lof_scores)
print(f"  ROC-AUC (LOF anomaly score): {auc_lof:.4f}")

print("\n" + "=" * 60)
print("EXPERIMENT 3: Mahalanobis Distance from unlabeled distribution")
print("=" * 60)

from numpy.linalg import pinv
unlabeled_mean = df_unlabeled[sensor_cols].mean().values
unlabeled_cov = df_unlabeled[sensor_cols].cov().values
cov_inv = pinv(unlabeled_cov)

labeled_vals = df_labeled[sensor_cols].values
mahal_dists = []
for row in labeled_vals:
    diff = row - unlabeled_mean
    md = np.sqrt(diff @ cov_inv @ diff)
    mahal_dists.append(md)

mahal_dists = np.array(mahal_dists)
auc_mahal = roc_auc_score(y, mahal_dists)
print(f"  ROC-AUC (Mahalanobis Distance): {auc_mahal:.4f}")

print("\n" + "=" * 60)
print("EXPERIMENT 4: KNN Distance to unlabeled data")
print("=" * 60)

from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(df_unlabeled[sensor_cols].values)
distances, _ = nn.kneighbors(df_labeled[sensor_cols].values)
knn_scores = distances.mean(axis=1)
auc_knn = roc_auc_score(y, knn_scores)
print(f"  ROC-AUC (Mean KNN Distance to unlabeled): {auc_knn:.4f}")

print("\n" + "=" * 60)
print("EXPERIMENT 5: Combine anomaly scores + Virtual_ID + CatBoost (strict CV)")
print("=" * 60)

from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier

# Build feature set
feats_raw = [c for c in df_labeled.columns if c not in ('idx', 'PassOrFail')]
df_labeled['Virtual_ID'] = df_labeled.groupby(feats_raw, sort=False).cumcount()
df_labeled['Cumulative_Shots'] = df_labeled.index
df_labeled['IsoForest_Score'] = anomaly_scores
df_labeled['LOF_Score'] = lof_scores
df_labeled['Mahalanobis_Dist'] = mahal_dists
df_labeled['KNN_Dist'] = knn_scores

feature_cols = ['Virtual_ID', 'Cumulative_Shots', 'IsoForest_Score', 'LOF_Score', 'Mahalanobis_Dist', 'KNN_Dist']
X = df_labeled[feature_cols]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
model = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03, auto_class_weights='Balanced', verbose=0, random_state=42)

aucs = []
for train_idx, test_idx in cv.split(X, y):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    aucs.append(roc_auc_score(y_test, probs))

print(f"  ROC-AUC (CatBoost with anomaly features): {np.mean(aucs):.4f}")

print("\n" + "=" * 60)
print("EXPERIMENT 6: All sensor cols + anomaly scores + Virtual_ID (strict CV)")
print("=" * 60)

feature_cols_full = sensor_cols + ['Virtual_ID', 'Cumulative_Shots', 'IsoForest_Score', 'LOF_Score', 'Mahalanobis_Dist', 'KNN_Dist']
X_full = df_labeled[feature_cols_full]

aucs_full = []
for train_idx, test_idx in cv.split(X_full, y):
    X_train, y_train = X_full.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X_full.iloc[test_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    aucs_full.append(roc_auc_score(y_test, probs))

print(f"  ROC-AUC (Full features + anomaly scores): {np.mean(aucs_full):.4f}")
