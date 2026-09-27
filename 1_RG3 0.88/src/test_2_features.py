import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT.parent / "①Molding" / "moldset_labeled_rg3.csv")
df.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
df = df.sort_values('idx').reset_index(drop=True)

feats = [c for c in df.columns if c not in ('idx', 'PassOrFail')]
df['Virtual_ID'] = df.groupby(feats, sort=False).cumcount()
df['Cumulative_Shots'] = df.index

X = df[['Virtual_ID', 'Cumulative_Shots']]
y = df['PassOrFail']

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

aucs = []
for train_idx, test_idx in cv.split(X, y):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    model = CatBoostClassifier(iterations=1000, depth=6, auto_class_weights='Balanced', verbose=0)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    aucs.append(roc_auc_score(y_test, preds))

print(f"ROC-AUC with only 2 features: {np.mean(aucs):.4f}")
