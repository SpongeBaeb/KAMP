import pandas as pd
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
data_dir = ROOT.parent / "①Molding"
df = pd.read_csv(data_dir / "moldset_labeled_rg3.csv")
df.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
df = df.sort_values('idx').reset_index(drop=True)

feats = [c for c in df.columns if c not in ('idx', 'PassOrFail')]
df['Virtual_ID'] = df.groupby(feats, sort=False).cumcount()

# Filter to only Virtual_ID == 0
df0 = df[df['Virtual_ID'] == 0].copy()
print(f"Total Virtual_ID=0: {len(df0)}, Defects: {df0['PassOrFail'].sum()}")

# Let's try to train a model JUST on df0 using original features
X0 = df0[feats].to_numpy()
y0 = df0['PassOrFail'].to_numpy()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
aucs = []
for tr, te in cv.split(X0, y0):
    m = HistGradientBoostingClassifier(class_weight='balanced')
    m.fit(X0[tr], y0[tr])
    aucs.append(roc_auc_score(y0[te], m.predict_proba(X0[te])[:, 1]))
print(f"ROC-AUC on Virtual_ID=0 using basic feats: {np.mean(aucs)}")

# Let's add rolling features
df0['MA_5'] = df0['Max_Injection_Pressure'].rolling(5).mean().fillna(0)
df0['MA_10'] = df0['Max_Injection_Pressure'].rolling(10).mean().fillna(0)
df0['idx_feature'] = df0['idx']

X0_ext = df0[feats + ['MA_5', 'MA_10', 'idx_feature']].to_numpy()
aucs_ext = []
for tr, te in cv.split(X0_ext, y0):
    m = HistGradientBoostingClassifier(class_weight='balanced')
    m.fit(X0_ext[tr], y0[tr])
    aucs_ext.append(roc_auc_score(y0[te], m.predict_proba(X0_ext[te])[:, 1]))
print(f"ROC-AUC on Virtual_ID=0 using extended feats (including idx leak): {np.mean(aucs_ext)}")
