import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, d))
    def forward(self, x):
        return self.decoder(self.encoder(x))


def add_physics_features(df):
    df = df.copy()
    df['Cooling_Time'] = df['Cycle_Time'] - (df['Injection_Time'] + df['Plasticizing_Time'] + df['Clamp_Close_Time'])
    df['Temp_Gradient'] = df['Barrel_Temperature_6'] - df['Barrel_Temperature_1']
    df['Avg_Barrel_Temp'] = df[[f'Barrel_Temperature_{i}' for i in range(1, 7)]].mean(axis=1)
    df['Pressure_Drop'] = df['Max_Injection_Pressure'] - df['Max_Switch_Over_Pressure']
    df['Injection_Efficiency'] = df['Max_Injection_Speed'] / (df['Injection_Time'] + 1e-5)
    return df


def run():
    print("=" * 60)
    print("Phase 9c: AE-Guided Pseudo-Labels + Multiple Strategies")
    print("=" * 60)

    df_lab = pd.read_csv('data/moldset_labeled_cn7.csv')
    df_unlab = pd.read_csv('data/moldset_unlabeled_cn7.csv')

    for df in [df_lab, df_unlab]:
        for c in ['Unnamed: 0', 'Clamp_Open_Position']:
            if c in df.columns: df.drop(columns=[c], inplace=True)

    df_lab = add_physics_features(df_lab)
    df_unlab = add_physics_features(df_unlab)

    target = 'PassOrFail'
    features = [c for c in df_lab.columns if c != target]

    X_lab = df_lab[features].values
    y_lab = df_lab[target].values
    X_unlab = df_unlab[features].values

    scaler = StandardScaler()
    X_unlab_sc = scaler.fit_transform(X_unlab)
    X_lab_sc = scaler.transform(X_lab)

    # Train AE
    print("\nTraining Autoencoder (100 epochs)...")
    ae = Autoencoder(X_lab_sc.shape[1])
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    t_unlab = torch.FloatTensor(X_unlab_sc)
    loader = DataLoader(TensorDataset(t_unlab, t_unlab), batch_size=256, shuffle=True)
    ae.train()
    for epoch in range(100):
        for bx, _ in loader:
            loss = nn.MSELoss()(ae(bx), bx)
            opt.zero_grad(); loss.backward(); opt.step()

    ae.eval()
    with torch.no_grad():
        re_unlab = np.mean((X_unlab_sc - ae(t_unlab).numpy()) ** 2, axis=1)

    # --- Strategy A: Original Phase 6 baseline (our 0.6385 champion) ---
    print("\n--- Strategy A: Phase 6 Baseline (LR pseudo-labels) ---")
    pseudo_lr = LogisticRegression(class_weight='balanced', random_state=42)
    pseudo_lr.fit(X_lab_sc, y_lab)
    p_unlab = pseudo_lr.predict_proba(X_unlab_sc)[:, 1]

    def make_pseudo(proba, hi=0.95, lo=0.05, max_n=5000):
        hi_idx = np.where(proba > hi)[0]
        lo_idx = np.where(proba < lo)[0]
        np.random.seed(42)
        if len(lo_idx) > max_n: lo_idx = np.random.choice(lo_idx, max_n, replace=False)
        Xp = np.vstack([X_unlab_sc[hi_idx], X_unlab_sc[lo_idx]])
        yp = np.concatenate([np.ones(len(hi_idx)), np.zeros(len(lo_idx))])
        return Xp, yp

    Xp_a, yp_a = make_pseudo(p_unlab)
    prauc_a = eval_cv(X_lab_sc, y_lab, Xp_a, yp_a, "VotingEnsemble")
    print(f"  Result: {prauc_a:.4f}")

    # --- Strategy B: AE-guided pseudo-labels ---
    # Use reconstruction error to find defects in unlabeled data
    print("\n--- Strategy B: AE-Guided Pseudo-Labels ---")
    re_threshold = np.percentile(re_unlab, 99)  # top 1% = likely defects
    ae_defect_idx = np.where(re_unlab > re_threshold)[0]
    ae_normal_idx = np.where(re_unlab < np.percentile(re_unlab, 50))[0]  # bottom 50%
    np.random.seed(42)
    if len(ae_normal_idx) > 5000:
        ae_normal_idx = np.random.choice(ae_normal_idx, 5000, replace=False)

    Xp_b = np.vstack([X_unlab_sc[ae_defect_idx], X_unlab_sc[ae_normal_idx]])
    yp_b = np.concatenate([np.ones(len(ae_defect_idx)), np.zeros(len(ae_normal_idx))])
    print(f"  AE Pseudo Defects: {len(ae_defect_idx)}, Normals: {len(ae_normal_idx)}")
    prauc_b = eval_cv(X_lab_sc, y_lab, Xp_b, yp_b, "VotingEnsemble")
    print(f"  Result: {prauc_b:.4f}")

    # --- Strategy C: Consensus pseudo-labels (LR agrees with AE) ---
    print("\n--- Strategy C: Consensus (LR + AE agree) ---")
    lr_defect = set(np.where(p_unlab > 0.90)[0])
    ae_defect = set(np.where(re_unlab > np.percentile(re_unlab, 95))[0])
    consensus_defect = np.array(list(lr_defect & ae_defect))
    
    lr_normal = set(np.where(p_unlab < 0.05)[0])
    ae_normal = set(np.where(re_unlab < np.percentile(re_unlab, 50))[0])
    consensus_normal = np.array(list(lr_normal & ae_normal))
    
    np.random.seed(42)
    if len(consensus_normal) > 5000:
        consensus_normal = np.random.choice(consensus_normal, 5000, replace=False)

    if len(consensus_defect) > 0:
        Xp_c = np.vstack([X_unlab_sc[consensus_defect], X_unlab_sc[consensus_normal]])
        yp_c = np.concatenate([np.ones(len(consensus_defect)), np.zeros(len(consensus_normal))])
        print(f"  Consensus Defects: {len(consensus_defect)}, Normals: {len(consensus_normal)}")
        prauc_c = eval_cv(X_lab_sc, y_lab, Xp_c, yp_c, "VotingEnsemble")
        print(f"  Result: {prauc_c:.4f}")
    else:
        prauc_c = 0
        print("  No consensus defects found.")

    # --- Strategy D: Calibrated model ---
    print("\n--- Strategy D: Calibrated Voting Ensemble (Isotonic) ---")
    prauc_d = eval_cv(X_lab_sc, y_lab, Xp_a, yp_a, "CalibratedVoting")
    print(f"  Result: {prauc_d:.4f}")

    # --- Strategy E: No pseudo labels, just original data with balanced bagging ---
    print("\n--- Strategy E: BalancedBagging (No Pseudo Labels) ---")
    prauc_e = eval_cv(X_lab_sc, y_lab, None, None, "BalancedBagging")
    print(f"  Result: {prauc_e:.4f}")

    # --- Strategy F: Phase 6 pseudo + BalancedBagging ---
    print("\n--- Strategy F: BalancedBagging + Phase 6 Pseudo ---")
    prauc_f = eval_cv(X_lab_sc, y_lab, Xp_a, yp_a, "BalancedBagging")
    print(f"  Result: {prauc_f:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL STRATEGIES:")
    print("=" * 60)
    results = {
        'A: Phase6 Baseline': prauc_a,
        'B: AE-Guided Pseudo': prauc_b,
        'C: Consensus Pseudo': prauc_c,
        'D: Calibrated Voting': prauc_d,
        'E: BalancedBagging (no pseudo)': prauc_e,
        'F: BalancedBagging + pseudo': prauc_f,
    }
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- BEST" if score == max(results.values()) else ""
        print(f"  {name}: {score:.4f}{marker}")
    print("=" * 60)


def eval_cv(X, y, Xp, yp, model_type):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    praucs = []

    for tr, te in skf.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        if Xp is not None:
            Xtr = np.vstack([Xtr, Xp])
            ytr = np.concatenate([ytr, yp])

        if model_type == "VotingEnsemble":
            m = VotingClassifier(estimators=[
                ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
                ('svc', SVC(probability=True, class_weight='balanced', random_state=42)),
                ('nb', GaussianNB()),
            ], voting='soft')
        elif model_type == "CalibratedVoting":
            base = VotingClassifier(estimators=[
                ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
                ('svc', SVC(probability=True, class_weight='balanced', random_state=42)),
                ('nb', GaussianNB()),
            ], voting='soft')
            m = CalibratedClassifierCV(base, cv=3, method='isotonic')
        elif model_type == "BalancedBagging":
            from imblearn.ensemble import BalancedBaggingClassifier
            m = BalancedBaggingClassifier(
                estimator=LogisticRegression(class_weight='balanced', random_state=42),
                n_estimators=50, random_state=42, sampling_strategy='auto'
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        m.fit(Xtr, ytr)
        probs = m.predict_proba(X[te])[:, 1]
        praucs.append(average_precision_score(y[te], probs))

    return np.mean(praucs)


if __name__ == "__main__":
    run()
