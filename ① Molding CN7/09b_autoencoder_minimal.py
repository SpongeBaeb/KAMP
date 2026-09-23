import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def add_physics_features(df):
    df = df.copy()
    df['Cooling_Time'] = df['Cycle_Time'] - (df['Injection_Time'] + df['Plasticizing_Time'] + df['Clamp_Close_Time'])
    df['Temp_Gradient'] = df['Barrel_Temperature_6'] - df['Barrel_Temperature_1']
    temp_cols = [f'Barrel_Temperature_{i}' for i in range(1, 7)]
    df['Avg_Barrel_Temp'] = df[temp_cols].mean(axis=1)
    df['Pressure_Drop'] = df['Max_Injection_Pressure'] - df['Max_Switch_Over_Pressure']
    df['Injection_Efficiency'] = df['Max_Injection_Speed'] / (df['Injection_Time'] + 1e-5)
    return df


def run():
    print("=" * 60)
    print("Phase 9b: Autoencoder (Single Recon Error Feature)")
    print("=" * 60)

    df_lab = pd.read_csv('data/moldset_labeled_cn7.csv')
    df_unlab = pd.read_csv('data/moldset_unlabeled_cn7.csv')

    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df_lab = df_lab.drop(columns=[col for col in cols_to_drop if col in df_lab.columns])
    df_unlab = df_unlab.drop(columns=[col for col in cols_to_drop if col in df_unlab.columns])

    df_lab = add_physics_features(df_lab)
    df_unlab = add_physics_features(df_unlab)

    target = 'PassOrFail'
    features = [c for c in df_lab.columns if c != target]

    X_lab = df_lab[features].values
    y_lab = df_lab[target].values
    X_unlab = df_unlab[features].values

    # Scale on unlabeled (external, no leakage)
    scaler = StandardScaler()
    X_unlab_sc = scaler.fit_transform(X_unlab)
    X_lab_sc = scaler.transform(X_lab)

    # Train Autoencoder on unlabeled
    print("\nTraining Autoencoder...")
    ae = Autoencoder(X_unlab_sc.shape[1])
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    crit = nn.MSELoss(reduction='none')

    t_unlab = torch.FloatTensor(X_unlab_sc)
    loader = DataLoader(TensorDataset(t_unlab, t_unlab), batch_size=256, shuffle=True)

    ae.train()
    for epoch in range(100):
        for bx, _ in loader:
            loss = crit(ae(bx), bx).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/100 - Loss: {loss.item():.6f}")

    # Compute SINGLE reconstruction error
    ae.eval()
    with torch.no_grad():
        t_lab = torch.FloatTensor(X_lab_sc)
        recon_lab = ae(t_lab).numpy()
        re_lab = np.mean((X_lab_sc - recon_lab) ** 2, axis=1).reshape(-1, 1)

        recon_unlab = ae(t_unlab).numpy()
        re_unlab = np.mean((X_unlab_sc - recon_unlab) ** 2, axis=1).reshape(-1, 1)

    # Sanity check: do defects have higher recon error?
    defect_re = re_lab[y_lab == 1].mean()
    normal_re = re_lab[y_lab == 0].mean()
    print(f"\nRecon Error - Defects: {defect_re:.6f}, Normals: {normal_re:.6f}, Ratio: {defect_re/normal_re:.2f}x")

    # Add ONLY the single recon error to original features
    X_lab_final = np.hstack([X_lab_sc, re_lab])
    X_unlab_final = np.hstack([X_unlab_sc, re_unlab])

    print(f"Final feature count: {X_lab_final.shape[1]} (original {len(features)} + 1 AE_Error)")

    # Pseudo-labeling
    pseudo_lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    pseudo_lr.fit(X_lab_final, y_lab)
    unlab_proba = pseudo_lr.predict_proba(X_unlab_final)[:, 1]

    hi_def = np.where(unlab_proba > 0.95)[0]
    hi_nor = np.where(unlab_proba < 0.05)[0]
    np.random.seed(42)
    if len(hi_nor) > 5000:
        hi_nor = np.random.choice(hi_nor, 5000, replace=False)

    X_pseudo = np.vstack([X_unlab_final[hi_def], X_unlab_final[hi_nor]])
    y_pseudo = np.concatenate([np.ones(len(hi_def)), np.zeros(len(hi_nor))])
    print(f"Pseudo Defects: {len(hi_def)}, Pseudo Normals: {len(hi_nor)}")

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    voting = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)),
            ('svc', SVC(probability=True, class_weight='balanced', random_state=42)),
            ('nb', GaussianNB()),
        ], voting='soft')

    praucs = []
    for fold, (tr, te) in enumerate(skf.split(X_lab_final, y_lab)):
        Xtr = np.vstack([X_lab_final[tr], X_pseudo])
        ytr = np.concatenate([y_lab[tr], y_pseudo])

        voting.fit(Xtr, ytr)
        probs = voting.predict_proba(X_lab_final[te])[:, 1]
        p = average_precision_score(y_lab[te], probs)
        praucs.append(p)
        print(f"  Fold {fold+1}: PR-AUC={p:.4f}")

    final = np.mean(praucs)
    print(f"\n{'='*60}")
    print(f"PHASE 9b RESULT: PR-AUC = {final:.4f}")
    print(f"{'='*60}")

    # Also try: pure reconstruction error as anomaly score (no supervised model)
    print("\n--- Bonus: Pure Unsupervised (Recon Error Only) ---")
    pure_prauc = average_precision_score(y_lab, re_lab.ravel())
    print(f"Pure AE Recon Error PR-AUC: {pure_prauc:.4f}")

    # Plot
    phases = ['Phase 2\nBaseline', 'Phase 5\nSemi-Sup', 'Phase 6\nEnsemble', 'Phase 9b\nAutoencoder']
    scores = [0.5357, 0.6243, 0.6385, final]
    colors = ['#808080', '#4682B4', '#2E8B57', '#DC143C']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(phases, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title("PR-AUC Evolution", fontsize=16, fontweight='bold')
    ax.set_ylabel("PR-AUC", fontsize=12)
    ax.set_ylim(0, 1.05)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/09b_autoencoder_minimal.png', dpi=300)
    plt.close()
    print("\nSaved plot to 'plots/09b_autoencoder_minimal.png'")


if __name__ == "__main__":
    run()
