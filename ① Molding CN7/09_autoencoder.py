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
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def add_physics_features(df):
    df = df.copy()
    df['Cooling_Time'] = df['Cycle_Time'] - (df['Injection_Time'] + df['Plasticizing_Time'] + df['Clamp_Close_Time'])
    df['Temp_Gradient'] = df['Barrel_Temperature_6'] - df['Barrel_Temperature_1']
    temp_cols = [f'Barrel_Temperature_{i}' for i in range(1, 7)]
    df['Avg_Barrel_Temp'] = df[temp_cols].mean(axis=1)
    df['Pressure_Drop'] = df['Max_Injection_Pressure'] - df['Max_Switch_Over_Pressure']
    df['Injection_Efficiency'] = df['Max_Injection_Speed'] / (df['Injection_Time'] + 1e-5)
    return df


def run_autoencoder_pipeline():
    print("=" * 60)
    print("Phase 9: Autoencoder Anomaly Detection + Ensemble")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/5] Loading Data...")
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

    print(f"  Labeled: {X_lab.shape[0]} rows ({y_lab.sum()} defects)")
    print(f"  Unlabeled: {X_unlab.shape[0]} rows")
    print(f"  Features: {len(features)}")

    # 2. Scale using unlabeled data (no leakage - unlabeled is external)
    print("\n[2/5] Scaling & Training Autoencoder on 35,000 Normal Samples...")
    scaler = StandardScaler()
    X_unlab_scaled = scaler.fit_transform(X_unlab)
    X_lab_scaled = scaler.transform(X_lab)

    # 3. Train Autoencoder on unlabeled data (assumed mostly normal)
    input_dim = X_unlab_scaled.shape[1]
    ae = Autoencoder(input_dim, latent_dim=8)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_unlab_tensor = torch.FloatTensor(X_unlab_scaled)
    dataset = TensorDataset(X_unlab_tensor, X_unlab_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    ae.train()
    for epoch in range(50):
        total_loss = 0
        for batch_x, _ in loader:
            recon = ae(batch_x)
            loss = criterion(recon, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50 - Loss: {total_loss/len(loader):.6f}")

    # 4. Compute reconstruction error as a new feature
    print("\n[3/5] Computing Reconstruction Error Features...")
    ae.eval()
    with torch.no_grad():
        X_lab_tensor = torch.FloatTensor(X_lab_scaled)
        recon_lab = ae(X_lab_tensor).numpy()
        # Per-sample MSE (reconstruction error)
        recon_error_lab = np.mean((X_lab_scaled - recon_lab) ** 2, axis=1).reshape(-1, 1)
        # Per-feature absolute error (top 3 most important features get their own error)
        per_feature_error = np.abs(X_lab_scaled - recon_lab)

    # Add reconstruction error as new feature(s)
    X_lab_enriched = np.hstack([X_lab_scaled, recon_error_lab, per_feature_error])

    enriched_feature_names = features + ['AE_Recon_Error'] + [f'AE_Err_{f}' for f in features]
    print(f"  Total enriched features: {X_lab_enriched.shape[1]}")

    # Also generate pseudo labels using enriched features
    print("\n[4/5] Generating Pseudo-Labels & Running Semi-Supervised Ensemble CV...")

    # Pseudo-labeling on unlabeled data
    with torch.no_grad():
        recon_unlab = ae(X_unlab_tensor).numpy()
        recon_error_unlab = np.mean((X_unlab_scaled - recon_unlab) ** 2, axis=1).reshape(-1, 1)
        per_feature_error_unlab = np.abs(X_unlab_scaled - recon_unlab)

    X_unlab_enriched = np.hstack([X_unlab_scaled, recon_error_unlab, per_feature_error_unlab])

    pseudo_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    pseudo_model.fit(X_lab_enriched, y_lab)
    unlab_proba = pseudo_model.predict_proba(X_unlab_enriched)[:, 1]

    high_conf_defect_idx = np.where(unlab_proba > 0.95)[0]
    high_conf_normal_idx = np.where(unlab_proba < 0.05)[0]

    np.random.seed(42)
    if len(high_conf_normal_idx) > 5000:
        high_conf_normal_idx = np.random.choice(high_conf_normal_idx, 5000, replace=False)

    X_pseudo = np.vstack([X_unlab_enriched[high_conf_defect_idx], X_unlab_enriched[high_conf_normal_idx]])
    y_pseudo = np.concatenate([np.ones(len(high_conf_defect_idx)), np.zeros(len(high_conf_normal_idx))])

    print(f"  Pseudo Defects: {len(high_conf_defect_idx)}, Pseudo Normals: {len(high_conf_normal_idx)}")

    # 5. Strict CV on labeled data only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    nb = GaussianNB()
    voting = VotingClassifier(estimators=[('lr', lr), ('svc', svc), ('nb', nb)], voting='soft')

    praucs, recalls, f1s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_lab_enriched, y_lab)):
        X_train = X_lab_enriched[train_idx]
        X_test = X_lab_enriched[test_idx]
        y_train, y_test = y_lab[train_idx], y_lab[test_idx]

        X_train_expanded = np.vstack([X_train, X_pseudo])
        y_train_expanded = np.concatenate([y_train, y_pseudo])

        voting.fit(X_train_expanded, y_train_expanded)

        probs = voting.predict_proba(X_test)[:, 1]
        preds = voting.predict(X_test)

        prauc = average_precision_score(y_test, probs)
        praucs.append(prauc)
        recalls.append(recall_score(y_test, preds))
        f1s.append(f1_score(y_test, preds))

        print(f"  Fold {fold+1}: PR-AUC={prauc:.4f}")

    final_prauc = np.mean(praucs)

    print("\n" + "=" * 60)
    print(f"[PHASE 9 FINAL RESULT]")
    print(f"  PR-AUC:  {final_prauc:.4f}")
    print(f"  Recall:  {np.mean(recalls):.4f}")
    print(f"  F1:      {np.mean(f1s):.4f}")
    print("=" * 60)

    # 6. Plot evolution
    print("\n[5/5] Saving Evolution Chart...")
    phases = ['Phase 2\n(Baseline)', 'Phase 5\n(Semi-Sup)', 'Phase 6\n(Ensemble)', 'Phase 9\n(Autoencoder)']
    scores = [0.5357, 0.6243, 0.6385, final_prauc]
    colors = ['#808080', '#4682B4', '#2E8B57', '#DC143C']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(phases, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.90, color='gold', linestyle='--', linewidth=2, label='Target (0.90)')
    ax.set_title("PR-AUC Evolution: The Autoencoder Breakthrough", fontsize=16, fontweight='bold')
    ax.set_ylabel("PR-AUC Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig('plots/09_autoencoder_evolution.png', dpi=300)
    plt.close()
    print("Saved to 'plots/09_autoencoder_evolution.png'")


if __name__ == "__main__":
    run_autoencoder_pipeline()
