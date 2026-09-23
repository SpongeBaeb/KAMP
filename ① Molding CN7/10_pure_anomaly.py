import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, d))
    def forward(self, x):
        return self.decoder(self.encoder(x))


def run():
    print("=" * 60)
    print("Phase 10: Pure Anomaly Detection (Learn Normal Only)")
    print("=" * 60)

    df = pd.read_csv('data/moldset_labeled_cn7.csv')
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    target = 'PassOrFail'
    features = [c for c in df.columns if c != target]
    X = df[features].values
    y = df[target].values

    print(f"Total: {len(y)}, Defects: {y.sum()}, Normals: {(y==0).sum()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We test 4 unsupervised anomaly detectors, each trained ONLY on normals in the train fold
    methods = {
        'Autoencoder': [],
        'OneClassSVM': [],
        'IsolationForest': [],
        'Mahalanobis': [],
    }

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_train_all, y_train = X[tr], y[tr]
        X_test, y_test = X[te], y[te]

        # ONLY normals for training (guaranteed clean)
        X_train_normal = X_train_all[y_train == 0]

        scaler = StandardScaler()
        X_train_normal_sc = scaler.fit_transform(X_train_normal)
        X_test_sc = scaler.transform(X_test)

        # --- Autoencoder ---
        ae = Autoencoder(X_train_normal_sc.shape[1])
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        t_normal = torch.FloatTensor(X_train_normal_sc)
        loader = DataLoader(TensorDataset(t_normal, t_normal), batch_size=64, shuffle=True)
        ae.train()
        for epoch in range(200):
            for bx, _ in loader:
                loss = nn.MSELoss()(ae(bx), bx)
                opt.zero_grad(); loss.backward(); opt.step()
        ae.eval()
        with torch.no_grad():
            t_test = torch.FloatTensor(X_test_sc)
            re = np.mean((X_test_sc - ae(t_test).numpy()) ** 2, axis=1)
        methods['Autoencoder'].append(average_precision_score(y_test, re))

        # --- One-Class SVM ---
        ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.02)
        ocsvm.fit(X_train_normal_sc)
        # decision_function: negative = anomaly, so negate it
        ocsvm_score = -ocsvm.decision_function(X_test_sc)
        methods['OneClassSVM'].append(average_precision_score(y_test, ocsvm_score))

        # --- Isolation Forest ---
        iso = IsolationForest(random_state=42, contamination=0.015, n_estimators=300)
        iso.fit(X_train_normal_sc)
        iso_score = -iso.decision_function(X_test_sc)
        methods['IsolationForest'].append(average_precision_score(y_test, iso_score))

        # --- Mahalanobis Distance ---
        try:
            ee = EllipticEnvelope(contamination=0.015, random_state=42)
            ee.fit(X_train_normal_sc)
            mah_score = -ee.decision_function(X_test_sc)
            methods['Mahalanobis'].append(average_precision_score(y_test, mah_score))
        except Exception:
            methods['Mahalanobis'].append(0.0)

        print(f"Fold {fold+1}: AE={methods['Autoencoder'][-1]:.4f}, "
              f"OCSVM={methods['OneClassSVM'][-1]:.4f}, "
              f"IF={methods['IsolationForest'][-1]:.4f}, "
              f"Mah={methods['Mahalanobis'][-1]:.4f}")

    print("\n" + "=" * 60)
    print("PURE ANOMALY DETECTION RESULTS (Trained on Normals Only)")
    print("=" * 60)
    for name, scores in sorted(methods.items(), key=lambda x: np.mean(x[1]), reverse=True):
        m = np.mean(scores)
        s = np.std(scores)
        marker = " <-- BEST" if m == max(np.mean(v) for v in methods.values()) else ""
        print(f"  {name:20s}: {m:.4f} (+/- {s:.4f}){marker}")

    best_name = max(methods, key=lambda k: np.mean(methods[k]))
    best_score = np.mean(methods[best_name])

    # Compare with supervised champion
    print(f"\n  Supervised Champion (Phase 6): 0.6385")
    if best_score > 0.6385:
        print(f"  >>> NEW CHAMPION: {best_name} at {best_score:.4f}! <<<")
    else:
        print(f"  Best unsupervised ({best_name}: {best_score:.4f}) vs supervised (0.6385)")

    # --- Bonus: Fusion (supervised + unsupervised) ---
    print("\n" + "=" * 60)
    print("FUSION: Combine Supervised + Unsupervised Scores")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import VotingClassifier

    # Rebuild pseudo labels (same as Phase 6)
    df_unlab = pd.read_csv('data/moldset_unlabeled_cn7.csv')
    df_unlab = df_unlab.drop(columns=[c for c in cols_to_drop if c in df_unlab.columns])
    X_unlab = df_unlab[features].values

    fusion_praucs = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_train_all, y_train_all = X[tr], y[tr]
        X_test, y_test = X[te], y[te]
        X_train_normal = X_train_all[y_train_all == 0]

        scaler = StandardScaler()
        scaler.fit(X_train_normal)
        X_train_sc = scaler.transform(X_train_all)
        X_test_sc = scaler.transform(X_test)
        X_unlab_sc = scaler.transform(X_unlab)
        X_normal_sc = scaler.transform(X_train_normal)

        # Supervised: Voting Ensemble with pseudo labels
        pseudo_scaler = StandardScaler()
        pseudo_scaler.fit(X_train_all)
        X_train_ps = pseudo_scaler.transform(X_train_all)
        X_unlab_ps = pseudo_scaler.transform(X_unlab)

        pseudo_lr = LogisticRegression(class_weight='balanced', random_state=42)
        pseudo_lr.fit(X_train_ps, y_train_all)
        p_unlab = pseudo_lr.predict_proba(X_unlab_ps)[:, 1]

        hi_def = np.where(p_unlab > 0.95)[0]
        hi_nor = np.where(p_unlab < 0.05)[0]
        np.random.seed(42)
        if len(hi_nor) > 5000: hi_nor = np.random.choice(hi_nor, 5000, replace=False)
        Xp = np.vstack([X_unlab_ps[hi_def], X_unlab_ps[hi_nor]])
        yp = np.concatenate([np.ones(len(hi_def)), np.zeros(len(hi_nor))])

        X_expanded = np.vstack([X_train_ps, Xp])
        y_expanded = np.concatenate([y_train_all, yp])

        voting = VotingClassifier(estimators=[
            ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
            ('svc', SVC(probability=True, class_weight='balanced', random_state=42)),
            ('nb', GaussianNB()),
        ], voting='soft')
        voting.fit(X_expanded, y_expanded)
        X_test_ps = pseudo_scaler.transform(X_test)
        sup_proba = voting.predict_proba(X_test_ps)[:, 1]

        # Unsupervised: Autoencoder
        ae = Autoencoder(X_normal_sc.shape[1])
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        t_n = torch.FloatTensor(X_normal_sc)
        loader = DataLoader(TensorDataset(t_n, t_n), batch_size=64, shuffle=True)
        ae.train()
        for _ in range(200):
            for bx, _ in loader:
                loss = nn.MSELoss()(ae(bx), bx)
                opt.zero_grad(); loss.backward(); opt.step()
        ae.eval()
        with torch.no_grad():
            re = np.mean((X_test_sc - ae(torch.FloatTensor(X_test_sc)).numpy()) ** 2, axis=1)

        # Normalize recon error to [0, 1]
        re_norm = (re - re.min()) / (re.max() - re.min() + 1e-10)

        # Fusion: weighted average
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            fused = alpha * sup_proba + (1 - alpha) * re_norm
            prauc = average_precision_score(y_test, fused)
            if fold == 0:
                if not hasattr(run, 'fusion_scores'):
                    run.fusion_scores = {a: [] for a in [0.3, 0.5, 0.7, 0.9]}
                run.fusion_scores[alpha] = [prauc]
            else:
                run.fusion_scores[alpha].append(prauc)

    print("\nFusion Results (alpha * Supervised + (1-alpha) * AE_Error):")
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        m = np.mean(run.fusion_scores[alpha])
        print(f"  alpha={alpha}: PR-AUC = {m:.4f}")


if __name__ == "__main__":
    run()
