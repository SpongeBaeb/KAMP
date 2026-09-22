"""
4차 최적화 실험: RMSE 10.0 이하 달성을 위한 최종 시도
- 실험 4-A: Huber Loss (LightGBM)
- 실험 4-B: EWM (지수 가중 이동평균) 피처 추가
- 실험 4-C: Feature Selection (상위 중요 피처만 사용)
- 실험 4-D: Stacking Meta-Learner (ExtraTrees + LightGBM → Ridge)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── 공통 데이터 준비 함수 ──
def prepare_data(add_ewm=False):
    df = pd.read_csv('preprocessed_data.csv')

    for i in range(1, 25):
        df[f'Peak_lag_{i}'] = df['Peak'].shift(i)

    for w in [2, 3, 4, 5, 6, 12, 24]:
        df[f'Peak_roll_mean_{w}'] = df['Peak'].shift(1).rolling(window=w).mean()
        df[f'Peak_roll_std_{w}'] = df['Peak'].shift(1).rolling(window=w).std()

    df['Peak_roll_max_24'] = df['Peak'].shift(1).rolling(window=24).max()
    df['Peak_roll_min_24'] = df['Peak'].shift(1).rolling(window=24).min()

    df['Peak_diff_1'] = df['Peak_lag_1'] - df['Peak_lag_2']
    df['Peak_diff_24'] = df['Peak_lag_1'] - df['Peak_lag_24']

    df['THI'] = df['기온'] - 0.55 * (1 - df['습도']/100.0) * (df['기온'] - 14.5)

    if add_ewm:
        for span in [3, 6, 12, 24]:
            df[f'Peak_ewm_{span}'] = df['Peak'].shift(1).ewm(span=span).mean()

    df = df.dropna().reset_index(drop=True)
    df = df[df['is_inrush'] == 0].reset_index(drop=True)

    y_abs = df['Peak'].copy()
    y_diff = df['Peak'] - df['Peak_lag_1']

    drop_cols = ['Peak', '날짜', '시간', '15분', '30분', '45분', '60분', '평균',
                 'is_inrush', 'Peak_MA3', 'Peak_MA6', 'Peak_MA12', 'Peak_MA24']
    X = df.drop(columns=drop_cols, errors='ignore')

    if 'day' in X.columns: X['day'] = X['day'].astype('category')
    if 'm' in X.columns: X['m'] = X['m'].astype('category')

    return X, y_abs, y_diff


def eval_cv(X, y_abs, y_diff, model_fn, name):
    """5-Fold Walk-Forward CV를 수행하고 결과를 반환."""
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores, r2_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_full = X.iloc[train_idx]
        y_train_full = y_diff.iloc[train_idx]
        X_test = X.iloc[test_idx]
        actual_abs = y_abs.iloc[test_idx]

        valid_size = max(int(len(X_train_full) * 0.1), 1)
        X_train = X_train_full.iloc[:-valid_size]
        y_train = y_train_full.iloc[:-valid_size]
        X_valid = X_train_full.iloc[-valid_size:]
        y_valid = y_train_full.iloc[-valid_size:]

        preds_diff = model_fn(X_train, y_train, X_valid, y_valid, X_test)
        preds_abs = X_test['Peak_lag_1'].values + preds_diff

        rmse = np.sqrt(mean_squared_error(actual_abs, preds_abs))
        r2 = r2_score(actual_abs, preds_abs)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        print(f"  Fold {fold+1}: RMSE={rmse:.4f}, R²={r2:.4f}")

    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)
    print(f"  ▶ {name} 평균 RMSE: {avg_rmse:.4f}, 평균 R²: {avg_r2:.4f}\n")
    return avg_rmse, avg_r2


# ── 실험 4-A: Huber Loss ──
def run_huber():
    print("=" * 60)
    print("[실험 4-A] LightGBM + Huber Loss (차분화 타겟)")
    print("=" * 60)
    X, y_abs, y_diff = prepare_data(add_ewm=False)

    def model_fn(X_tr, y_tr, X_val, y_val, X_te):
        m = lgb.LGBMRegressor(
            objective='huber', alpha=0.9,  # huber delta
            n_estimators=1000, learning_rate=0.036, num_leaves=72,
            max_depth=14, min_data_in_leaf=11, feature_fraction=0.62,
            bagging_fraction=0.89, bagging_freq=3,
            lambda_l1=0.0009, lambda_l2=1.9e-05,
            verbosity=-1, random_state=42
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.predict(X_te)

    return eval_cv(X, y_abs, y_diff, model_fn, "Huber Loss")


# ── 실험 4-B: EWM 피처 추가 + ExtraTrees ──
def run_ewm():
    print("=" * 60)
    print("[실험 4-B] ExtraTrees + EWM 피처 (차분화 타겟)")
    print("=" * 60)
    X, y_abs, y_diff = prepare_data(add_ewm=True)

    # ExtraTrees는 category를 지원하지 않으므로 int로 변환
    X_num = X.copy()
    for c in X_num.select_dtypes(['category']).columns:
        X_num[c] = X_num[c].cat.codes

    def model_fn(X_tr, y_tr, X_val, y_val, X_te):
        m = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1, max_depth=15)
        # ExtraTrees는 early stopping이 없으므로 train+valid 합쳐서 학습
        X_full = pd.concat([X_tr, X_val])
        y_full = pd.concat([y_tr, y_val])
        m.fit(X_full, y_full)
        return m.predict(X_te)

    return eval_cv(X_num, y_abs, y_diff, model_fn, "EWM + ExtraTrees")


# ── 실험 4-C: Feature Selection ──
def run_feature_selection():
    print("=" * 60)
    print("[실험 4-C] Feature Selection (상위 20개 피처)")
    print("=" * 60)
    X, y_abs, y_diff = prepare_data(add_ewm=True)

    X_num = X.copy()
    for c in X_num.select_dtypes(['category']).columns:
        X_num[c] = X_num[c].cat.codes

    # 전체 데이터로 피처 중요도 추출
    selector = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    selector.fit(X_num, y_diff)
    importances = pd.Series(selector.feature_importances_, index=X_num.columns)
    top_features = importances.nlargest(20).index.tolist()
    print(f"  선택된 피처: {top_features}")

    X_sel = X_num[top_features]

    def model_fn(X_tr, y_tr, X_val, y_val, X_te):
        m = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1, max_depth=15)
        X_full = pd.concat([X_tr, X_val])
        y_full = pd.concat([y_tr, y_val])
        m.fit(X_full, y_full)
        return m.predict(X_te)

    return eval_cv(X_sel, y_abs, y_diff, model_fn, "Feature Selection")


# ── 실험 4-D: Stacking Meta-Learner ──
def run_stacking():
    print("=" * 60)
    print("[실험 4-D] Stacking (ExtraTrees + LightGBM → Ridge)")
    print("=" * 60)
    X, y_abs, y_diff = prepare_data(add_ewm=True)

    X_num = X.copy()
    for c in X_num.select_dtypes(['category']).columns:
        X_num[c] = X_num[c].cat.codes

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores, r2_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_full = X.iloc[train_idx]
        X_train_full_num = X_num.iloc[train_idx]
        y_train_full = y_diff.iloc[train_idx]
        X_test = X.iloc[test_idx]
        X_test_num = X_num.iloc[test_idx]
        actual_abs = y_abs.iloc[test_idx]

        valid_size = max(int(len(X_train_full) * 0.1), 1)

        # Split: train / valid(for early stopping) / meta-train(for stacking)
        X_tr = X_train_full.iloc[:-valid_size]
        X_tr_num = X_train_full_num.iloc[:-valid_size]
        y_tr = y_train_full.iloc[:-valid_size]
        X_val = X_train_full.iloc[-valid_size:]
        X_val_num = X_train_full_num.iloc[-valid_size:]
        y_val = y_train_full.iloc[-valid_size:]

        # Base model 1: LightGBM
        m_lgb = lgb.LGBMRegressor(
            objective='huber', alpha=0.9,
            n_estimators=1000, learning_rate=0.036, num_leaves=72,
            max_depth=14, min_data_in_leaf=11, feature_fraction=0.62,
            bagging_fraction=0.89, bagging_freq=3,
            lambda_l1=0.0009, lambda_l2=1.9e-05,
            verbosity=-1, random_state=42
        )
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        # Base model 2: ExtraTrees
        m_et = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1, max_depth=15)
        m_et.fit(X_tr_num, y_tr)

        # Generate meta-features on validation set
        meta_val = np.column_stack([
            m_lgb.predict(X_val),
            m_et.predict(X_val_num)
        ])

        # Train Ridge meta-learner on validation set predictions
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_val, y_val)

        # Generate meta-features on test set
        meta_test = np.column_stack([
            m_lgb.predict(X_test),
            m_et.predict(X_test_num)
        ])

        preds_diff = meta_model.predict(meta_test)
        preds_abs = X_test['Peak_lag_1'].values + preds_diff

        rmse = np.sqrt(mean_squared_error(actual_abs, preds_abs))
        r2 = r2_score(actual_abs, preds_abs)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        print(f"  Fold {fold+1}: RMSE={rmse:.4f}, R²={r2:.4f}")

    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)
    print(f"  ▶ Stacking 평균 RMSE: {avg_rmse:.4f}, 평균 R²: {avg_r2:.4f}\n")
    return avg_rmse, avg_r2


# ── 전체 실행 ──
if __name__ == "__main__":
    results = {}

    rmse_a, r2_a = run_huber()
    results['4-A Huber'] = (rmse_a, r2_a)

    rmse_b, r2_b = run_ewm()
    results['4-B EWM+ET'] = (rmse_b, r2_b)

    rmse_c, r2_c = run_feature_selection()
    results['4-C FeatSel'] = (rmse_c, r2_c)

    rmse_d, r2_d = run_stacking()
    results['4-D Stacking'] = (rmse_d, r2_d)

    print("=" * 60)
    print("최종 결과 요약")
    print("=" * 60)
    for name, (rmse, r2) in results.items():
        marker = "✅" if rmse < 10.0 else "❌"
        print(f"  {marker} {name}: RMSE={rmse:.4f}, R²={r2:.4f}")

    best = min(results, key=lambda k: results[k][0])
    print(f"\n  🏆 최고 성능: {best} → RMSE={results[best][0]:.4f}")
