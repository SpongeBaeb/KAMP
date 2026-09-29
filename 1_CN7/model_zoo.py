"""CN7 모델 비교: 지도학습 · 불균형 대응 · 이상탐지 · 앙상블을 같은 교차검증으로 평가한다.

- 평가 E1 (전체 기간): labeled 전체, StratifiedGroupKFold(shot_id) 5-fold × 10회 (CN7.ipynb 5-1과 같은 분할)
- 평가 E2 (불량 시기 안): 샷 0 ~ 마지막 불량 샷만 사용, 같은 분할 방식 → 시기 효과를 뺀 판정용
- 임계값(F1·Recall·Precision용)은 학습 fold 안의 3-fold 예측으로 정한다 (평가 fold 미사용)
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 42
TARGET = 'PassOrFail'
N_SPLITS, N_REPEATS = 5, 10
TOP_K = [10, 20, 30]
RECALL_TARGETS = [1.0, 0.9]
DATA = Path(__file__).parent / 'data' / 'moldset_labeled_cn7.csv'


# ---------------------------------------------------------------- 데이터
def load_evalset():
    """labeled + shot_id(샷 번호) / pair_pos(쌍 순서). CN7.ipynb 3-7과 같은 정의 (역변환은 열별 선형변환이라 표준화 후 결과에 영향 없음)"""
    lab = pd.read_csv(DATA, index_col=0)
    feats = [c for c in lab.columns if c != TARGET and lab[c].nunique() > 1]
    grp = lab.groupby(feats, sort=False)
    ev = lab[feats + [TARGET]].assign(shot_id=grp.ngroup(), pair_pos=grp.cumcount())
    return ev, feats


def make_splits(y, groups, seed=SEED, n_repeats=N_REPEATS):
    return [(tr, te) for r in range(n_repeats)
            for tr, te in StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + r).split(y, y, groups)]


# ---------------------------------------------------------------- 모델
def _scaled(model):
    return make_pipeline(StandardScaler(), model)


def supervised_models():
    """이름 → (모델, 계열). 기본값은 소표본·불균형에 맞춘 보수적 설정 (결과를 보고 고치지 않음)"""
    bal = 'balanced'
    spw = 70  # 양품/불량 비 (약 1194/17)
    m = {
        # 선형 · 확률 모델
        '로지스틱 L2': (_scaled(LogisticRegression(class_weight=bal, max_iter=5000)), '선형'),
        '로지스틱 L1': (_scaled(LogisticRegression(class_weight=bal, l1_ratio=1.0, C=0.5, solver='liblinear', max_iter=5000)), '선형'),
        '로지스틱 ElasticNet': (_scaled(LogisticRegression(class_weight=bal, l1_ratio=0.5, C=0.5, solver='saga', max_iter=20000)), '선형'),
        'LDA (shrinkage)': (_scaled(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')), '선형'),
        '나이브 베이즈': (_scaled(GaussianNB()), '확률'),
        'kNN (k=15)': (_scaled(KNeighborsClassifier(n_neighbors=15, weights='distance')), '거리'),
        'SVM 선형': (_scaled(SVC(kernel='linear', class_weight=bal, C=0.1)), '커널'),
        'SVM RBF': (_scaled(SVC(kernel='rbf', class_weight=bal, C=1.0, gamma='scale')), '커널'),
        'MLP': (_scaled(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=2000, random_state=SEED)), '신경망'),
        # 트리 · 앙상블
        '결정트리 (depth 4)': (DecisionTreeClassifier(max_depth=4, class_weight=bal, random_state=SEED), '트리'),
        '랜덤포레스트': (RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight=bal, random_state=SEED, n_jobs=1), '트리'),
        '엑스트라트리': (ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight=bal, random_state=SEED, n_jobs=1), '트리'),
        'AdaBoost': (AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED), '부스팅'),
        'GradientBoosting': (GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED), '부스팅'),
        'HistGradientBoosting': (HistGradientBoostingClassifier(class_weight=bal, learning_rate=0.05, max_iter=200, min_samples_leaf=5, random_state=SEED), '부스팅'),
        'XGBoost': (XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=spw, n_jobs=1, random_state=SEED, verbosity=0), '부스팅'),
        'LightGBM': (LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.05, min_child_samples=5, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, class_weight=bal, n_jobs=1, random_state=SEED, verbose=-1), '부스팅'),
        'CatBoost': (CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                                        thread_count=1, random_seed=SEED, verbose=0, allow_writing_files=False), '부스팅'),
        # 불균형 대응 (샘플링은 학습 fold 안에서만)
        'SMOTE + 로지스틱': (make_imb_pipeline(StandardScaler(), SMOTE(k_neighbors=3, random_state=SEED), LogisticRegression(max_iter=5000)), '불균형'),
        'ADASYN + 로지스틱': (make_imb_pipeline(StandardScaler(), ADASYN(n_neighbors=3, random_state=SEED), LogisticRegression(max_iter=5000)), '불균형'),
        'SMOTE + 랜덤포레스트': (make_imb_pipeline(SMOTE(k_neighbors=3, random_state=SEED),
                                                RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=1)), '불균형'),
        'Balanced 랜덤포레스트': (BalancedRandomForestClassifier(n_estimators=300, sampling_strategy='all', replacement=True, bootstrap=False,
                                                          random_state=SEED, n_jobs=1), '불균형'),
        'EasyEnsemble': (EasyEnsembleClassifier(n_estimators=20, random_state=SEED, n_jobs=1), '불균형'),
        'RUSBoost': (RUSBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=200, learning_rate=0.5,
                                        random_state=SEED), '불균형'),
        # 스태킹 (구성은 결과를 보기 전에 고정)
        '스태킹 (LR·SVM·RF → LR)': (GroupedStacking(
            estimators=[('lr', _scaled(LogisticRegression(class_weight=bal, max_iter=5000))),
                        ('svm', _scaled(SVC(kernel='rbf', class_weight=bal))),
                        ('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight=bal, random_state=SEED, n_jobs=1))],
            final_estimator=_scaled(LogisticRegression(class_weight=bal, max_iter=5000))), '앙상블'),
    }
    return m


def anomaly_models():
    """학습 fold의 양품만으로 학습하는 이상탐지 (점수 = 이상할수록 큼)"""
    return {
        'Isolation Forest': (IsolationForest(n_estimators=300, random_state=SEED, n_jobs=1), '이상탐지'),
        'LOF (novelty)': (_scaled(LocalOutlierFactor(n_neighbors=20, novelty=True)), '이상탐지'),
        'One-Class SVM': (_scaled(OneClassSVM(nu=0.05, gamma='scale')), '이상탐지'),
        'Elliptic Envelope': (_scaled(EllipticEnvelope(support_fraction=0.9, random_state=SEED)), '이상탐지'),
    }


N_PROCESS = 23  # 입력 앞쪽 23열 = 공정 값. 같은 샷의 두 부품은 이 23개 값이 같다


def shot_groups(X):
    """공정 값(앞 23열)이 같은 행 = 같은 샷 → 내부 CV에서 한 묶음으로 (쌍 누수 방지)"""
    return np.unique(X[:, :N_PROCESS], axis=0, return_inverse=True)[1].ravel()


class GroupedStacking(ClassifierMixin, BaseEstimator):
    """스태킹: 기본 모델의 내부 예측을 샷 단위 3-fold로 만들고, 그 예측으로 최종 로지스틱을 학습.
    sklearn StackingClassifier의 내부 CV는 샷을 묶지 않아 같은 샷의 두 부품이 갈라져 누수가 생긴다."""

    def __init__(self, estimators, final_estimator, n_splits=3, random_state=SEED):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        g = shot_groups(X)
        meta = np.zeros((len(y), len(self.estimators)))
        for tr, te in StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state).split(X, y, g):
            for j, (_, est) in enumerate(self.estimators):
                meta[te, j] = _score(clone(est).fit(X[tr], y[tr]), X[te], False)
        self.fitted_ = [clone(est).fit(X, y) for _, est in self.estimators]
        self.final_ = clone(self.final_estimator).fit(meta, y)
        return self

    def predict_proba(self, X):
        meta = np.column_stack([_score(est, X, False) for est in self.fitted_])
        return self.final_.predict_proba(meta)


class TunedModel(ClassifierMixin, BaseEstimator):
    """중첩 튜닝: 학습 데이터 안에서 샷 단위 3-fold로 후보 설정의 PR-AUC를 비교해 최고 설정으로 다시 학습.
    평가 fold는 보지 않는다 (바깥 교차검증의 학습 fold 안에서만 호출됨)."""

    def __init__(self, estimator, param_list, n_splits=3, random_state=SEED):
        self.estimator = estimator
        self.param_list = param_list
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        g = shot_groups(X)
        cv = list(StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state).split(X, y, g))
        best, best_ap = None, -1
        for params in self.param_list:
            aps = []
            for tr, te in cv:
                m = clone(self.estimator).set_params(**params).fit(X[tr], y[tr])
                aps.append(average_precision_score(y[te], _score(m, X[te], False)))
            if np.mean(aps) > best_ap:
                best, best_ap = params, np.mean(aps)
        self.best_params_ = best
        self.best_ = clone(self.estimator).set_params(**best).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.best_.predict_proba(X)


def _sample(grid, n=12):
    return list(ParameterSampler(grid, n_iter=n, random_state=SEED)) if np.prod([len(v) for v in grid.values()]) > n else list(ParameterGrid(grid))


def tuned_models():
    """1차 비교 상위 계열의 중첩 튜닝 버전 (후보 설정은 최대 12개, 고정 시드로 추출)"""
    sup = supervised_models()
    grids = {
        '로지스틱 (튜닝)': (_scaled(LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=5000)),
                        {'logisticregression__C': [0.01, 0.03, 0.1, 0.3, 1, 3], 'logisticregression__l1_ratio': [0.0, 1.0]}),
        '랜덤포레스트 (튜닝)': (sup['랜덤포레스트'][0], {'max_features': ['sqrt', 0.5, 1.0], 'min_samples_leaf': [1, 3, 5, 10], 'max_depth': [None, 6]}),
        'XGBoost (튜닝)': (sup['XGBoost'][0], {'max_depth': [2, 3, 4, 6], 'learning_rate': [0.03, 0.1], 'n_estimators': [100, 300],
                                              'min_child_weight': [1, 3], 'subsample': [0.7, 1.0]}),
        'LightGBM (튜닝)': (sup['LightGBM'][0], {'num_leaves': [4, 8, 16], 'learning_rate': [0.03, 0.1], 'n_estimators': [100, 300],
                                                'min_child_samples': [3, 5, 10]}),
        'CatBoost (튜닝)': (sup['CatBoost'][0], {'depth': [3, 4, 6], 'learning_rate': [0.03, 0.1], 'iterations': [200, 400], 'l2_leaf_reg': [1, 3, 10]}),
    }
    return {name: (TunedModel(est, _sample(grid)), '튜닝') for name, (est, grid) in grids.items()}


# 결과를 보기 전에 고정한 순위 평균 앙상블 구성 (서로 다른 계열 5개)
VOTING_MEMBERS = ['로지스틱 L2', 'SVM RBF', '랜덤포레스트', '엑스트라트리', 'CatBoost']


def _score(model, X, anomaly):
    if anomaly:
        return -model.score_samples(X)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _fit(model, X, y, anomaly):
    m = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if anomaly:
            m.fit(X[y == 0])
        else:
            m.fit(X, y)
    return m


def f1_threshold(score, y):
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return thr[np.argmax(f1)]


# ---------------------------------------------------------------- 지표
def rank_with_ties(score, rng):
    """동점은 무작위 순서: 점수 내림차순 순위를 1..n 점수로 (클수록 위험)"""
    n = len(score)
    order = np.lexsort((rng.random(n), -score))
    r = np.empty(n)
    r[order] = np.arange(n, 0, -1)
    return r, order


def fold_metrics(y, score, pred, rng):
    rs, order = rank_with_ties(score, rng)
    n, p = len(y), y.sum()
    caught = np.cumsum(y[order])
    out = {'PR-AUC': average_precision_score(y, rs)}
    for k in TOP_K:
        out[f'Recall@{k}%'] = caught[int(np.ceil(n * k / 100)) - 1] / p
    out.update({'F1': f1_score(y, pred), 'Recall': recall_score(y, pred),
                'Precision': precision_score(y, pred, zero_division=0), 'ROC-AUC': roc_auc_score(y, rs)})
    return out


def inspection_rates(scores, names, y, splits, seed=SEED):
    """검사율@재현율: fold당 불량이 3~4개라 fold 안에서는 재현율 0.9 = 1.0이 된다.
    → 반복(5개 fold = 전체 부품 한 번씩)마다 평가 fold 점수를 fold 안 백분위로 바꿔 모은 뒤, 불량 전체로 계산 (반복 10개의 평균·표준편차)"""
    rows = []
    n_rep = len(splits) // N_SPLITS
    for name in names:
        for r in range(n_rep):
            folds = range(r * N_SPLITS, (r + 1) * N_SPLITS)
            if any(scores[(name, f)][1] is None for f in folds):
                continue
            pct = np.empty(len(y))
            for f in folds:
                te, s_te = scores[(name, f)][0], scores[(name, f)][1]
                pct[te] = pd.Series(s_te).rank(pct=True).values
            rng = np.random.default_rng(seed + r)
            order = np.lexsort((rng.random(len(y)), -pct))
            caught = np.cumsum(y[order])
            rows.append({'model': name, 'repeat': r,
                         **{f'검사율@재현율{t:g}': (np.argmax(caught >= np.ceil(t * y.sum() - 1e-9)) + 1) / len(y) for t in RECALL_TARGETS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 한 fold 평가
def _eval_fold(name, model, anomaly, X, y, groups, tr, te, fold, seed):
    """학습 fold에서 모델 학습 + 내부 3-fold 예측으로 F1 최대 임계값 → 평가 fold 지표. 평가 fold 점수도 반환 (앙상블용)"""
    rng = np.random.default_rng(seed + fold)
    try:
        m = _fit(model, X[tr], y[tr], anomaly)
        s_te = _score(m, X[te], anomaly)
        # 임계값: 학습 fold 안 3-fold 예측
        oof = np.empty(len(tr))
        for itr, ite in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + fold).split(tr, y[tr], groups[tr]):
            mi = _fit(model, X[tr][itr], y[tr][itr], anomaly)
            oof[ite] = _score(mi, X[tr][ite], anomaly)
    except Exception as e:  # 학습 실패 fold는 지표 없이 기록 (결과표에 실패 횟수로 표시)
        return {'model': name, 'fold': fold, 'error': f'{type(e).__name__}: {e}'[:200]}, (name, fold, te, None, None, None)
    thr = f1_threshold(oof, y[tr])
    res = fold_metrics(y[te], s_te, s_te >= thr, rng)
    return {'model': name, 'fold': fold, 'threshold': thr, **res}, (name, fold, te, s_te, oof, thr)


def evaluate_models(models, X, y, groups, splits, n_jobs=-1, seed=SEED):
    """models: {이름: (모델, 계열, anomaly 여부)} → (fold별 지표 DataFrame, 점수 dict)"""
    tasks = [(name, mdl, anom, f, tr, te) for name, (mdl, _, anom) in models.items() for f, (tr, te) in enumerate(splits)]
    out = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_eval_fold)(name, mdl, anom, X, y, groups, tr, te, f, seed) for name, mdl, anom, f, tr, te in tasks)
    rows = pd.DataFrame([o[0] for o in out])
    scores = {(o[1][0], o[1][1]): o[1][2:] for o in out}
    return rows, scores


def baseline_rows(ev, y, splits, seed=SEED):
    """기준선: 무작위 / 쌍 순서 규칙(앞쪽 위험) / 샷 순서 (방향은 학습 fold에서 결정)"""
    rows, scores = [], {}
    front = (ev['pair_pos'].values == 0).astype(float)
    sid = ev['shot_id'].values.astype(float)
    for f, (tr, te) in enumerate(splits):
        rng = np.random.default_rng(seed + f)
        sign = np.sign(np.corrcoef(sid[tr], y[tr])[0, 1]) or 1.0
        cands = {'[기준] 무작위': rng.random(len(y)), '[기준] 쌍 순서 규칙': front, '[기준] 샷 순서': sign * sid}
        for name, s in cands.items():
            thr = f1_threshold(s[tr], y[tr])  # 점수에 학습이 없어 학습 fold 점수로 바로 결정
            rows.append({'model': name, 'fold': f, 'threshold': thr, **fold_metrics(y[te], s[te], s[te] >= thr, rng)})
            scores[(name, f)] = (te, s[te], s[tr], thr)
    return pd.DataFrame(rows), scores


def voting_rows(scores, members, y, splits, name, seed=SEED):
    """멤버 모델의 평가 fold 점수를 순위로 바꿔 평균. 임계값도 학습 fold 내부 예측(oof)의 순위 평균으로 결정"""
    rows, v_scores = [], {}
    for f, (tr, te) in enumerate(splits):
        rng = np.random.default_rng(seed + f)
        s_te = np.mean([pd.Series(scores[(m, f)][1]).rank(pct=True).values for m in members], axis=0)
        oof = np.mean([pd.Series(scores[(m, f)][2]).rank(pct=True).values for m in members], axis=0)
        # 평가 fold 점수를 학습 fold 내부 예측 분포 기준 백분위로 맞춘 뒤 임계값 적용
        te_pct = np.mean([np.searchsorted(np.sort(scores[(m, f)][2]), scores[(m, f)][1], side='right') / len(tr) for m in members], axis=0)
        thr = f1_threshold(oof, y[tr])
        rows.append({'model': name, 'fold': f, 'threshold': thr, **fold_metrics(y[te], s_te, te_pct >= thr, rng)})
        v_scores[(name, f)] = (te, s_te, oof, thr)
    return pd.DataFrame(rows), v_scores


METRICS = ['PR-AUC'] + [f'Recall@{k}%' for k in TOP_K] + ['F1', 'Recall', 'Precision', 'ROC-AUC']
INSPECT = [f'검사율@재현율{r:g}' for r in RECALL_TARGETS]


def summarize(rows):
    g = rows.groupby('model', sort=False)[METRICS]
    mean, std = g.mean(), g.std()
    return mean, std


def fmt_table(mean, std, order=None):
    t = mean.combine(std, lambda m, s: m.map('{:.3f}'.format) + ' ± ' + s.map('{:.3f}'.format))
    return t.loc[order] if order is not None else t
