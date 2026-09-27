"""모델 정의, fold 내부 오버샘플링, fold 내부 임계값 탐색."""
import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SEED = 42
ALGOS = ("LogReg", "RF", "HGB")
IMBALANCE = ("class_weight", "ROS", "SMOTE")
RECALL_TARGET = 0.8


# ---------------------------------------------------------------- 오버샘플링
def random_oversample(X, y, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    idx = np.r_[np.arange(len(y)), extra]
    return X[idx], y[idx]


def smote(X, y, rng, k=5):
    pos = X[y == 1]
    n_new = int((y == 0).sum() - len(pos))
    k = min(k, len(pos) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pos)
    neigh = nn.kneighbors(pos, return_distance=False)[:, 1:]
    base = rng.integers(0, len(pos), n_new)
    mate = neigh[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    synth = pos[base] + gap * (pos[mate] - pos[base])
    return np.vstack([X, synth]), np.r_[y, np.ones(n_new, dtype=y.dtype)]


class Resampled(ClassifierMixin, BaseEstimator):
    """fit 시점에만 학습 데이터를 오버샘플링한다. Pipeline 안에서 쓰면 CV fold 내부에서만 적용된다."""

    def __init__(self, estimator=None, method="ROS", random_state=SEED):
        self.estimator = estimator
        self.method = method
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        sampler = smote if self.method == "SMOTE" else random_oversample
        Xr, yr = sampler(X, y, rng)
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


# ---------------------------------------------------------------- 모델
def base_estimator(algo, balanced):
    cw = "balanced" if balanced else None
    if algo == "LogReg":
        return LogisticRegression(class_weight=cw, C=1.0, max_iter=5000, random_state=SEED)
    if algo == "RF":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=SEED, n_jobs=1)
    if algo == "HGB":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=3,
                                              class_weight=cw, early_stopping=False, random_state=SEED)
    raise ValueError(algo)


def make_model(algo, imbalance):
    """상수 변수 제거 → (LogReg만) 표준화 → 불균형 대응 → 분류기. 모두 fold 내부에서 fit된다."""
    est = base_estimator(algo, imbalance == "class_weight")
    if imbalance != "class_weight":
        est = Resampled(est, imbalance)
    steps = [("drop_const", VarianceThreshold(0.0))]
    if algo == "LogReg":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", est))
    return Pipeline(steps)


class RedundancyFilter(TransformerMixin, BaseEstimator):
    """학습 fold 기준으로 상수·준상수(최빈값 비율 ≥ top_ratio) 변수와,
    |Spearman ρ| ≥ max_corr 인 쌍의 뒤쪽 변수를 제거한다. 라벨은 쓰지 않는다."""

    def __init__(self, top_ratio=0.95, max_corr=0.95):
        self.top_ratio = top_ratio
        self.max_corr = max_corr

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        keep = []
        for j in range(X.shape[1]):
            _, cnt = np.unique(X[:, j], return_counts=True)
            if len(cnt) > 1 and cnt.max() / len(X) < self.top_ratio:
                keep.append(j)
        R = np.abs(np.corrcoef(np.apply_along_axis(rankdata, 0, X[:, keep]), rowvar=False)) if len(keep) > 1 \
            else np.zeros((len(keep), len(keep)))
        chosen = []
        for a in range(len(keep)):
            if all(R[a, b] < self.max_corr for b in chosen):
                chosen.append(a)
        self.support_ = np.array([keep[a] for a in chosen], dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


def make_model_fs(algo):
    """변수 세트 비교용: fold 내부 중복·준상수 제거 → (LogReg만) 표준화 → class_weight 분류기."""
    steps = [("filter", RedundancyFilter())]
    if algo == "LogReg":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", base_estimator(algo, True)))
    return Pipeline(steps)


# ---------------------------------------------------------------- 임계값
def select_thresholds(y, p, recall_target=RECALL_TARGET):
    """F1 최대 임계값과, Recall ≥ 목표를 만족하는 가장 높은 임계값."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    ok = rec >= recall_target
    return {"F1max": float(thr[np.argmax(f1)]),
            "Recall80": float(thr[ok].max()) if ok.any() else float(thr.min())}


COST_RATIOS = (5, 10, 20)
NO_ALARM = 1.0 + 1e-9  # 이 임계값이면 아무것도 경보하지 않음


def select_cost_thresholds(y, p, ratios=COST_RATIOS):
    """FN 1건 비용 = ratio, FP 1건 비용 = 1 일 때 총비용이 최소인 임계값 (동률이면 Recall이 높은 쪽)."""
    cand = np.r_[np.unique(p), NO_ALARM]
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    fn = np.searchsorted(pos, cand, side="left")
    fp = len(neg) - np.searchsorted(neg, cand, side="left")
    return {f"Cost{r}": float(cand[np.argmin(r * fn + fp)]) for r in ratios}


def inner_oof(model, X, y, n_splits=3):
    n_splits = int(min(n_splits, y.sum(), (y == 0).sum()))
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(X, y):
        oof[va] = clone(model).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return oof


def fit_with_thresholds(model, X, y, cost=False):
    """학습 fold 안에서 inner CV로 임계값을 정한 뒤, 학습 fold 전체로 최종 적합."""
    oof = inner_oof(model, X, y)
    thr = select_thresholds(y, oof)
    if cost:
        thr.update(select_cost_thresholds(y, oof))
    return clone(model).fit(X, y), thr


# ---------------------------------------------------------------- 이상탐지 후보
ANOMALY = ("IsolationForest", "Mahalanobis", "OneClassSVM", "LOF")


class NoveltyDetector(ClassifierMixin, BaseEstimator):
    """정상(y=0) 샷만으로 학습하는 이상탐지 모델. 불량 라벨은 fit에 쓰지 않는다.

    상수 변수 제거·표준화도 학습 정상 샷으로만 적합한다.
    predict_proba의 불량 열 = 이상 점수를 학습 정상 샷의 중앙값·IQR로 척도화한 뒤 시그모이드
    (확률이 아니라 0~1 이상 정도 점수, 순위는 원래 이상 점수와 동일).
    """

    def __init__(self, method="IsolationForest"):
        self.method = method

    def _raw(self, Z):
        if self.method == "Mahalanobis":
            return self.model_.mahalanobis(Z)
        if self.method == "OneClassSVM":
            return -self.model_.decision_function(Z)
        return -self.model_.score_samples(Z)  # IsolationForest, LOF

    def fit(self, X, y):
        X0 = np.asarray(X, dtype=float)[np.asarray(y) == 0]
        self.prep_ = Pipeline([("drop_const", VarianceThreshold(0.0)), ("scale", StandardScaler())]).fit(X0)
        Z = self.prep_.transform(X0)
        if self.method == "IsolationForest":
            self.model_ = IsolationForest(n_estimators=300, random_state=SEED).fit(Z)
        elif self.method == "Mahalanobis":
            self.model_ = LedoitWolf().fit(Z)
        elif self.method == "OneClassSVM":
            self.model_ = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(Z)
        elif self.method == "LOF":
            self.model_ = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Z)
        else:
            raise ValueError(self.method)
        s = self._raw(Z)
        q1, self.center_, q3 = np.percentile(s, [25, 50, 75])
        self.scale_ = (q3 - q1) or (s.std() or 1.0)
        self.n_train_normals_ = len(X0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = (self._raw(self.prep_.transform(np.asarray(X, dtype=float))) - self.center_) / self.scale_
        p = 1 / (1 + np.exp(-s))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------- Phase 2-B 후보
class SoftVote(ClassifierMixin, BaseEstimator):
    """구성 모델의 불량 확률 평균 (소프트보팅)."""

    def __init__(self, estimators=None):
        self.estimators = estimators

    def fit(self, X, y):
        self.estimators_ = [clone(e).fit(X, y) for e in self.estimators]
        self.classes_ = self.estimators_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SelfTraining(ClassifierMixin, BaseEstimator):
    """비라벨 풀을 학습 데이터 기준으로 순위 변환(quantile mapping)한 뒤 self-training.

    unlabeled_u: 비라벨 행의 변수별 분위(0~1). 전체 비라벨 배치 안에서 계산해 두고,
    fit 때마다 그 fold의 학습 데이터 분위값으로 옮기므로 변환도 fold 내부에서 이뤄진다.
    반복마다 확률 최상위 n_pos개를 불량, 최하위 n_neg개를 양품으로 pseudo-label 한다.
    """

    def __init__(self, estimator=None, unlabeled_u=None, n_iter=3, n_pos=5, n_neg=100):
        self.estimator = estimator
        self.unlabeled_u = unlabeled_u
        self.n_iter = n_iter
        self.n_pos = n_pos
        self.n_neg = n_neg

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        U = np.column_stack([np.quantile(X[:, j], self.unlabeled_u[:, j]) for j in range(X.shape[1])])
        remain, Xa, ya = np.arange(len(U)), X, y
        for _ in range(self.n_iter):
            p = clone(self.estimator).fit(Xa, ya).predict_proba(U[remain])[:, 1]
            order = np.argsort(p, kind="stable")
            neg, pos = remain[order[:self.n_neg]], remain[order[-self.n_pos:]]
            Xa = np.vstack([Xa, U[pos], U[neg]])
            ya = np.r_[ya, np.ones(len(pos), dtype=y.dtype), np.zeros(len(neg), dtype=y.dtype)]
            remain = np.setdiff1d(remain, np.r_[pos, neg])
        self.estimator_ = clone(self.estimator).fit(Xa, ya)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)
