"""CN7 모델 비교 실행: 평가(E1 전체 기간 / E2 불량 시기 안) × 피처(공정 값 / 공정 값 + pair_pos)

결과 (outputs/tables/)
- model_folds.csv      : fold별 PR-AUC · Recall@k% · F1 · Recall · Precision · ROC-AUC
- model_inspection.csv : 반복별 검사율@재현율 (fold 점수를 모아 불량 전체로 계산)
- model_summary.csv    : 위 두 표의 평균·표준편차
- time_increment.csv   : E2에서 샷 순서 대비 공정 값의 추가 기여 (조건부 순열검정)
실행: python run_models.py  (24코어 기준 약 30분)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import model_zoo as mz

OUT = Path(__file__).parent / 'outputs' / 'tables'
POST_HOC = ['LightGBM (튜닝)', 'XGBoost (튜닝)', 'CatBoost (튜닝)']  # 1차 결과를 보고 고른 구성 → 낙관적일 수 있음


def settings():
    ev, feats = mz.load_evalset()
    last_ng = ev.loc[ev[mz.TARGET] == 1, 'shot_id'].max()
    evals = {'E1 전체 기간': ev, f'E2 불량 시기 안 (샷 0~{last_ng})': ev[ev['shot_id'] <= last_ng].reset_index(drop=True)}
    feature_sets = {'공정 값 23': feats, '공정 값 23 + pair_pos': feats + ['pair_pos']}
    return evals, feature_sets, feats


def all_models():
    return {**{k: (m, f, False) for k, (m, f) in mz.supervised_models().items()},
            **{k: (m, f, False) for k, (m, f) in mz.tuned_models().items()},
            **{k: (m, f, True) for k, (m, f) in mz.anomaly_models().items()}}


def compare():
    evals, feature_sets, _ = settings()
    models = all_models()
    family = {k: v[1] for k, v in models.items()}
    folds, inspect = [], []
    for e_name, ev in evals.items():
        y, g = ev[mz.TARGET].values, ev['shot_id'].values
        splits = mz.make_splits(y, g)
        base, b_scores = mz.baseline_rows(ev, y, splits)
        folds.append(base.assign(eval=e_name, features='-', family='기준선'))
        inspect.append(mz.inspection_rates(b_scores, base['model'].unique(), y, splits).assign(eval=e_name, features='-', family='기준선'))
        for f_name, cols in feature_sets.items():
            t = time.time()
            X = ev[cols].values.astype(float)
            r, scores = mz.evaluate_models(models, X, y, g, splits)
            r['family'] = r['model'].map(family)
            for v_name, members in [('순위 평균 앙상블 (5종, 사전 고정)', mz.VOTING_MEMBERS), ('부스팅 튜닝 3종 평균 (사후 선택)', POST_HOC)]:
                v_rows, v_scores = mz.voting_rows(scores, members, y, splits, v_name)
                r = pd.concat([r, v_rows.assign(family='앙상블')], ignore_index=True)
                scores.update(v_scores)
            fam = r.groupby('model', sort=False)['family'].first()
            folds.append(r.assign(eval=e_name, features=f_name))
            ins = mz.inspection_rates(scores, r['model'].unique(), y, splits)
            inspect.append(ins.assign(eval=e_name, features=f_name, family=ins['model'].map(fam)))
            print(f'{e_name} / {f_name}: {time.time() - t:.0f}s', flush=True)
    folds, inspect = pd.concat(folds, ignore_index=True), pd.concat(inspect, ignore_index=True)

    keys = ['eval', 'features', 'family', 'model']
    g, gi = folds.groupby(keys, sort=False), inspect.groupby(keys, sort=False)
    summary = (g[mz.METRICS + ['threshold']].mean().add_suffix('_mean').join(g[mz.METRICS].std().add_suffix('_std'))
               .join(gi[mz.INSPECT].mean().add_suffix('_mean')).join(gi[mz.INSPECT].std().add_suffix('_std')))
    summary['failed_folds'] = g['PR-AUC'].apply(lambda s: int(s.isna().sum()))
    summary = summary.reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / 'model_folds.csv', index=False, encoding='utf-8-sig')
    inspect.to_csv(OUT / 'model_inspection.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUT / 'model_summary.csv', index=False, encoding='utf-8-sig')
    return summary


# ---------------------------------------------------------------- 샷 순서 대비 공정 값의 추가 기여 (E2)
def _cv_ap(model, X, y, splits):
    """fold별 PR-AUC (임계값 없음)"""
    out = []
    for tr, te in splits:
        m = mz._fit(model, X[tr], y[tr], False)
        out.append(mz.average_precision_score(y[te], mz._score(m, X[te], False)))
    return np.array(out)


def _permute_within_windows(ev, feats, window, rng):
    """샷을 연속 window개씩 묶고, 묶음 안에서 샷끼리 공정 값을 뒤섞는다 (같은 샷의 두 부품은 같이 이동).
    → 시간 흐름(몇 샷 단위)은 유지하고 '어느 샷이 불량인가'에 대한 공정 값 정보만 지운다"""
    shot_vals = ev.groupby('shot_id')[feats].first()
    sid = shot_vals.index.values
    new = sid.copy()
    for w0 in range(0, len(sid), window):
        idx = np.arange(w0, min(w0 + window, len(sid)))
        new[idx] = sid[rng.permutation(idx)]
    mapped = shot_vals.loc[new].set_axis(sid)
    return mapped.loc[ev['shot_id']].values


def time_increment(n_perm=200, window=10):
    evals, _, feats = settings()
    e_name, ev = list(evals.items())[1]
    y, g = ev[mz.TARGET].values, ev['shot_id'].values
    splits = mz.make_splits(y, g)
    sid = ev[['shot_id']].values.astype(float)
    sup = mz.supervised_models()
    rows = []
    for m_name in ['로지스틱 L2', 'LightGBM']:
        model = sup[m_name][0]
        ap_time = _cv_ap(model, sid, y, splits)
        ap_full = _cv_ap(model, np.column_stack([ev[feats].values, sid]), y, splits)
        null_splits = splits[:10]  # 순열검정은 계산량 때문에 2회 반복(10 fold)
        obs = ap_full[:10].mean() - ap_time[:10].mean()

        def one(seed):
            Xp = np.column_stack([_permute_within_windows(ev, feats, window, np.random.default_rng(seed)), sid])
            return _cv_ap(model, Xp, y, null_splits).mean() - ap_time[:10].mean()

        null = np.array(Parallel(n_jobs=-1)(delayed(one)(mz.SEED + i) for i in range(n_perm)))
        rows.append({'모델': m_name, 'PR-AUC 샷 순서만': ap_time.mean(), 'PR-AUC 샷 순서 + 공정 값': ap_full.mean(),
                     '추가 기여 (50 fold 평균)': ap_full.mean() - ap_time.mean(),
                     '추가 기여가 양수인 fold 비율': (ap_full > ap_time).mean(),
                     '추가 기여 (10 fold)': obs, f'순열 기준 평균 ({window}샷 창 안에서 섞음)': null.mean(),
                     '순열 기준 95% 상한': np.quantile(null, 0.95), '순열 p': (np.sum(null >= obs) + 1) / (n_perm + 1)})
        print(m_name, 'done', flush=True)
    tbl = pd.DataFrame(rows).set_index('모델')
    tbl.to_csv(OUT / 'time_increment.csv', encoding='utf-8-sig')
    return tbl


if __name__ == '__main__':
    t0 = time.time()
    compare()
    time_increment()
    print(f'total {time.time() - t0:.0f}s')
