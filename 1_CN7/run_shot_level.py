"""샷 단위 재평가: "공정 값이 불량 샷을 구분하는가"를 부품 위치(pair_pos)와 분리해 직접 본다.

- 한 샷의 두 부품은 공정 값 23개가 같으므로, 공정 값으로 답할 수 있는 질문은 샷 단위뿐이다.
- 샷 라벨 = 두 부품 중 하나라도 불량이면 1. 입력 = 공정 값 23개 (pair_pos 없음)
- 분할·지표·임계값 규칙은 1차(run_models.py)와 같다: 5-fold × 10회, 샷이 한 행이라 그룹 = 샷
- 모델은 결과를 보기 전에 고정: 1차 보고서에 쓴 계열 5종 + Isolation Forest. 하이퍼파라미터도 1차 그대로
  (XGBoost scale_pos_weight는 부품 단위 비율 그대로 둠 → 샷 단위에서는 약간 과대 가중)
- 추가 기여 검정: 샷 순서 대비 공정 값의 기여를 1차와 같은 조건부 순열(10샷 창)로 본다

결과 (outputs/tables/)
- shot_level_folds.csv, shot_level_summary.csv, shot_level_increment.csv
실행: python run_shot_level.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import model_zoo as mz
from run_models import _cv_ap, _permute_within_windows

OUT = Path(__file__).parent / 'outputs' / 'tables'
MODELS = ['로지스틱 L2', '랜덤포레스트', 'XGBoost', 'LightGBM', 'CatBoost']


def shot_table():
    """부품 단위 평가셋 → 샷 단위 (공정 값은 두 부품이 같으므로 첫 행, 라벨은 최대값)"""
    ev, feats = mz.load_evalset()
    shots = ev.groupby('shot_id', sort=True).agg({**{f: 'first' for f in feats}, mz.TARGET: 'max'}).reset_index()
    shots['pair_pos'] = 0  # 기준선 함수 호환용. 샷 단위에는 쌍 순서가 없다
    return shots, feats


def settings():
    shots, feats = shot_table()
    evals = {'S1 전체 기간 (샷 단위)': shots}
    if Path(__file__).parent.name == '1_CN7':  # CN7은 불량이 한 시기에 몰려 있어, 시기 효과를 뺀 판정용 평가를 둔다
        last_ng = shots.loc[shots[mz.TARGET] == 1, 'shot_id'].max()
        evals[f'S2 불량 시기 안 (샷 0~{last_ng}, 샷 단위)'] = shots[shots['shot_id'] <= last_ng].reset_index(drop=True)
    return evals, feats


def models():
    sup = mz.supervised_models()
    m = {k: (sup[k][0], sup[k][1], False) for k in MODELS}
    m['Isolation Forest'] = (mz.anomaly_models()['Isolation Forest'][0], '이상탐지', True)
    return m


def compare():
    evals, feats = settings()
    mdl = models()
    folds, inspect = [], []
    for e_name, ev in evals.items():
        y, g = ev[mz.TARGET].values, ev['shot_id'].values
        splits = mz.make_splits(y, g)
        base, b_scores = mz.baseline_rows(ev, y, splits)
        base = base[base['model'] != '[기준] 쌍 순서 규칙']
        names = base['model'].unique()
        folds.append(base.assign(eval=e_name))
        inspect.append(mz.inspection_rates(b_scores, names, y, splits).assign(eval=e_name))
        r, scores = mz.evaluate_models(mdl, ev[feats].values.astype(float), y, g, splits)
        folds.append(r.assign(eval=e_name))
        inspect.append(mz.inspection_rates(scores, r['model'].unique(), y, splits).assign(eval=e_name))
        print(f'{e_name}: 샷 {len(ev)}, 불량 샷 {y.sum()}', flush=True)
    folds, inspect = pd.concat(folds, ignore_index=True), pd.concat(inspect, ignore_index=True)
    keys = ['eval', 'model']
    g, gi = folds.groupby(keys, sort=False), inspect.groupby(keys, sort=False)
    summary = (g[mz.METRICS].mean().add_suffix('_mean').join(g[mz.METRICS].std().add_suffix('_std'))
               .join(gi[mz.INSPECT].mean().add_suffix('_mean')).join(gi[mz.INSPECT].std().add_suffix('_std')))
    summary['failed_folds'] = g['PR-AUC'].apply(lambda s: int(s.isna().sum()))
    summary = summary.reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / 'shot_level_folds.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUT / 'shot_level_summary.csv', index=False, encoding='utf-8-sig')
    return summary


def increment(n_perm=200, window=10):
    """판정용 평가(CN7: S2, RG3: S1)에서 샷 순서 대비 공정 값의 추가 기여 (run_models.time_increment와 같은 방식)"""
    evals, feats = settings()
    e_name, ev = list(evals.items())[-1]
    y, g = ev[mz.TARGET].values, ev['shot_id'].values
    splits = mz.make_splits(y, g)
    sid = ev[['shot_id']].values.astype(float)
    sup = mz.supervised_models()
    rows = []
    for m_name in ['로지스틱 L2', 'LightGBM']:  # 1차와 같은 사전 고정 모델
        model = sup[m_name][0]
        ap_time = _cv_ap(model, sid, y, splits)
        ap_full = _cv_ap(model, np.column_stack([ev[feats].values, sid]), y, splits)
        obs = ap_full[:10].mean() - ap_time[:10].mean()

        def one(seed):
            Xp = np.column_stack([_permute_within_windows(ev, feats, window, np.random.default_rng(seed)), sid])
            return _cv_ap(model, Xp, y, splits[:10]).mean() - ap_time[:10].mean()

        null = np.array(Parallel(n_jobs=-1)(delayed(one)(mz.SEED + i) for i in range(n_perm)))
        rows.append({'평가': e_name, '모델': m_name, 'PR-AUC 샷 순서만': ap_time.mean(), 'PR-AUC 샷 순서 + 공정 값': ap_full.mean(),
                     '추가 기여 (50 fold 평균)': ap_full.mean() - ap_time.mean(),
                     '추가 기여가 양수인 fold 비율': (ap_full > ap_time).mean(),
                     '추가 기여 (10 fold)': obs, '순열 기준 95% 상한': np.quantile(null, 0.95),
                     '순열 p': (np.sum(null >= obs) + 1) / (n_perm + 1)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / 'shot_level_increment.csv', index=False, encoding='utf-8-sig')
    return tbl


if __name__ == '__main__':
    t0 = time.time()
    s = compare()
    cols = ['eval', 'model', 'PR-AUC_mean', 'PR-AUC_std', 'F1_mean', '검사율@재현율0.9_mean']
    print(s[cols].round(3).to_string(index=False))
    print(increment().round(3).to_string(index=False))
    print(f'total {time.time() - t0:.0f}s')
