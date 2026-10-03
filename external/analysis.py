"""ImPure 외부 데이터 분석 (기획서 5~6단계): 사출기 값(M) 대 M + 캐비티 센서(C)

부품 단위 표: 사용 가능한 사이클 1개 → 캐비티 2행. 한 행 = M(두 부품 공통) + 자기 캐비티의 C + 캐비티 번호
주 비교 (사전 고정): 트라이얼 간 평가(16일 학습→17일 평가, 반대도)에서 부품 단위 PR-AUC 차이 (M+C − M)
- 신뢰구간: 평가 트라이얼의 사이클 단위 부트스트랩 2,000회 (같은 사이클의 두 부품은 같이 뽑힘)
- 순열 p: 연속 10사이클 묶음 안에서만 C 특징을 사이클끼리 섞고(학습·평가 모두) 다시 학습, 1,000회
모델: L2 로지스틱, GradientBoosting (설정은 결과를 보기 전에 고정)

실행 순서
  python analysis.py power   # 2단계: 순열한 C로 차이의 우연 변동 폭 확인 (실제 C 결과를 보기 전)
  python analysis.py h1      # 5단계: 캐비티 간 내압 차이와 M으로 설명되는 비율
  python analysis.py h3      # 6단계: 주 비교
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import C_COLS, M_COLS

SEED = 42
DATA = Path(__file__).parent / 'data'
OUT = Path(__file__).parent / 'outputs'
TRIALS = ['trial_16_05', 'trial_17_05']
C_OWN = ['peak', 't_peak', 'rise_max', 'int', 'p_hold_end', 'temp_start', 'temp_max']
MODELS = {'로지스틱 L2': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000)),
          'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED)}
N_BOOT, N_PERM, WINDOW = 2000, 1000, 10


def cycles():
    t = pd.read_csv(DATA / 'cycle_features.csv')
    t = t[t['ok'] == True].copy()
    t['M_plast_time'] = t['M_plast_time'].fillna(t.groupby('trial')['M_plast_time'].transform('median'))
    t['block'] = t.groupby('trial')['Cycle'].rank(method='first').sub(1).floordiv(WINDOW).astype(int)
    return t.reset_index(drop=True)


def parts(cyc):
    """사이클 표 → 부품 표. C는 자기 캐비티 값 + 상대 캐비티와의 차이(자기 − 상대)"""
    rows = []
    for c, o in [('1', '2'), ('2', '1')]:
        p = cyc[['trial', 'Cycle', 'block'] + M_COLS].copy()
        p['cavity'] = int(c)
        p['y'] = cyc[f'Cavity {c}'].values
        for k in C_OWN:
            p[f'C_{k}'] = cyc[f'C{c}_{k}'].values
        p['C_d_peak'] = cyc[f'C{c}_peak'].values - cyc[f'C{o}_peak'].values
        p['C_d_int'] = cyc[f'C{c}_int'].values - cyc[f'C{o}_int'].values
        rows.append(p)
    return pd.concat(rows, ignore_index=True).sort_values(['trial', 'Cycle', 'cavity']).reset_index(drop=True)


C_PART = [f'C_{k}' for k in C_OWN] + ['C_d_peak', 'C_d_int']
SETS = {'M': M_COLS + ['cavity'], 'M+C': M_COLS + ['cavity'] + C_PART}


def permute_c(df, rng):
    """같은 트라이얼·같은 10사이클 묶음 안에서 사이클끼리 C를 섞는다 (한 사이클의 두 캐비티는 같이 이동)"""
    df = df.copy()
    for (_, _), g in df.groupby(['trial', 'block']):
        cyc_ids = g['Cycle'].unique()
        mapping = dict(zip(cyc_ids, rng.permutation(cyc_ids)))
        src = df.loc[g.index].set_index(['Cycle', 'cavity'])[C_PART]
        new = src.loc[[(mapping[c], cav) for c, cav in zip(g['Cycle'], g['cavity'])]].values
        df.loc[g.index, C_PART] = new
    return df


def fit_score(model, tr, te, cols):
    m = clone(model).fit(tr[cols].values, tr['y'].values)
    return m.predict_proba(te[cols].values)[:, 1]


def boot_idx(te, rng):
    cyc_ids = te['Cycle'].unique()
    pick = rng.choice(cyc_ids, len(cyc_ids), replace=True)
    pos = te.groupby('Cycle').indices
    return np.concatenate([pos[c] for c in pick])


def direction_pairs(df):
    a, b = (df[df['trial'] == t].reset_index(drop=True) for t in TRIALS)
    return {'16일 학습 → 17일 평가': (a, b), '17일 학습 → 16일 평가': (b, a)}


def inspect_rate(y, s, target=0.9):
    order = np.argsort(-s, kind='stable')
    caught = np.cumsum(y[order])
    return (np.argmax(caught >= np.ceil(target * y.sum() - 1e-9)) + 1) / len(y)


def power():
    """C를 묶음 안에서 섞은 상태(정보 없음)에서 M+C − M 차이의 분포 → 우연으로 생기는 폭과 검출 가능한 최소 차이"""
    df = parts(cycles())
    rows = []
    for d_name, (tr, te) in direction_pairs(df).items():
        for m_name, model in MODELS.items():
            ap_m = average_precision_score(te['y'], fit_score(model, tr, te, SETS['M']))

            def one(seed):
                rng = np.random.default_rng(seed)
                both = permute_c(pd.concat([tr.assign(_s=0), te.assign(_s=1)], ignore_index=True), rng)
                trp, tep = both[both._s == 0], both[both._s == 1]
                return average_precision_score(tep['y'], fit_score(model, trp, tep, SETS['M+C'])) - ap_m

            null = np.array(Parallel(n_jobs=-1)(delayed(one)(SEED + i) for i in range(200)))
            rows.append({'방향': d_name, '모델': m_name, '평가 부품': len(te), '평가 불량': int(te['y'].sum()),
                         '무정보 차이 평균': null.mean(), '무정보 차이 SD': null.std(ddof=1),
                         '무정보 95% 상한': np.quantile(null, 0.95),
                         '검출 가능 최소 차이 (≈2.8 SD)': 2.8 * null.std(ddof=1)})
    tbl = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    tbl.to_csv(OUT / 'power.csv', index=False, encoding='utf-8-sig')
    return tbl


def h1():
    """캐비티 간 내압 피크 차이의 분포, M으로 설명되는 비율 (사이클 단위 5-fold 교차검증 R²)"""
    cyc = cycles()
    rows = []
    for tr_name, g in [('전체', cyc)] + list(cyc.groupby('trial')):
        d = g['D_peak'].values
        rng = np.random.default_rng(SEED)
        bs = [np.abs(rng.choice(d, len(d))).mean() for _ in range(N_BOOT)]
        X = g[M_COLS].values
        cv = KFold(5, shuffle=True, random_state=SEED)
        r2 = {name: r2_score(d, cross_val_predict(est, X, d, cv=cv)) for name, est in
              [('Ridge', make_pipeline(StandardScaler(), RidgeCV())), ('GB', GradientBoostingRegressor(random_state=SEED))]}
        rows.append({'트라이얼': tr_name, '사이클': len(g), '|P1−P2| 피크 평균': np.abs(d).mean(),
                     '95% CI': f'{np.quantile(bs, 0.025):.1f}~{np.quantile(bs, 0.975):.1f}', '최대': np.abs(d).max(),
                     'P1 < P2 비율': (d < 0).mean(), 'M으로 설명 R² (Ridge)': r2['Ridge'], 'M으로 설명 R² (GB)': r2['GB']})
    tbl = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    tbl.to_csv(OUT / 'h1_imbalance.csv', index=False, encoding='utf-8-sig')
    return tbl


def h3():
    """주 비교 + 기준선 + 보조 지표. 판정: 두 방향 모두 Δ 95% CI 하한 > 0 이고 순열 p < 0.05 → 신호 있음"""
    df = parts(cycles())
    rows, base_rows = [], []
    for d_name, (tr, te) in direction_pairs(df).items():
        y = te['y'].values
        rng = np.random.default_rng(SEED)
        # 기준선 (학습 없음 또는 학습 트라이얼에서 방향만 결정)
        sign = np.sign(np.corrcoef(tr['Cycle'], tr['y'])[0, 1]) or 1.0
        prior = tr.groupby('cavity')['y'].mean()
        base = {'무작위': np.mean([average_precision_score(y, rng.random(len(y))) for _ in range(200)]),
                '캐비티 번호만': average_precision_score(y, te['cavity'].map(prior).values),
                '사이클 순서만': average_precision_score(y, sign * te['Cycle'].values)}
        base_rows += [{'방향': d_name, '기준선': k, 'PR-AUC': v, '평가 불량 비율': y.mean()} for k, v in base.items()]
        for m_name, model in MODELS.items():
            s_m, s_mc = fit_score(model, tr, te, SETS['M']), fit_score(model, tr, te, SETS['M+C'])
            ap_m, ap_mc = average_precision_score(y, s_m), average_precision_score(y, s_mc)
            obs = ap_mc - ap_m
            boot = []
            for _ in range(N_BOOT):
                i = boot_idx(te, rng)
                if y[i].sum() == 0:
                    continue
                boot.append(average_precision_score(y[i], s_mc[i]) - average_precision_score(y[i], s_m[i]))

            def one(seed):
                r = np.random.default_rng(seed)
                both = permute_c(pd.concat([tr.assign(_s=0), te.assign(_s=1)], ignore_index=True), r)
                trp, tep = both[both._s == 0], both[both._s == 1]
                return average_precision_score(tep['y'], fit_score(model, trp, tep, SETS['M+C'])) - ap_m

            null = np.array(Parallel(n_jobs=-1)(delayed(one)(SEED + 10_000 + i) for i in range(N_PERM)))
            # 보조: 캐비티 1이 불량인 사이클에서 캐비티 2도 불량인가 (M은 두 캐비티에 같으므로 구조적으로 불리)
            sub = (te['cavity'] == 2) & te['Cycle'].isin(te.loc[(te['cavity'] == 1) & (te['y'] == 1), 'Cycle'])
            ys = y[sub.values]
            sec = {f'보조 {k}': average_precision_score(ys, s[sub.values]) if 0 < ys.sum() < len(ys) else np.nan
                   for k, s in [('M', s_m), ('M+C', s_mc)]}
            rows.append({'방향': d_name, '모델': m_name, 'PR-AUC M': ap_m, 'PR-AUC M+C': ap_mc, 'Δ': obs,
                         'Δ 95% CI 하한': np.quantile(boot, 0.025), 'Δ 95% CI 상한': np.quantile(boot, 0.975),
                         '순열 기준 평균': null.mean(), '순열 95% 상한': np.quantile(null, 0.95),
                         '순열 p': (np.sum(null >= obs) + 1) / (N_PERM + 1),
                         '검사율@재현율0.9 M': inspect_rate(y, s_m), '검사율@재현율0.9 M+C': inspect_rate(y, s_mc),
                         '보조 대상 (캐비티2 불량/전체)': f'{int(ys.sum())}/{len(ys)}', **sec})
    tbl, base_tbl = pd.DataFrame(rows), pd.DataFrame(base_rows)
    OUT.mkdir(exist_ok=True)
    tbl.to_csv(OUT / 'h3_main.csv', index=False, encoding='utf-8-sig')
    base_tbl.to_csv(OUT / 'h3_baselines.csv', index=False, encoding='utf-8-sig')
    print(base_tbl.round(3).to_string(index=False))
    return tbl


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'power'
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print({'power': power, 'h1': h1, 'h3': h3}[step]().round(3).T.to_string())
