"""v2 예측 평가: 예측 구간 2개 × 특징 세트 2개, 피크 포함, 기준선 비교, 월 단위 전진 검증.

예측 구간 (출제 문구 "향후 지정된 시간구간")
- H1  : 1시간 앞. t-1시까지의 실측으로 t시를 예측 (실시간 피크 경보용)
- H24 : 익일 24시간. 전날 23시까지의 실측으로 다음 날 0~23시를 예측 (생산·가동 일정 조정용)

특징 세트
- 달력: 시각·요일·월 + 과거 전력(예측 시점에 알 수 있는 것만)
- 달력+계획: 위 + 그 시각의 생산량·공장인원·기온·습도·가동일 여부.
  생산 계획과 기온 예보를 미리 안다고 가정하고 실측값을 대신 썼다 → 실제 운영보다 낙관적일 수 있음

검증: 4~9월 각 달을 평가, 그 달 이전 전체로 학습 (6 fold). 피크 시간을 빼지 않는다.
모델 설정·특징 목록은 결과를 보기 전에 고정했고, 특징 선택은 하지 않는다 (선택 누수 방지).
구간 예측: 학습 기간 마지막 20%로 잔차 90% 분위수를 구해 ±q 구간 → 평가 달에서 실제 포함률 확인.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prep import load

OUT = Path(__file__).parent / 'outputs'
SEED = 42
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
PLAN = ['생산량', '공장인원', '기온', '습도', 'op_day', 'producing']
CAL = ['hour', 'dow', 'month']
TEST_MONTHS = [4, 5, 6, 7, 8, 9]


def features(d):
    p = d['Peak']
    f = pd.DataFrame(index=d.index)
    # H1: t-1까지
    for k in list(range(1, 25)) + [48, 168]:
        f[f'h1_lag{k}'] = p.shift(k)
    for w in [3, 24, 168]:
        f[f'h1_mean{w}'] = p.shift(1).rolling(w).mean()
        f[f'h1_max{w}'] = p.shift(1).rolling(w).max()
    # H24: 전날 23시까지 (그날 시각과 무관하게 같은 정보)
    day = d.groupby('date')['Peak']
    prev = pd.DataFrame({'dmax': day.max(), 'dmean': day.mean(), 'd23': day.last()}).shift(1)
    f[['h24_prev_dmax', 'h24_prev_dmean', 'h24_prev_d23']] = prev.loc[d['date']].values
    f['h24_lag24'] = p.shift(24)
    f['h24_lag168'] = p.shift(168)
    f['h24_samehour_mean7'] = sum(p.shift(24 * k) for k in range(1, 8)) / 7
    f['h24_prev_week_dmax'] = pd.Series(day.max().shift(1).rolling(7).max().loc[d['date']].values, index=d.index)
    for c in CAL + PLAN:
        f[c] = d[c]
    return f


def cols(h, s, f):
    base = [c for c in f.columns if c.startswith('h1_')] if h == 'H1' else [c for c in f.columns if c.startswith('h24_')]
    return base + CAL + (PLAN if s == '달력+계획' else [])


def metrics(y, yhat, peak_thr):
    e = yhat - y
    pk = y >= peak_thr
    return {'RMSE': np.sqrt(np.mean(e ** 2)), 'MAE': np.mean(np.abs(e)),
            'RMSE_피크시간': np.sqrt(np.mean(e[pk] ** 2)) if pk.any() else np.nan, '피크시간 수': int(pk.sum())}


def run():
    d = load()
    f = features(d)
    rows, preds = [], []
    for m in TEST_MONTHS:
        tr = (d['month'] < m) & f['h1_lag168'].notna() & f['h24_prev_week_dmax'].notna()
        te = d['month'] == m
        y_tr, y_te = d.loc[tr, 'Peak'].values, d.loc[te, 'Peak'].values
        thr = np.quantile(y_tr, 0.95)  # 피크 시간 = 학습 기간 상위 5% 이상 (평가 달 정보 미사용)
        out = {'dt': d.loc[te, 'dt'].values, 'y': y_te, 'fold': m}
        base = {('H1', '기준: 직전 1시간'): f.loc[te, 'h1_lag1'], ('H24', '기준: 전날 같은 시각'): f.loc[te, 'h24_lag24'],
                ('H24', '기준: 전주 같은 시각'): f.loc[te, 'h24_lag168'], ('H24', '기준: 최근 7일 같은 시각 평균'): f.loc[te, 'h24_samehour_mean7']}
        for (h, name), yhat in base.items():
            rows.append({'월': m, '구간': h, '방법': name, **metrics(y_te, yhat.values, thr)})
            out[f'{h}|{name}'] = yhat.values
        for h in ['H1', 'H24']:
            for s in ['달력', '달력+계획']:
                c = cols(h, s, f)
                Xtr, Xte = f.loc[tr, c], f.loc[te, c]
                n_cal = int(len(Xtr) * 0.2)
                m_cal = lgb.LGBMRegressor(**PARAMS).fit(Xtr.iloc[:-n_cal], y_tr[:-n_cal])
                q90 = np.quantile(np.abs(y_tr[-n_cal:] - m_cal.predict(Xtr.iloc[-n_cal:])), 0.9)
                model = lgb.LGBMRegressor(**PARAMS).fit(Xtr, y_tr)
                yhat = model.predict(Xte)
                name = f'LightGBM ({s})'
                cover = np.mean(np.abs(y_te - yhat) <= q90)
                pk = y_te >= thr
                rows.append({'월': m, '구간': h, '방법': name, **metrics(y_te, yhat, thr),
                             '90% 구간 폭(±)': q90, '구간 포함률': cover,
                             '구간 포함률_피크시간': np.mean(np.abs(y_te[pk] - yhat[pk]) <= q90) if pk.any() else np.nan})
                out[f'{h}|{name}'] = yhat
                out[f'{h}|{name}|q90'] = np.full(len(yhat), q90)
        preds.append(pd.DataFrame(out))
    res = pd.DataFrame(rows)
    pred = pd.concat(preds, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / 'forecast_by_month.csv', index=False, encoding='utf-8-sig')
    pred.to_csv(OUT / 'forecast_predictions.csv', index=False, encoding='utf-8-sig')
    w = res.groupby(['구간', '방법'], sort=False)
    summ = w[['RMSE', 'MAE', 'RMSE_피크시간']].mean().join(w[['RMSE']].std().rename(columns={'RMSE': 'RMSE 월간 SD'}))
    summ = summ.join(w[['90% 구간 폭(±)', '구간 포함률', '구간 포함률_피크시간']].mean())
    summ.to_csv(OUT / 'forecast_summary.csv', encoding='utf-8-sig')
    return summ, pred


def daily_max_error(pred):
    """H24: 날짜별 최대수요(일 최대 Peak) 예측 오차"""
    pred['date'] = pd.to_datetime(pred['dt']).dt.normalize()
    cols_ = [c for c in pred.columns if c.startswith('H24|') and not c.endswith('q90')]
    g = pred.groupby('date')
    actual = g['y'].max()
    out = {c.split('|', 1)[1]: (g[c].max() - actual).abs().mean() for c in cols_}
    s = pd.Series(out, name='일 최대수요 MAE (kW)')
    s.to_csv(OUT / 'forecast_daily_max_error.csv', encoding='utf-8-sig')
    return s


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    summ, pred = run()
    print(summ.round(2).to_string())
    print(daily_max_error(pred).round(2).to_string())
