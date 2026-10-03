"""v2 데이터 준비: 원본(okm_augumented_2021.csv) → 시간 순서가 보장된 시간 단위 표.

- 타깃 Peak = 그 시간 15분 단위 수요전력 4개 중 최댓값 (최대수요전력은 15분 단위로 정해짐)
- 2021-07-13, 07-15 이틀은 '시간' 열에 시각 대신 전력값이 들어가 있다(48행).
  4개 구간 값과 평균은 서로 맞으므로, 그날 행 순서대로 0~23시를 다시 매긴다
- 결측: 풍속·강수량·공장인원 소수 → 직전 값으로 채움 (미래 값을 쓰지 않음)
- 가동일 = 그날 생산량 합 > 0
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'okm_augumented_2021.csv'
Q = ['15분', '30분', '45분', '60분']


def load():
    d = pd.read_csv(RAW)
    bad_days = d.loc[d['시간'] > 23, '날짜'].unique()
    fix = d['날짜'].isin(bad_days)
    d.loc[fix, '시간'] = d[fix].groupby('날짜').cumcount()
    d['dt'] = pd.to_datetime(d['날짜'].astype(str)) + pd.to_timedelta(d['시간'], unit='h')
    d = d.sort_values('dt').reset_index(drop=True)
    assert d['dt'].is_unique and (d['dt'].diff().dropna() == pd.Timedelta('1h')).all()
    d[['풍속', '강수량', '공장인원']] = d[['풍속', '강수량', '공장인원']].ffill().fillna(0)
    d['Peak'] = d[Q].max(axis=1)
    d['hour'] = d['dt'].dt.hour
    d['dow'] = d['dt'].dt.dayofweek          # 0 = 월
    d['month'] = d['dt'].dt.month
    d['date'] = d['dt'].dt.normalize()
    d['op_day'] = (d.groupby('date')['생산량'].transform('sum') > 0).astype(int)
    d['producing'] = (d['생산량'] > 0).astype(int)
    d.attrs['fixed_days'] = [str(x) for x in bad_days]
    return d


if __name__ == '__main__':
    d = load()
    print(len(d), d['dt'].min(), d['dt'].max(), '시간 열 복구:', d.attrs['fixed_days'])
