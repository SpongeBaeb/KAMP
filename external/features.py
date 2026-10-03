"""ImPure 사이클 시계열 → 사이클별 특징 (기획서 4단계). 규칙은 결과를 보기 전에 고정한다.

구간 검출 (ScrewPosition, HydPressure만 사용)
- 사출 시작 t0: 스크류가 시작 위치(첫 1초 중앙값)보다 1 mm 이상 전진한 첫 시점
- 보압 끝 t_hold: t0 이후 유압이 최대값의 10% 아래로 처음 떨어진 시점
- 계량: t_hold 이후 스크류가 보압 끝 위치 + 1 mm 위로 올라간 시점부터 시작 위치 − 1 mm에 닿을 때까지

특징 세트
- M (사출기): 유압 최대·적분·고압 유지 시간, 스크류 시작 위치·쿠션·사출 시간·최대 사출 속도·계량 시간, 사이클 길이
- C (캐비티 센서): 캐비티별 내압 피크·피크 도달 시간·최대 상승 기울기·적분·보압 끝 압력, 금형 온도 시작값·최대값
  + 캐비티 간 차이 (P1 − P2 피크·적분)
Analog Input [1]·[2]는 의미를 확인하기 전까지 쓰지 않는다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / 'data'
TRIALS = {'trial_16_05': 'Trial 16 05', 'trial_17_05': 'Trial 17 05'}
COL = {'hyd': 'HydPressure[IRT/Pascoe]', 'screw': 'ScrewPosition[IRT/Pascoe]',
       'p1': 'Pressure1[IRT/Pascoe]', 'p2': 'Pressure2[IRT/Pascoe]',
       't1': 'TempMold1[IRT/Pascoe]', 't2': 'TempMold2[IRT/Pascoe]'}


def _first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def cycle_features(df):
    # 시각 표기가 불규칙한 행이 있어("12:56:010.000") 직접 나눠 읽는다. 샘플 간격은 평상시 약 43 ms,
    # 사출 같은 이벤트 구간에서 1~3 ms로 일정하지 않으므로 실제 시각을 쓴다. 같은 시각이 반복되면 간격을 1 ms로 본다
    hms = df['Time'].str.split(':', expand=True).astype(float)
    raw = (hms[0] * 3600 + hms[1] * 60 + hms[2]).values
    t = np.concatenate([[0.0], np.cumsum(np.clip(np.diff(raw), 1e-3, None))])
    s = {k: df[c].values.astype(float) for k, c in COL.items()}
    dt = np.gradient(t)
    start_pos = np.median(s['screw'][t < 1.0])
    i0 = _first(s['screw'] < start_pos - 1.0)
    if i0 is None:
        return {'ok': False, 'reason': '사출 없음'}
    hmax = s['hyd'][i0:].max()
    ih = _first((np.arange(len(t)) > i0 + np.argmax(s['hyd'][i0:])) & (s['hyd'] < 0.1 * hmax))
    if ih is None:
        return {'ok': False, 'reason': '보압 끝 없음'}
    cushion = s['screw'][i0:ih + 1].min()
    i_inj_end = _first((np.arange(len(t)) >= i0) & (s['screw'] <= cushion + 1.0))
    i_pl0 = _first((np.arange(len(t)) > ih) & (s['screw'] > s['screw'][ih] + 1.0))
    i_pl1 = _first((np.arange(len(t)) > (i_pl0 or ih)) & (s['screw'] >= start_pos - 1.0)) if i_pl0 is not None else None
    speed = -np.gradient(s['screw'], t)
    f = {'ok': True, 'n_rows': len(t),
         'M_cycle_len': t[-1],
         'M_hyd_max': hmax,
         'M_hyd_int': np.sum(s['hyd'][i0:ih] * dt[i0:ih]),
         'M_hyd_high_time': np.sum(dt[i0:ih][s['hyd'][i0:ih] > 0.5 * hmax]),
         'M_screw_start': start_pos,
         'M_cushion': cushion,
         'M_inj_time': t[i_inj_end] - t[i0] if i_inj_end is not None else np.nan,
         'M_inj_speed_max': speed[i0:ih].max(),
         'M_plast_time': (t[i_pl1] - t[i_pl0]) if (i_pl0 is not None and i_pl1 is not None) else np.nan}
    for c, pk, tk in [('1', 'p1', 't1'), ('2', 'p2', 't2')]:
        p = s[pk]
        ip = i0 + int(np.argmax(p[i0:]))
        f.update({f'C{c}_peak': p[ip], f'C{c}_t_peak': t[ip] - t[i0],
                  f'C{c}_rise_max': np.gradient(p, t)[i0:ip + 1].max(),
                  f'C{c}_int': np.sum(np.clip(p[i0:], 0, None) * dt[i0:]),
                  f'C{c}_p_hold_end': p[ih],
                  f'C{c}_temp_start': np.median(s[tk][t < 1.0]), f'C{c}_temp_max': s[tk].max()})
    f['D_peak'] = f['C1_peak'] - f['C2_peak']
    f['D_int'] = f['C1_int'] - f['C2_int']
    return f


def load_trial(trial):
    d = DATA / trial
    prefix = TRIALS[trial]
    lab = pd.read_csv(d / f'{prefix} all stages_ires-labels.csv')
    rows = []
    for fp in d.glob('*Cycle*.csv'):
        cyc = int(fp.stem.split('Cycle')[1])
        rows.append({'Cycle': cyc, **cycle_features(pd.read_csv(fp))})
    feats = pd.DataFrame(rows)
    out = lab.merge(feats, on='Cycle', how='left')
    out['has_series'] = out['ok'].fillna(False).astype(bool)
    out.insert(0, 'trial', trial)
    return out.sort_values('Cycle').reset_index(drop=True)


def build():
    tbl = pd.concat([load_trial(t) for t in TRIALS], ignore_index=True)
    tbl.to_csv(DATA / 'cycle_features.csv', index=False, encoding='utf-8-sig')
    return tbl


M_COLS = ['M_cycle_len', 'M_hyd_max', 'M_hyd_int', 'M_hyd_high_time', 'M_screw_start', 'M_cushion',
          'M_inj_time', 'M_inj_speed_max', 'M_plast_time']
C_COLS = [f'C{c}_{k}' for c in '12' for k in ['peak', 't_peak', 'rise_max', 'int', 'p_hold_end', 'temp_start', 'temp_max']] + ['D_peak', 'D_int']

if __name__ == '__main__':
    tbl = build()
    print(tbl.groupby('trial').agg(labels=('Cycle', 'size'), with_series=('has_series', 'sum'),
                                   cav1_ng=('Cavity 1', 'sum'), cav2_ng=('Cavity 2', 'sum')))
    print(tbl.loc[tbl['ok'] == False, ['trial', 'Cycle', 'reason']].to_string())
