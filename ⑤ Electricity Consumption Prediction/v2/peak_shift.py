"""v2 피크 저감 시뮬레이션 (출제 요구: 생산일정·설비가동 시점 조정을 통한 피크전력 저감방안)

질문: 같은 날 안에서 부하를 앞뒤 W시간 안으로 옮길 수 있다면, 각 달의 최대수요전력(15분)을 어디까지 낮출 수 있는가?

방법 (15분 단위 수요 사용)
- 목표 상한 L을 1 kW씩 낮추며, 그 달 모든 날에서 다음이 가능한지 본다:
  상한을 넘는 15분 구간의 초과분(kW × 0.25h)을 같은 날, 앞뒤 W시간 안에서 상한보다 낮은 구간의 여유분으로 옮긴다
  (가까운 구간부터 채움, 하루 사용 전력량은 그대로)
- 가능한 가장 낮은 L = 그 달의 달성 가능 최대수요. W = 1시간, 2시간 두 경우

가정과 한계
- 설비별 부하·공정 순서 제약을 모른다. 15분 단위로 옮길 수 있다고 본 상한(upper bound)이다
- 기본요금은 최대수요가 요금적용전력이 될 때만 줄어든다. 계약전력 기준으로 부과되는 계약이면 효과가 없다
- 단가: 한전 산업용(갑)Ⅰ 고압A 선택Ⅱ 기본요금 7,470원/kW (2023-05-16 시행 요금표, 계약전력 300kW 미만) — 현재 단가로 바꿔 쓸 것
"""
from pathlib import Path

import numpy as np
import pandas as pd

from prep import Q, load

OUT = Path(__file__).parent / 'outputs'
BASIC_RATE = 7470  # 원/kW·월


def quarter_series(d):
    q = d[['dt'] + Q].melt(id_vars='dt', var_name='q', value_name='kW')
    q['dt'] = q['dt'] + pd.to_timedelta(q['q'].map({'15분': 0, '30분': 15, '45분': 30, '60분': 45}), unit='m')
    q = q.sort_values('dt').reset_index(drop=True)
    q['date'] = q['dt'].dt.normalize()
    return q


def day_feasible(kw, cap, w_steps):
    """한 날(96개 15분 값)에서 상한 cap을 지킬 수 있는지 (가까운 여유 구간부터 채우는 탐욕적 배분)"""
    excess = np.clip(kw - cap, 0, None)
    room = np.clip(cap - kw, 0, None)
    for i in np.flatnonzero(excess > 0):
        need = excess[i]
        for dist in range(1, w_steps + 1):
            for j in (i - dist, i + dist):
                if 0 <= j < len(kw) and room[j] > 0 and need > 0:
                    take = min(room[j], need)
                    room[j] -= take
                    need -= take
            if need <= 0:
                break
        if need > 1e-9:
            return False
    return True


def min_cap(days, w_steps):
    hi = max(k.max() for k in days)
    lo = int(np.ceil(max(k.mean() for k in days)))  # 하루 평균보다 낮출 수는 없음
    for cap in range(int(hi), lo - 1, -1):
        if not all(day_feasible(k, cap, w_steps) for k in days):
            return cap + 1
    return lo


def run():
    d = load()
    q = quarter_series(d)
    q['month'] = q['dt'].dt.month
    rows = []
    for m, g in q.groupby('month'):
        days = [x['kW'].values.astype(float) for _, x in g.groupby('date')]
        base = g['kW'].max()
        at = g.loc[g['kW'].idxmax(), 'dt']
        r = {'월': m, '현재 최대수요 (kW)': base, '발생 시각': at.strftime('%m-%d %H:%M')}
        for w in [1, 2]:
            cap = min_cap(days, w * 4)
            # 그 상한을 위해 옮겨야 하는 날·전력량
            ex = np.concatenate([np.clip(k - cap, 0, None) for k in days])
            r[f'±{w}시간 조정 시 (kW)'] = cap
            r[f'±{w}시간 저감 (kW)'] = base - cap
            r[f'±{w}시간 옮길 전력량 (kWh/월)'] = ex.sum() * 0.25
            r[f'±{w}시간 조정 필요 일수'] = int(sum((k > cap).any() for k in days))
        rows.append(r)
    t = pd.DataFrame(rows)
    for w in [1, 2]:
        t[f'±{w}시간 기본요금 절감 (원/월)'] = t[f'±{w}시간 저감 (kW)'] * BASIC_RATE
    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / 'peak_shift_simulation.csv', index=False, encoding='utf-8-sig')
    return t


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    t = run()
    print(t.to_string(index=False))
    for w in [1, 2]:
        print(f'±{w}시간: 월 평균 저감 {t[f"±{w}시간 저감 (kW)"].mean():.1f} kW, 9개월 합 절감 {t[f"±{w}시간 기본요금 절감 (원/월)"].sum():,.0f}원')
