"""v2 조건 분석 (출제 요구 ②·③)

② 예측오차가 크게 나는 생산조건: forecast.py의 최종 모델(달력+계획) 예측 오차를 생산·운영 조건별로 집계
③ 최대전력 피크 발생 조건: 전력 실측값과 그 파생값은 쓰지 않고 운영 변수(시각·요일·가동 여부·생산량·인원·기온)만으로 설명
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from prep import load

OUT = Path(__file__).parent / 'outputs'


def prod_bin(x):
    return pd.cut(x, [-1, 0, 300, 800, 1500, np.inf], labels=['0', '1~300', '301~800', '801~1,500', '1,500 초과'])


def error_conditions():
    d = load()
    p = pd.read_csv(OUT / 'forecast_predictions.csv', parse_dates=['dt']).merge(d, on='dt', suffixes=('', '_d'))
    p['요일구분'] = np.where(p['dow'] >= 5, '주말', '평일')
    p['가동일'] = np.where(p['op_day'] == 1, '가동일', '비가동일')
    p['생산량 구간'] = prod_bin(p['생산량'])
    p['시간대'] = pd.cut(p['hour'], [-1, 6, 7, 8, 11, 12, 17, 23], labels=['0~6시', '7시', '8시', '9~11시', '12시', '13~17시', '18~23시'])
    # 가동일인데 평소보다 낮은 날(평일 비가동, 조기 종료 등) = 가동 상태 전환
    p['전날과 가동 상태 다름'] = (p.groupby('date')['op_day'].transform('first').diff().fillna(0) != 0)
    rows = []
    for h in ['H1', 'H24']:
        col = f'{h}|LightGBM (달력+계획)'
        p['ae'] = (p[col] - p['y']).abs()
        p['bias'] = p[col] - p['y']
        for g in ['시간대', '요일구분', '가동일', '생산량 구간']:
            s = p.groupby(g, observed=True).agg(시간수=('ae', 'size'), MAE=('ae', 'mean'), 평균편향=('bias', 'mean'),
                                                  상위5퍼센트오차비중=('ae', lambda x: np.mean(x >= p['ae'].quantile(0.95))))
            rows.append(s.reset_index().rename(columns={g: '조건값'}).assign(구간=h, 조건=g))
    t = pd.concat(rows, ignore_index=True)[['구간', '조건', '조건값', '시간수', 'MAE', '평균편향', '상위5퍼센트오차비중']]
    t.to_csv(OUT / 'error_conditions.csv', index=False, encoding='utf-8-sig')
    return t


def peak_conditions():
    d = load()
    thr = d['Peak'].quantile(0.95)
    d['피크'] = (d['Peak'] >= thr).astype(int)
    d['생산량 구간'] = prod_bin(d['생산량'])
    d['요일구분'] = np.where(d['dow'] >= 5, '주말', '평일')
    by_hour = d.groupby('hour').agg(피크시간수=('피크', 'sum'), 피크비율=('피크', 'mean'), 평균Peak=('Peak', 'mean'))
    by_prod = d.groupby('생산량 구간', observed=True).agg(시간수=('피크', 'size'), 피크비율=('피크', 'mean'))
    by_dow = d.groupby('요일구분').agg(시간수=('피크', 'size'), 피크비율=('피크', 'mean'))
    daily = d.loc[d.groupby('date')['Peak'].idxmax()]
    monthly = d.loc[d.groupby('month')['Peak'].idxmax(), ['dt', 'Peak', '생산량', '기온', 'op_day']]
    # 기동 피크: 8시 4개 구간 중 최댓값 위치, 7시 대비 상승폭
    h8 = d[d['hour'] == 8].copy()
    h7 = d[d['hour'] == 7].set_index('date')['Peak']
    h8['7시 대비 상승'] = h8['Peak'].values - h7.reindex(h8['date']).values
    feats = ['hour', 'dow', 'op_day', '생산량', '공장인원', '기온', '습도']
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, class_weight='balanced', random_state=42).fit(d[feats], d['피크'])
    rules = export_text(tree, feature_names=feats)
    OUT.mkdir(exist_ok=True)
    by_hour.to_csv(OUT / 'peak_by_hour.csv', encoding='utf-8-sig')
    (OUT / 'peak_rules.txt').write_text(f'피크 = Peak >= {thr:.0f} kW (상위 5%)\n운영 변수만 사용\n\n' + rules, encoding='utf-8')
    return {'thr': thr, 'by_hour': by_hour, 'by_prod': by_prod, 'by_dow': by_dow,
            'daily_max_hour': daily['hour'].value_counts(), 'monthly': monthly, 'h8': h8, 'rules': rules,
            'corr': d[['Peak', '생산량', '공장인원', '기온', '습도']].corr()['Peak']}


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    t = error_conditions()
    print(t.round(2).to_string(index=False))
    r = peak_conditions()
    print('피크 기준', r['thr'])
    print(r['by_hour'][r['by_hour']['피크시간수'] > 0].round(3).to_string())
    print(r['by_prod'].round(3).to_string()); print(r['by_dow'].round(3).to_string())
    print('일 최대 발생 시각', r['daily_max_hour'].head(6).to_dict())
    print(r['monthly'].to_string(index=False))
    print('8시: 최대 구간', r['h8'][['15분', '30분', '45분', '60분']].idxmax(axis=1).value_counts().to_dict(),
          ' 7시 대비 상승 중앙값', r['h8']['7시 대비 상승'].median(), ' 가동일 8시 평균', r['h8'].loc[r['h8'].op_day == 1, 'Peak'].mean().round(1),
          ' 비가동일 8시 평균', r['h8'].loc[r['h8'].op_day == 0, 'Peak'].mean().round(1))
    print(r['corr'].round(3).to_dict()); print(r['rules'])
