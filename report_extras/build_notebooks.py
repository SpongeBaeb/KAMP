"""report_extras 노트북 생성 스크립트 (nbformat). 셀 내용을 바꾸려면 이 파일을 고치고 다시 실행한다."""
import nbformat as nbf

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == 'md' else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    return n

effects = [
('md', """# CN7 불량 시기 효과크기 (보고서 제3장 <표 6>, 제4장 <표 7> CN7 행)

- 입력: `../1_CN7/data/moldset_labeled_cn7.csv` (labeled 원자료, 역변환 전 z값)
- 샷 정의는 `1_CN7/model_zoo.load_evalset`과 같다: 공정 값 조합이 같은 두 행 = 한 샷, 처음 나온 순서가 샷 번호
- 효과크기 = (두 집단 샷 평균 차이) ÷ (전체 샷 표준편차). 샷 단위로 계산해 같은 샷의 두 부품이 두 번 세어지지 않게 한다"""),
('code', """import numpy as np
import pandas as pd
from scipy.stats import beta

TARGET = 'PassOrFail'
lab = pd.read_csv('../1_CN7/data/moldset_labeled_cn7.csv', index_col=0)
feats = [c for c in lab.columns if c != TARGET and lab[c].nunique() > 1]
grp = lab.groupby(feats, sort=False)
ev = lab[feats + [TARGET]].assign(shot_id=grp.ngroup())
shots = ev.groupby('shot_id').agg({**{f: 'first' for f in feats}, TARGET: 'max'})
print('샷', len(shots), '/ 불량 샷', shots[TARGET].sum(), '/ 불량 샷 번호', shots.index[shots[TARGET] == 1].tolist())"""),
('md', "## 불량 시기(샷 0~60) vs 이후 구간"),
('code', """last_ng = shots.index[shots[TARGET] == 1].max()
period = shots.index <= last_ng
es = ((shots.loc[period, feats].mean() - shots.loc[~period, feats].mean()) / shots[feats].std())
table6 = es.reindex(es.abs().sort_values(ascending=False).index).head(8).round(2).rename('효과크기 (SD)')
table6"""),
('md', "## 불량 시기 안: 불량 샷 vs 정상 샷 (불량 샷 14개라 참고용)"),
('code', """A = shots[period]
es2 = (A.loc[A[TARGET] == 1, feats].mean() - A.loc[A[TARGET] == 0, feats].mean()) / A[feats].std()
es2.reindex(es2.abs().sort_values(ascending=False).index).head(6).round(2)"""),
('md', """## 참고: 라벨 구간 처음 N샷 전수 검사 (제4장 <표 7> 참고 행)

표본 안에서 관측한 값이며, 구간(샷 0~60)은 불량 위치를 본 뒤 정한 것이다. 신뢰구간은 샷 단위 Clopper–Pearson 95%.
**운영 규칙으로 쓸 수 없다**: 비라벨 데이터 기준 라벨 샷 0(비라벨 인덱스 751,091)은 직전 큰 중단·조건 변경(인덱스 692,358→735,217)보다 약 440샷 뒤의 생산 중간이다."""),
('code', """rows = []
for n in [30, 61, 100]:
    insp = ev['shot_id'] < n
    caught_parts = ev.loc[insp, TARGET].sum()
    caught_shots = int(shots.loc[shots.index < n, TARGET].sum())
    k, m = caught_shots, int(shots[TARGET].sum())
    lo = beta.ppf(0.025, k, m - k + 1) if k > 0 else 0.0
    rows.append({'처음 N샷': n, '검사율': round(insp.mean(), 3), '포착 불량 부품': f'{caught_parts}/{ev[TARGET].sum()}',
                 '포착 불량 샷': f'{k}/{m}', '재현율 95% 하한 (샷 기준)': round(lo, 2)})
pd.DataFrame(rows)"""),
]

roi = [
('md', """# 센서 투자 ROI 시나리오 (보고서 제4장 <표 9>·<표 10>·<그림 1>, 파일럿 판정 기준)

연간 절감액 S = N[(r0 − r1)Ci + (R1 − R0)·p·Co] − M, 회수 기간 T = I / S

- 입력값은 보고서 <표 9>의 가정이다. 센서·DAQ 단가(I)는 견적 확보 전 가정치다
- 시나리오의 r1·R1은 범위를 보기 위한 가정이다. 외부 공개 데이터(external/)에서는 센서의 추가 효과가 확인되지 않았다"""),
('code', """import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

P = dict(N=300_000, Ci=100, p=0.02, M=1_000_000, I=12_000_000, r0=0.50, R0=0.92)
scen = pd.DataFrame([
    ('낙관 (가정)', 0.10, 0.995, 50_000),
    ('중립', 0.20, 0.97, 30_000),
    ('검사비 절감만 인정', 0.10, 0.92, 0),
    ('보수', 0.30, 0.95, 10_000),
    ('효과 없음', 0.50, 0.92, 0),
], columns=['시나리오', 'r1', 'R1', 'Co'])
scen['부품당 절감 (원)'] = (P['r0'] - scen.r1) * P['Ci'] + (scen.R1 - P['R0']) * P['p'] * scen.Co
scen['연간 절감 S (만 원)'] = (P['N'] * scen['부품당 절감 (원)'] - P['M']) / 1e4
scen['회수 기간 T (개월)'] = [P['I'] / (s * 1e4) * 12 if s > 0 else float('nan') for s in scen['연간 절감 S (만 원)']]
scen.round({'부품당 절감 (원)': 1, '연간 절감 S (만 원)': 0, '회수 기간 T (개월)': 1})"""),
('code', """fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
y = range(len(scen))[::-1]
a1.barh(list(y), scen['연간 절감 S (만 원)'], color=['tab:red' if v < 0 else 'grey' for v in scen['연간 절감 S (만 원)']])
a1.axvline(0, color='black', lw=0.8); a1.set_xlabel('연간 절감액 (만 원)')
a1.set_yticks(list(y)); a1.set_yticklabels(scen['시나리오'])
a2.barh(list(y), scen['회수 기간 T (개월)'].fillna(0), color='tab:blue')
a2.axvline(12, ls='--', color='grey'); a2.set_xlabel('회수 기간 (개월), 점선 = 1년')
for yi, t in zip(y, scen['회수 기간 T (개월)']):
    a2.text(0.5 if pd.isna(t) else t + 0.3, yi, '회수 불가' if pd.isna(t) else f'{t:.1f}', va='center')
fig.tight_layout()"""),
('md', """## 투자비 ±50% 민감도 (보고서 <그림 2>)

투자비(I)는 견적 전 가정치라, 600만·1,200만·1,800만 원에서 회수 기간을 다시 계산한다. 회수 기간은 투자비에 비례한다."""),
('code', """sens = scen[['시나리오']].copy()
for inv in [6_000_000, 12_000_000, 18_000_000]:
    sens[f'I = {inv / 1e4:,.0f}만 원'] = [inv / (s * 1e4) * 12 if s > 0 else float('nan') for s in scen['연간 절감 S (만 원)']]
sens.round(1)"""),
('md', """## 파일럿 판정 기준: 허용 회수 기간에서 역산

검사비 절감만 인정(R1 = R0)하면 r1 ≤ r0 − (I/T + M) / (N·Ci)"""),
('code', """rows = []
for months in [12, 24]:
    need = P['I'] / (months / 12) + P['M']
    r1_max = P['r0'] - need / (P['N'] * P['Ci'])
    rows.append({'허용 회수 기간 (개월)': months, '필요 연간 절감 (만 원)': (need - P['M']) / 1e4,
                 '필요 검사율 r1 (재현율 0.92 유지)': f'{r1_max:.1%}', '현행 대비 감소 (%p)': round((P['r0'] - r1_max) * 100, 1)})
pd.DataFrame(rows)"""),
]

nbf.write(nb(effects), 'CN7_effects.ipynb')
nbf.write(nb(roi), 'ROI_scenarios.ipynb')
print('written')
