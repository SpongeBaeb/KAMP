# report_extras — 보고서 수정본에서 새로 추가한 분석

| 파일 | 보고서 위치 | 내용 |
|---|---|---|
| `CN7_effects.ipynb` | 제3장 <표 6>, 제4장 <표 7> CN7 행 | CN7 불량 시기(샷 0~60) vs 이후 구간 효과크기, 처음 N샷 전수 검사의 포착률과 Clopper–Pearson 95% 하한 |
| `ROI_scenarios.ipynb` | 제4장 <표 9>·<표 10>·<그림 1>, 파일럿 판정 기준 | 시나리오별 연간 절감액·회수 기간, 허용 회수 기간(12/24개월)에서 역산한 필요 검사율 |
| `build_notebooks.py` | — | 위 두 노트북을 만드는 스크립트 (셀을 고치려면 이 파일을 고치고 `python build_notebooks.py`) |

- 노트북은 실행 결과 없이 저장되어 있다. Jupyter에서 위부터 순서대로 실행한다 (각 수 초).
- `CN7_effects.ipynb`는 `../1_CN7/data/moldset_labeled_cn7.csv`를 읽는다.
- ROI 입력값(투자비 1,200만 원, 검사비 100원/개 등)은 보고서 <표 9>의 가정치다. 견적·현장 값이 확보되면 `P`만 바꾼다.

샷 단위 재평가(보고서 제2장)는 제품 폴더의 `run_shot_level.py`에 있다 (결과: `outputs/tables/shot_level_*.csv`).
