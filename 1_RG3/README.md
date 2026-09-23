# 1_RG3 — 사출성형 품질불량 사전예측 (RG3 제품)

제6회 K-인공지능 제조데이터 분석 경진대회 1번 과제 중 **RG3 제품** 분석. CN7은 팀원이 같은 규칙으로 별도 진행한다.
작업 규칙·Phase 구성은 [CLAUDE.md](CLAUDE.md), 지금까지의 결정 사항은 아래 [결정 기록](#결정-기록)을 따른다.

## 한 줄 결론 (Phase 1~3 기준)

RG3는 현재 공정 변수로 불량을 예측할 수 없다. 모델 11종·22개 후보 탐색 모두 라벨 순열 기준선을 넘지 못했고
(순열 p ≥ 0.55), 불량 25건은 모두 특징값이 같은 양품 쌍둥이를 가진다. 유일한 신호는 쌍 안의 기록 순서
(불량 23/25가 앞쪽 기록, p=0.00002)이며 의미 확인 전까지 부록 실험으로 둔다.

## 실행 방법

```bash
# Python 3.14 (3.12도 가능)
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

데이터는 `1_RG3/data/` 에 두거나, 없으면 저장소의 `../①Molding/` 폴더를 자동으로 읽는다 (`src/preprocess.py`).

스크립트는 **아래 순서대로** 실행한다. 팀 공통 결과표(`model_comparison_RG3.csv`)는 Phase 2가 새로 쓰고
2-B·2-C·2-E가 행을 덧붙이므로, 앞 단계만 다시 돌리면 뒤 단계 행이 사라진다.

| 순서 | 스크립트 | 내용 | 소요 (24코어 기준) |
|---|---|---|---|
| 1 | `src/phase1_diagnosis.py` | 데이터 진단 | 약 1분 |
| 2 | `src/phase2_models.py` | 모델 비교, 누수·순열 검정, 1891 변화점, 시간 분할, 비라벨 한계 근거 | 약 3분 |
| 3 | `src/phase2b_ensemble_ssl.py` | 소프트보팅 앙상블, 자기학습(준지도), 비용 기반 임계값 | 약 2분 |
| 4 | `src/phase2c_anomaly.py` | 이상탐지 4종 (정상만 학습), 최근접 정상 거리 | 약 1분 |
| 5 | `src/phase2d_features.py` | 상관관계, 변수 정리·파생변수 세트 비교 | 약 2.5분 |
| 6 | `src/phase2e_maximize.py` | 22개 후보 중첩 CV 탐색, 쌍 기록 순서 실험 | 약 3분 |
| 7 | `src/phase2_summary.py` | 계열별 요약표 + 추가 순열 검정 | 약 1.5분 |
| 8 | `src/phase3_analysis.py` | SHAP, RG3 실패 원인, FN/FP 구간화 | 약 15초 |

공통 모듈: `preprocess.py`(로드·쌍 병합·파생변수·순위 변환), `train.py`(모델·fold 내부 임계값), `evaluate.py`(지표·팀 형식 표), `viz.py`(그래프 스타일).

## 폴더 구조

```
1_RG3/
├─ CLAUDE.md, README.md, requirements.txt
├─ src/                 # 위 스크립트
└─ outputs/
   ├─ figures/          # p1_*, p2*_*, p3_* 그래프
   └─ tables/           # p1_*, p2*_*, p3_* 표
                        # 팀 공통: model_comparison_RG3.csv, feature_importance_RG3.csv
```

## 결정 기록

| 항목 | 결정 | 근거 |
|---|---|---|
| ① 중복 행 처리 | **쌍 병합, 라벨 = max** (591행, 불량 25건 4.2%). CN7도 같은 방식 적용 예정 | 모든 행이 정확히 2개씩 짝, 불량 25건 모두 양품 쌍둥이 보유 |
| ② 비라벨 스케일 | **학습에 쓰지 않음.** 테스트 예측은 순위 변환(quantile mapping) + 한계 명시 | 두 파일이 따로 z-표준화, 원단위 산포 10~24,000배 차이 (`p2_unlabeled_rank_evidence.csv`) |
| ② 예외 | Phase 2-B 준지도학습만 순위 변환한 비라벨을 실험 후보로 사용 | 사용자 결정 (개선 없음 확인) |
| ③ 시간 분할 | **랜덤 CV 메인, 시간 분할 보조.** 병합 데이터 기준 | idx가 시간 순서, idx 1891에서 공정 조건 변화 |
| 비용 임계값 | FN:FP = 5·10·20 모두 보고 | 실제 비용 미정 |
| 최종 모델 | **HGB, class_weight='balanced'** (원본 변수) | 명목 Recall@20% 최고, 중첩 CV 최다 선택(19/50), SHAP 호환. 단 모든 모델이 우연 수준 |
| Phase 3 범위 | RG3 쪽 먼저, CN7 비교는 데이터/결과 수령 후 | CN7 파일이 `../①Molding/`에 있음 → 다음 단계에서 사용 가능 |
| 모델 추가 실험 | 종료 | 사용자 결정 |

## 진행 현황

- [x] Phase 1 데이터 진단
- [x] Phase 2 모델 비교 (+2-B 앙상블·준지도·비용 임계값, 2-C 이상탐지, 2-D 파생변수, 2-E 중첩 CV 탐색)
- [x] Phase 3 RG3 부분 (SHAP, 실패 원인, FN/FP 구간화)
- [ ] Phase 3 CN7 비교 (핵심 챕터 완성) — `../①Molding/moldset_labeled_cn7.csv` 로 같은 지표 산출
- [ ] Phase 4 현장 활용 (pseudo-labeling은 스케일 문제 미해결로 건너뜀; 쌍 기록 순서 기반 운영안 검토)
- [ ] Phase 5 재현성 패키징 (`run_all.py`, 비라벨 예측 파일)

## 핵심 결과 파일

| 파일 | 내용 |
|---|---|
| `outputs/tables/p2_summary_by_family.csv` | 계열별 성능 요약 + 무작위 기준선 |
| `outputs/tables/model_comparison_RG3.csv` | 팀 공통 형식 결과표 (38행) |
| `outputs/tables/p2e_nested_summary.csv` | 중첩 CV 결과 (선택 편향 제거) |
| `outputs/tables/p3_rg3_failure_causes.csv` | RG3 실패 원인 7가지와 근거 |
| `outputs/tables/p3_shap_importance.csv` | SHAP 중요도 vs 라벨 순열 귀무분포 |
| `outputs/tables/p2e_pair_position_test.csv` | 쌍 기록 순서 검정 |

## 주의

- 평가 fold당 불량이 약 5건이라, 신호가 없어도 PR-AUC가 약 0.08~0.09 나온다. 비교는 불량률(0.042)이 아니라 라벨 순열 기준선과 해야 한다.
- 블라인드 평가 규칙: 코드·주석·README·파일명·그래프 어디에도 학교명·소속·로고를 넣지 않는다.
