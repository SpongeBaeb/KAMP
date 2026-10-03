# RG3 사출성형 품질불량 사전예측 · 검사 우선순위 (1_RG3_3)

과제: **사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정** — RG3 제품 담당 분석.
`1_CN7`(CN7 분석)과 **같은 구성**으로 RG3를 정리한 버전입니다. CN7과의 비교는 넣지 않았습니다.

> 두 노트북 모두 **실행 결과(표·그림)가 저장된 상태**라 GitHub에서 바로 볼 수 있습니다.

## 구성

| 파일 | 대응하는 CN7 파일 | 내용 |
|---|---|---|
| `RG33.ipynb` (메인) | `1_CN7/CN7.ipynb` | 섹션 1~8: 데이터 진단 · 정제(역변환·누수 구간·운전 조건·부품 쌍) · 특성 · 평가 설계 · 이상탐지 · 검사 우선순위 · 결과 분석 |
| `RG33_models.ipynb` | `1_CN7/CN7_models.ipynb` | 지도학습·불균형 대응·이상탐지·튜닝·앙상블 36종 비교 (저장된 결과를 읽어 해석) |
| `RG31.ipynb` | `1_CN7/CN71.ipynb` | 2차 시도: 선행 논문 기반 파생 변수로 재평가 (결과 `outputs/*/derived_*`) |
| `model_zoo.py` | `1_CN7/model_zoo.py` | 모델 정의 · 평가 함수 (샷 단위 스태킹, 중첩 튜닝 포함) |
| `run_models.py` | `1_CN7/run_models.py` | 모델 비교 실행 → `outputs/tables/model_*.csv`, `time_increment.csv` |

- `RG33.ipynb`는 `../1_RG3_2/RG3.ipynb`에 CN7 분석에서 일반화한 코드(센서 오류 기준, 역변환 후보 규칙·계수 저장, 그림 빈 칸 처리, 극단값 행 수, 시차 피처 자동 선택)를 반영한 것입니다. RG3 결과는 원본과 같습니다.
- `model_zoo.py`·`run_models.py`는 CN7 파일에서 **데이터 경로, XGBoost 불균형 가중치(양품/불량 비 46), 평가 구성**만 바꿨습니다. 모델 36종의 설정은 같습니다.

## RG3 전용으로 바꾼 부분

CN7에만 맞는 분석 두 곳을 RG3에 맞게 바꿨습니다.

| 위치 | CN7 | RG3 | 이유 |
|---|---|---|---|
| `RG33.ipynb` 8-2 (b-2) | 불량 시기 효과 확인 (불량이 샷 24~60에 집중) | **공정 조건 변화점 전후 확인** (샷 340, 원본 인덱스 1891) | RG3는 불량이 샷 68~566에 흩어져 불량 시기가 없고, 대신 중간에 공정 조건이 크게 바뀌는 지점이 있다 |
| `RG33_models.ipynb` 평가 E2 | 불량 시기 안 (시간 신호 제거) | **앞쪽 부품 안** (`pair_pos = 0`, 591부품, 불량 23) | RG3에서 가장 강한 비공정 신호는 시간이 아니라 부품 위치다. 부품 위치를 고정하고 공정 값만으로 판정한다 |

## 핵심 결과

**변화점 전후 (`RG33.ipynb` 8-2 b-2)**
- 공정 값으로 변화점 전후가 거의 완벽히 구분되지만(ROC-AUC 0.998), 불량률은 같다 (불량 샷 15/340 vs 10/251, p = 0.84).
- 쌍 순서 규칙(앞쪽 부품 우선)은 두 시기 모두 작동한다 (ROC-AUC 0.69 / 0.76). 공정 값 모델은 시기마다 방향이 뒤집힌다.

**모델 비교 (`RG33_models.ipynb`, 샷 묶음 5-fold × 10)**

| | PR-AUC | Recall@10/20/30% | 검사율@재현율0.9 |
|---|---|---|---|
| E2 앞쪽 부품 안: 무작위 | 0.072 | 0.09 / 0.19 / 0.30 | 91% |
| E2 앞쪽 부품 안: 최고 XGBoost (공정 값) | 0.096 ± 0.061 | 0.17 / 0.28 / 0.40 | 88% |
| E1 전체: 쌍 순서 규칙 | 0.072 | 0.22 / 0.40 / 0.56 | **48%** |
| E1 전체: 최고 XGBoost (공정 값 + pair_pos) | 0.084 | 0.26 / 0.39 / 0.51 | 76% |

- 36개 모델 모두 E2에서 무작위, E1에서 쌍 순서 규칙을 편차 이상으로 넘지 못했다. 튜닝·앙상블·불균형 샘플링도 기본 설정 부스팅을 넘지 못했다.
- 공정 값이 시간 흐름에 더하는 정보도 유의하지 않다 (E2 조건부 순열검정 LightGBM p = 0.35).

**결론은 쌍 순서(`pair_pos`)의 해석에 달려 있다**
- 해석 A (쌍 순서 = 부품 위치 LH/RH): **앞쪽 부품 우선 검사**. 검사 50%로 불량 92%를 잡는다.
- 해석 B (쌍 순서 = 검사 결과 기록 순서 → 정답 누수): `pair_pos`를 쓸 수 없고, 공정 값만으로는 무작위 수준 → **전수검사 유지**.
- 확인 방법: KAMP 원본 1차 가공 데이터(`labeled_data.csv`)의 `PART_NAME`·`TimeStamp`.

**2차 시도: 파생 변수 (`RG31.ipynb`)** — `1_CN7/CN71.ipynb`와 같은 파생 변수 63개(샷 내 13 + 샷 간 변동 50), 같은 8개 모델, 같은 분할·지표로 다시 평가했다.

| E2 앞쪽 부품 안 | PR-AUC | 검사율@재현율0.9 |
|---|---|---|
| 무작위 (기준선) | 0.072 ± 0.057 | 91% |
| 1차 최고: XGBoost (공정 값) | 0.096 ± 0.061 | 88% |
| 2차 최고: GradientBoosting (공정 값 + 파생 전체) | 0.128 ± 0.087 | 85% |

- **판정: 실패** (사전에 정한 기준). 차이가 fold 표준편차보다 작다. 순열 p = 0.035는 결과를 보고 고른 모델에서만 나왔고, 사전에 고정한 LightGBM(p = 0.79)·로지스틱(p = 0.16)은 유의하지 않다.
- 단변량으로 유의한 파생 변수는 0개다. 1차 상위 모델(XGBoost·LightGBM)은 파생 변수를 넣으면 오히려 낮아진다. → "공정 값에 불량 정보가 없음"이라는 결론은 그대로다.
- `RUN = False`로 저장된 결과(`outputs/tables/derived_*.csv`)를 읽는다. 다시 계산하려면 `True` (16코어 기준 약 22분).

## 실행 방법

Python 3.12 기준.

```bash
cd 1_RG3_3
python -m venv .venv
.venv\Scripts\activate          # 맥/리눅스: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook RG33.ipynb     # 또는 VS Code에서 열고 커널로 .venv 선택
```

- **데이터는 이 폴더에 없습니다** (중복 방지). `RG33.ipynb`와 `model_zoo.py`는 `data/`가 없으면 같은 저장소의 `../1_RG3_2/data/`(`moldset_labeled_rg3.csv`, `moldset_unlabeled_rg3.csv`)를 읽습니다.
- `RG33.ipynb`는 위에서부터 순서대로 실행합니다 (24코어 기준 약 4분, 6-6 순열검정 셀이 가장 오래 걸림). 실행하면 역변환 계수가 `outputs/tables/inverse_transform_coef.csv`로 저장됩니다.
- `RG33_models.ipynb`는 저장된 결과(`outputs/tables/`)를 읽습니다. 다시 계산하려면 노트북의 `RUN = True` 또는 `python run_models.py` (24코어 기준 약 28분).
- 난수 시드는 `SEED = 42`로 통일되어 있어 결과가 재현됩니다.

## 폴더 구성

```
1_RG3_3/
├── RG33.ipynb            # 메인 분석 (실행 결과 포함)
├── RG33_models.ipynb     # 모델 비교 36종 결과·해석 (실행 결과 포함)
├── RG31.ipynb            # 2차 시도: 파생 변수로 재평가 (실행 결과 포함, outputs/*/derived_*)
├── model_zoo.py          # 모델 정의 · 평가 함수
├── run_models.py         # 모델 비교 실행
├── outputs/
│   ├── tables/
│   │   ├── model_summary.csv     # 전 모델·전 지표 평균·표준편차 (E1, E2)
│   │   ├── model_folds.csv       # fold별 지표
│   │   ├── model_inspection.csv  # 반복별 검사율@재현율
│   │   └── time_increment.csv    # E2에서 샷 순서 대비 공정 값의 추가 기여
│   └── figures/model_prauc.png   # 모델별 PR-AUC (E2 / E1)
├── requirements.txt
└── README.md
```

## 샷 단위 재평가 (`run_shot_level.py`)

한 샷의 두 부품은 공정 값이 같으므로, 공정 값으로 답할 수 있는 질문은 "이 샷에 불량이 있는가"다. 샷 라벨(두 부품 중 하나라도 불량이면 1)과 공정 값 23개(pair_pos 없음)로 1차와 같은 분할·지표·임계값 규칙을 적용한다. 모델은 결과를 보기 전에 5종(로지스틱 L2, 랜덤포레스트, XGBoost, LightGBM, CatBoost)과 Isolation Forest로 고정했다.

- 실행: `python run_shot_level.py` (1분 내외)
- 결과: `outputs/tables/shot_level_folds.csv`, `shot_level_summary.csv`, `shot_level_increment.csv`
