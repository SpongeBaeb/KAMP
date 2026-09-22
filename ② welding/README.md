# 로봇용접 제조데이터 기반 용접불량 예측 및 안정 공정조건 분석

2026년 제6회 K-인공지능 제조데이터 분석 경진대회 — 일반국민/대학(원)생 부문
**출제과제 ②**: 로봇용접 제조데이터 기반 용접불량 예측 및 안정 공정조건 분석

## 1. 폴더 구조

```
② welding/
├── welding.ipynb               # 메인 분석 노트북 (전처리~학습~추론~산출물 자동 실행)
├── data/
│   ├── Welding Data Set_01.xlsx   # AI 용접기 데이터셋 원본 (KAMP 제공)
│   └── scaled_data.csv            # 스케일링된 참고 데이터
├── outputs/                   # 노트북 실행 시 자동 생성되는 산출물
│   ├── 01~11_*.png             # 분석 단계별 시각화
│   ├── best_model.pkl          # 최종 선정 모델(직렬화)
│   └── test_predictions.csv    # 테스트데이터 예측결과 (제출 필수 산출물)
├── requirements.txt            # 재현 환경 패키지 버전
└── README.md
```

가상환경(`.venv`)은 용량이 커 저장소에 포함하지 않았습니다. 아래 절차로 로컬에서 새로 생성하세요.

## 2. 실행 환경 재현 방법

```bash
# 0) 이 폴더로 이동
cd "② welding"

# 1) 가상환경 생성 (Python 3.12 권장 — xgboost/lightgbm/shap 호환성)
python -m venv .venv

# 2) 패키지 설치
.venv\Scripts\pip install -r requirements.txt

# 3) Jupyter 커널 등록
.venv\Scripts\python -m ipykernel install --user --name welding --display-name "Welding"

# 4) 노트북을 처음부터 끝까지 실행 (전처리→학습→추론→산출물 자동 생성)
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace ^
    --ExecutePreprocessor.kernel_name=welding welding.ipynb
```

모든 난수 시드는 `RANDOM_STATE=42`로 고정되어 있고, 학습/검증 분할은 시간 순서 기반(무작위 셔플 없음)이므로 동일 환경에서 재실행 시 동일한 결과가 재현됩니다.

## 3. 데이터셋 요약

- `Raw data` 시트: 개별 용접(spot) 단위 공정 센서 실측값 11,939건 (2020-03-24 ~ 2020-04-07, 9개 작업일, 단일 설비 Spot-01, 단일 두께 0.7mm)
- `result` 시트: 작업일자 × 불량유형(파임불량/용접부족/크랙발생) 단위 **집계** 불량 건수 23건
- `data set` 시트: 각 공정변수의 정상 작동범위(스펙) 메타데이터

**핵심 제약사항**: 원본 데이터에는 개별 용접 단위의 정답(합/불) 라벨이 없습니다. 노트북 Part 3~4에서 이를 데이터로 직접 증명하고, 스펙범위 + SPC(3-sigma) 관리한계 기반의 방어 가능한 proxy 라벨을 설계·검증하는 과정을 상세히 기술했습니다.

## 4. 분석 파이프라인 개요 (노트북 Part 구성)

| Part | 내용 |
|---|---|
| 0 | 환경설정 및 재현성(시드 고정) |
| 1 | 데이터 로드 및 구조 파악 |
| 2 | 데이터 진단(결측/중복/상수열/이상치/불균형) |
| 3 | 생산단위(raw vs result) 관계 분석 — proxy 라벨 필요성 증명 |
| 4 | 라벨 설계(스펙+SPC) 및 타당성 검증 |
| 5 | 특징공학 — **라벨 누수 방지**(현재 행 원본값 제외, lag/rolling만 사용) |
| 6 | 시간 기반 분할, 베이스라인 포함 4개 모델 비교, 전환시점 진단 |
| 7 | 변수중요도/SHAP, FN/FP 오류분석(OOF 보조표본) |
| 8 | 현장 활용방안 |
| 9 | 임계값 최적화·확률보정·앙상블·불확실성 추정 |
| 10 | 모델/예측결과 저장, requirements.txt 자동 생성 |

## 5. 주요 한계 및 향후 과제 (정직한 고지)

- 라벨은 실제 최종검사 결과가 아닌 스펙/SPC 기반 proxy이며, 1차 스크리닝 신호로만 사용해야 합니다.
- 단일 설비·단일 두께 조건에서 수집된 9일치 데이터에 기반하므로, 결론의 일반화에는 한계가 있습니다.
- 실 배치 전 실제 검사결과 라벨을 활용한 소규모 파일럿 검증을 권고합니다.

## 6. 문의

작성자 이메일: hs104405@gmail.com
