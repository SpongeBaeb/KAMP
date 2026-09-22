# ① 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정

2026년 제6회 K-인공지능 제조데이터 분석 경진대회 (일반국민/대학원생 부문, 과제 ①)

## 과제 개요

사출성형기(CN7, RG3) 공정데이터를 이용해 최종 품질검사 이전 시점에 `PassOrFail`(양불판정)을
예측하는 AI 모델을 개발하고, 모델이 어떤 조건에서 정확하게 작동하고 어떤 조건에서 실패하는지
분석한다. 예측 결과를 검사 우선순위 결정에 활용하는 방안까지 제시한다.

## 폴더 구성

```
① Molding/
├── molding.ipynb          # 분석 노트북 (EDA -> 전처리 -> 모델링 -> 영향요인분석 -> 검사우선순위)
├── moldset_labeled_cn7.csv    # CN7 설비, 라벨(PassOrFail) 있음
├── moldset_labeled_rg3.csv    # RG3 설비, 라벨(PassOrFail) 있음
├── moldset_unlabeled_cn7.csv  # CN7 설비, 라벨 없음 (대량, EDA/분포비교용)
├── moldset_unlabeled_rg3.csv  # RG3 설비, 라벨 없음 (대량, EDA/분포비교용)
├── requirements.txt        # 재현에 필요한 패키지 버전
└── README.md
```

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook molding.ipynb   # 또는 VSCode에서 열어 커널을 .venv로 지정 후 Run All
```

노트북은 위에서 아래로 순서대로 실행하면 데이터 로드 -> EDA(결측/중복/불균형 진단) ->
전처리(중복 제거, 무분산 변수 제거) -> 모델 비교(로지스틱회귀/RandomForest/XGBoost) ->
교차검증 기반 최종모델 선정 -> SHAP 영향요인 분석 -> 오류(FN/FP)분석 -> 확률보정 ->
검사 우선순위 등급 산출까지 자동으로 진행된다.

## 핵심 진단 결과 (요약)

- 라벨 데이터의 약 절반이 중복행 → 분할 전 제거하지 않으면 데이터 누수 발생 (제거 후 반영)
- `Clamp_Open_Position`은 표준편차 0인 무분산 변수 → 제거
- 중복 제거 후 양성(불량) 표본은 CN7+RG3 합쳐 39건으로 극소수 → 성능의 근본적 제약 요인
- 단일 train/test split 평가는 표본이 적을 때 고분산이라 신뢰할 수 없음 → 반복 교차검증(5-fold x10회)으로 재평가
- 최종모델: 규제 튜닝된 로지스틱회귀 (C=0.01), 교차검증 PR-AUC ≈ 0.159 (무작위 기준 0.032 대비 약 5배)
- SMOTE, PU-learning식 unlabeled 활용, 이상탐지(Mahalanobis) 단독/블렌딩, PCA, 다항 상호작용항,
  다양한 분류기(RF/XGBoost/LightGBM/SVM/앙상블 등) 시도 → 대부분 기준선을 넘지 못함
- 검사 우선순위 3단계 등급화: 고위험 등급의 실제 불량비율이 전체 평균의 약 3배로, 검사 리소스
  집중 배치에 활용 가능
