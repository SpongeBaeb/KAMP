# 프로젝트: 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정

제6회 K-인공지능 제조데이터 분석 경진대회(일반국민·대학(원)생 부문) 1번 과제다.
마감은 2026-10-08 23:59, 목표 제출일은 10-07이다.

## 작업 방식 (반드시 지킬 것)
- 아래 Phase 단위로 진행하고, 각 Phase가 끝나면 결과를 요약한 뒤 멈추고 내 확인을 받아라.
- [결정 필요]로 표시된 항목은 절대 임의로 정하지 말고, 근거와 선택지를 제시한 뒤 내 답을 기다려라.
- 모든 그래프는 outputs/figures/, 표는 outputs/tables/에 저장하라.
- 난수 시드는 42로 전역 고정하라.

## 팀 공통 규칙 (임의 변경 금지)
- 시드 42, Repeated Stratified K-Fold 5-fold × 10회
- 결과표는 outputs/tables/에 CSV로 저장하고 컬럼은
  model, product, f1_mean, f1_std, recall_mean, recall_std,
  precision_mean, precision_std, prauc_mean, prauc_std, threshold 순으로 통일
- 특징 중요도는 outputs/tables/feature_importance_{제품}.csv로 저장
- 이 규칙은 CN7 담당과 결과를 합치기 위한 것이므로 바꾸지 마라

## 데이터
경로: data/
- moldset_labeled_rg3.csv : 라벨 PassOrFail (1=불량)
- moldset_unlabeled_rg3.csv : 라벨 없음 (테스트/준지도학습용)
- 이 프로젝트는 RG3 제품만 담당한다. CN7은 팀원이 같은 규칙으로 별도 진행한다.
- 'Unnamed: 0' 컬럼은 원본 인덱스다. 특징으로 쓰지 말되, 시간 순서 정보인지 분석하라.

사전 관찰(직접 재확인할 것):
- 불량 개수 CN7 17/1211, RG3 25/1182 (불량률 약 1.4%, 2.1%)
- 라벨 데이터의 약 50%가 특징값 기준 중복 행
- 라벨 데이터는 이미 표준화된 것으로 보이며, 비라벨 데이터는 스케일이 다를 가능성이 있음
- 간이 5-fold CV에서 CN7은 ROC-AUC 약 0.96, RG3는 0.5 미만

## 평가 지표
- 주요 지표: F1, Recall, PR-AUC
- 결과표에는 Precision도 함께 기재
- 현장 활용용: Recall@상위 k% (k = 5, 10, 20%)
- ROC-AUC는 참고용으로만 기재
- 모든 지표는 Repeated Stratified K-Fold(5-fold × 10회)의 평균±표준편차로 보고
- Recall을 주요 지표로 둔 근거: 불량 유출(FN) 비용이 과검출(FP)보다 큼

## Phase 1. 데이터 진단
1. 제품별 기초 통계, 결측, 상수/준상수 변수, 불균형 확인
2. 중복 행 구조 분석: 중복이 라벨까지 같은지, 불량 행도 중복되는지
3. 'Unnamed: 0' 순서에 따라 불량이 특정 구간에 몰리는지 시각화
4. 라벨 vs 비라벨 데이터의 변수별 분포 비교 (같은 스케일인지 판정)
[결정 필요] 중복 행 처리 방식 (제거 / GroupKFold로 묶기 / 유지)
[결정 필요] 비라벨 데이터 스케일이 다를 경우의 처리 방식
[결정 필요] 시간 순서가 확인될 경우 시간 기반 분할 병행 여부

## Phase 2. 모델 비교
- 베이스라인: 로지스틱 회귀(class_weight='balanced')
- 비교 모델: 랜덤포레스트, 그래디언트 부스팅 계열 최소 1종
- 불균형 대응: class weight vs 오버샘플링 비교 (오버샘플링은 반드시 fold 내부에서만 적용)
- 임계값: F1 최대점과 Recall 우선점(예: Recall ≥ 0.8) 두 가지를 제시
- 누수 방지: 전처리·샘플링·임계값 탐색은 모두 CV fold 내부에서만 수행

## Phase 2-B. 추가 비교 모델: 준지도학습 + 소프트보팅 앙상블 + 비용 기반 임계값
Phase 2 모델 비교의 후보로 추가한다. 결과는 Phase 2 결과표에 같은 형식으로 합친다.
최종 모델 선정 기준(Phase 2 결정 사항)은 그대로 적용한다.
[결정 필요] 최종 모델 선정 (근거와 함께 후보 제시)

## Phase 3. 영향요인 및 오류분석
- SHAP으로 주요 변수와 상호작용 분석
- CN7에서는 예측되고 RG3에서는 실패하는 원인 분석 (핵심 챕터)
- FN과 FP가 집중되는 공정 조건을 변수 구간화로 도출

## Phase 4. 현장 활용 및 차별화
- 불량 확률 순 정렬 시 검사 비율 대비 불량 포착률 곡선
- 확률 구간별 3단계 운영안(경보/재검사/통과) 제안용 수치 산출
- 확률 보정(Platt, Isotonic) 전후 Brier Score와 reliability diagram 비교
- 비라벨 데이터로 pseudo-labeling 시도, 성능 변화와 원인 분석
  (Phase 1에서 스케일 문제가 해결된 경우에만 진행)

## Phase 5. 재현성 패키징
- `python run_all.py` 한 번으로 전처리 → 학습 → 평가 → 비라벨 데이터 예측 파일 생성까지 실행
- requirements.txt, README.md 작성
- 비라벨 데이터 예측 결과를 outputs/predictions/에 CSV로 저장 (원본 인덱스, 불량확률, 판정)

## 폴더 구조
data/raw/, src/(preprocess.py, train.py, evaluate.py, predict.py), outputs/(figures, tables, predictions), run_all.py, requirements.txt, README.md

## 진행 현황 및 결정 기록 (2026-09-23 기준)
- 완료: Phase 1, Phase 2(2-B 포함, 추가 실험 2-C·2-D·2-E), Phase 3의 RG3 부분. 다음: Phase 3 CN7 비교 → Phase 4.
- 결정 사항 전체와 스크립트 실행 순서는 README.md의 "결정 기록"·"실행 방법"을 따른다. 요약:
  - ① 중복 행: 쌍 병합, 라벨 = max (591행, 불량 25건)
  - ② 비라벨: 학습에 쓰지 않음, 테스트 예측은 순위 변환 + 한계 명시 (2-B 준지도학습만 예외)
  - ③ 시간 분할: 랜덤 CV 메인, 시간 분할 보조 (변화점 idx 1891)
  - 최종 모델: HGB, class_weight='balanced'. 모델 추가 실험은 종료
- CN7 데이터는 팀 저장소 ../①Molding/ 에 있다.

## 블라인드 평가 규칙
코드, 주석, README, 파일명, 그래프 어디에도 학교명·소속·로고를 넣지 마라.
최종 단계에서 소속 정보가 없는지 전체 검사하라.
