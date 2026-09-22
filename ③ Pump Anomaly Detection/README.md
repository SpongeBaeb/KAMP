# ③ 소성가공 예지보전 — 프레스 유압펌프 이상 조기탐지 및 오경보 분석

2026 제6회 K-인공지능 제조데이터 분석 경진대회 (일반국민/대학원생 부문) 과제 ③.

## 폴더 구성
- `pump.ipynb` — 전체 분석/모델링 노트북 (데이터 진단 → 특징추출 → 모델 7종 비교 → 앙상블 → 오류분석 → 현장 활용방안)
- `data/` — 원본 데이터 (`press_data_normal.csv`: 정상 20,000행, `outlier_data.csv`: 이상 600행)
- `scripts/` — 노트북 로직 프로토타이핑용 스크립트 (참고용, 노트북 실행에는 불필요)
- `requirements.txt` — 실행 환경 의존성 고정 목록

## 실행 환경 설정 (팀원용)

이 프로젝트는 시스템 파이썬을 쓰지 않고, 프로젝트 전용 가상환경을 사용합니다.

```powershell
# 1. 이 폴더가 있는 저장소 루트에서, 가상환경 생성 (Python 3.12 권장)
py -3.12 -m venv .venv

# 2. 패키지 설치
.venv\Scripts\python.exe -m pip install -r "③ Pump Anomaly Detection\requirements.txt"

# 3. Jupyter 커널로 등록
.venv\Scripts\python.exe -m ipykernel install --user --name presspump-venv --display-name "Python (Presspump venv)"
```

이후 `pump.ipynb`를 VSCode/Jupyter에서 열고 커널을 **"Python (Presspump venv)"**로 선택하면 위에서부터 순서대로 실행 시 전체 파이프라인이 재현됩니다.

## 분석 요약
- 접근법: 정상 데이터로만 학습하고 이상 정도를 점수화하는 semi-supervised 이상탐지
- 핵심 전처리: 타임스탬프 gap(최대 16초) 기반 세그먼트 분리 — 프레스 사이클 간 유휴시간이 섞여 특징이 왜곡되는 것을 방지
- 특징: 채널(진동x2, 전류)별 17종 × 3 + 교차상관 3종 = 57차원 (RMS/crest/shape/impulse/clearance factor, envelope RMS, 대역 에너지비 등)
- 모델: 3-sigma 베이스라인, Isolation Forest, One-Class SVM, LOF, PCA T²/SPE, Conv-Autoencoder, GRU-Autoencoder + rank-평균 앙상블 + Platt scaling 확률보정
- 최종 앙상블 성능(held-out 기준): F1 ≈ 0.97, 오경보율 ≈ 0.8%, 탐지율 100%
- 영향요인 분석(permutation importance): 모터 전류(`AI2_Current`)의 급변화율·envelope 특징이 이상탐지에 가장 크게 기여
