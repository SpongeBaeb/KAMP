import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 CN7 데이터 분석을 시작합니다...")

# 1. 데이터 로드
file_path = 'c:/Hackerton/Final_Deliverables_CN7/1_Raw_Data_Labeled_CN7.csv'
df = pd.read_csv(file_path)

# 인덱스 컬럼 처리
if df.columns[0] == 'Unnamed: 0' or df.columns[0].startswith('Unnamed'):
    df = df.rename(columns={df.columns[0]: 'Original_Index'})
else:
    # 첫 컬럼이 이름이 없는 경우
    df = df.rename(columns={df.columns[0]: 'Original_Index'})

# 센서 컬럼만 추출
sensor_cols = [col for col in df.columns if col not in ['Original_Index', 'PassOrFail']]

# 2. Virtual ID (쌍둥이/캐비티) 식별
df['shot_id'] = df.groupby(sensor_cols, sort=False).ngroup()
df['Virtual_ID'] = df.groupby('shot_id').cumcount()

print("\n" + "="*50)
print(" 📊 [CN7] 캐비티 위치(Virtual_ID) 별 불량 분포")
print("="*50)
print(pd.crosstab(df['Virtual_ID'], df['PassOrFail'], margins=True))

v0_defects = df[(df['Virtual_ID']==0) & (df['PassOrFail']==1)].shape[0]
total_defects = df[df['PassOrFail']==1].shape[0]
print(f"\n👉 CN7 역시 전체 불량 {total_defects}개 중 {v0_defects}개({v0_defects/total_defects*100:.1f}%)가 앞쪽 캐비티(0번)에 쏠려 있습니다!")

# 3. 누수 없는 시계열 피로도(EMA) 계산
unique_shots = df.drop_duplicates(subset=['shot_id']).copy()
if 'Original_Index' in unique_shots.columns:
    unique_shots = unique_shots.sort_values('Original_Index').reset_index(drop=True)

unique_shots['EMA_Injection_Time'] = unique_shots['Injection_Time'].ewm(span=10, adjust=False).mean()
unique_shots['EMA_Mold_Temp_3'] = unique_shots['Mold_Temperature_3'].ewm(span=10, adjust=False).mean()

df = pd.merge(df, unique_shots[['shot_id', 'EMA_Injection_Time', 'EMA_Mold_Temp_3']], on='shot_id', how='left')

# 4. 모델 평가 (Target Leakage를 막기 위해 shot_id 기준 GroupKFold 수행)
features = ['Virtual_ID', 'EMA_Injection_Time', 'EMA_Mold_Temp_3'] + sensor_cols
X = df[features]
y = df['PassOrFail']
groups = df['shot_id']

# 과적합을 막기 위해 깊이가 얕은 ExtraTrees 모델 사용
gkf = GroupKFold(n_splits=5)
model = ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')

oof_preds = np.zeros(len(df))

for train_idx, val_idx in gkf.split(X, y, groups):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 훈련셋에 불량이 하나도 없는 폴드가 발생할 수 있으므로 예외처리
    if y_train.sum() == 0:
        continue
        
    model.fit(X_train, y_train)
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

print("\n" + "="*50)
print(" 🤖 [CN7] AI 모델 평가 결과 (누수 0% 엄격한 조건)")
print("="*50)
try:
    auc = roc_auc_score(y, oof_preds)
    print(f"✅ CN7 교차검증(CV) 실제 ROC-AUC: {auc:.4f}")
except Exception as e:
    print("ROC-AUC 계산 실패 (테스트셋 불량 부족 등):", e)

# 5. 최종 데이터 덮어쓰기
df.to_csv('c:/Hackerton/Final_Deliverables_CN7/2_Preprocessed_Data_CN7.csv', index=False)
print("\n[저장 완료] 새로운 '2_Preprocessed_Data_CN7.csv' 덮어쓰기 완료.")

res_df = df.copy()
res_df['Predicted_Probability'] = oof_preds
res_df['Predicted_Label'] = (oof_preds > 0.5).astype(int)
res_df.to_csv('c:/Hackerton/Final_Deliverables_CN7/3_Validation_Predictions_Result_CN7.csv', index=False)
print("[저장 완료] 진짜 예측값이 들어간 '3_Validation_Predictions_Result_CN7.csv' 덮어쓰기 완료.")
