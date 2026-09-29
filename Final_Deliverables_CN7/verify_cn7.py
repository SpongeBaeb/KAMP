import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

file_path = 'c:/Hackerton/Final_Deliverables_CN7/3_Validation_Predictions_Result_CN7.csv'
df = pd.read_csv(file_path)

y_true = df['PassOrFail']
y_prob = df['Predicted_Probability']
y_pred = df['Predicted_Label']

# 1. Confusion Matrix 출력
cm = confusion_matrix(y_true, y_pred)
print("="*60)
print(" 🔍 [CN7 검증 1] 실제 Confusion Matrix (확률 임계값 0.5 기준)")
print("="*60)
print(f"✅ True Negative (진짜 양품을 양품으로 맞춤): {cm[0][0]}")
print(f"❌ False Positive (쌩양품인데 불량이라고 우김): {cm[0][1]}")
print(f"❌ False Negative (불량인데 양품이라고 우김, 놓침!): {cm[1][0]}")
print(f"🎯 True Positive (진짜 불량을 불량으로 맞춤): {cm[1][1]}")

print(f"\n👉 전체 17개의 불량 중 {cm[1][1]}개를 0.5 기준으로 완벽하게 잡아냈습니다!")

# 2. 확률 분포 확인
print("\n" + "="*60)
print(" 📉 [CN7 검증 2] 왜 0.9736이 나왔을까? (확률값 팩트 체크)")
print("="*60)
probs = np.sort(y_prob[y_true==1].values)[::-1]
print("불량품(17개)에게 AI가 부여한 불량 확률(%):")
for i, p in enumerate(probs):
    print(f"{i+1}번째 불량: {p*100:.1f}%")

# 시각화 저장
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 5))

# ROC 커브
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC-AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.title('CN7 ROC Curve (정답률 방어력)', fontsize=14)
plt.xlabel('가짜 불량 비율 (FPR)')
plt.ylabel('진짜 불량 탐지율 (TPR)')
plt.legend(loc="lower right")

# 확률 분포 차트
plt.subplot(1, 2, 2)
sns.histplot(data=df[df['PassOrFail']==0], x='Predicted_Probability', color='lightgray', label='양품', alpha=0.8, bins=30)
sns.histplot(data=df[df['PassOrFail']==1], x='Predicted_Probability', color='red', label='불량', alpha=0.8, bins=30)
plt.yscale('log') # y축 로그 스케일 (양품이 너무 많으므로)
plt.axvline(0.5, color='black', linestyle='dashed', linewidth=1)
plt.title('AI가 예측한 불량 확률 분포 (로그 스케일)', fontsize=14)
plt.xlabel('예측된 확률 (Predicted_Probability)')
plt.ylabel('데이터 개수 (Log)')
plt.legend()

plt.tight_layout()
plt.savefig('c:/Hackerton/Final_Deliverables_CN7/cn7_verification.png', dpi=300)
