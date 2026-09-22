import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('preprocessed_data.csv')
df = df.dropna().reset_index(drop=True)

# 1. 최대 전력 피크가 발생하는 조건 (Top 5% Peak)
threshold_95 = df['Peak'].quantile(0.95)
df['is_peak'] = (df['Peak'] >= threshold_95).astype(int)

features = ['기온', '풍속', '습도', '공장인원', '생산량', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
X = df[features]
y_peak = df['is_peak']

dt_peak = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
dt_peak.fit(X, y_peak)
print("=== 최대 전력 피크(상위 5%) 발생 주요 조건 ===")
print(export_text(dt_peak, feature_names=features))

# 2. 예측 오차가 크게 발생하는 조건 (Top 5% Residuals)
# 간이 모델로 오차 추출
X_train, X_test, y_train, y_test = train_test_split(X, df['Peak'], test_size=0.2, random_state=42, shuffle=False)
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
residuals = np.abs(y_test - preds)

threshold_err = np.quantile(residuals, 0.95)
error_df = X_test.copy()
error_df['is_high_error'] = (residuals >= threshold_err).astype(int)

dt_error = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
dt_error.fit(error_df[features], error_df['is_high_error'])
print("\n=== 예측 오차(상위 5%)가 크게 발생하는 주요 조건 ===")
print(export_text(dt_error, feature_names=features))
