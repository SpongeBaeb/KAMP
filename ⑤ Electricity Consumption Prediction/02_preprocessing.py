# %%
import pandas as pd
import numpy as np
import os

# Load data
df = pd.read_csv('okm_augumented_2021.csv')

# 1. Missing Value Imputation
df['풍속'] = df['풍속'].interpolate(method='linear').ffill().bfill()
df['강수량'] = df['강수량'].fillna(0)
df['공장인원'] = df['공장인원'].interpolate(method='linear').ffill().bfill()

# 2. Target Variable Generation and In-rush Current Filtering
df['Peak'] = df[['15분', '30분', '45분', '60분']].max(axis=1)
p99 = df['Peak'].quantile(0.99)
df['is_inrush'] = (df['Peak'] > p99).astype(int)

# 3. Time Series Lags and Moving Averages
df['Peak_lag1'] = df['Peak'].shift(1)
df['Peak_lag2'] = df['Peak'].shift(2)
df['Peak_lag24'] = df['Peak'].shift(24)
df['Peak_lag48'] = df['Peak'].shift(48)
df['Peak_MA3'] = df['Peak'].rolling(window=3).mean()
df['Peak_MA6'] = df['Peak'].rolling(window=6).mean()
df['Peak_MA12'] = df['Peak'].rolling(window=12).mean()
df['Peak_MA24'] = df['Peak'].rolling(window=24).mean()

# Cyclic time features
df['hour_sin'] = np.sin(2 * np.pi * df['시간'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['시간'] / 24.0)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7.0)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7.0)
df['month_sin'] = np.sin(2 * np.pi * df['m'] / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['m'] / 12.0)

df = df.dropna().reset_index(drop=True)

# 4. Financial Risk Feature
df['Financial_Risk_Index'] = df['전기요금(계절)'] * df['인건비']

# 전처리된 데이터 저장
df.to_csv('preprocessed_data.csv', index=False, encoding='utf-8-sig')
print("Preprocessed data saved to 'preprocessed_data.csv'.")
