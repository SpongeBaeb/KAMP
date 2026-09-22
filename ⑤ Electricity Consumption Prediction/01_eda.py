import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Load data
df = pd.read_csv('okm_augumented_2021.csv')

print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())

# 1. Target variable distribution and 99th percentile
target_cols = ['15분', '30분', '45분', '60분', '평균']
df['Peak'] = df[['15분', '30분', '45분', '60분']].max(axis=1)

print("\nPeak Power Distribution:")
print(df['Peak'].describe())

p99 = df['Peak'].quantile(0.99)
print(f"\n99th Percentile of Peak Power (In-rush current threshold): {p99}")

plt.figure(figsize=(10, 6))
sns.histplot(df['Peak'], bins=50, kde=True)
plt.axvline(p99, color='red', linestyle='dashed', linewidth=2, label=f'99th percentile: {p99:.2f}')
plt.title('Distribution of Peak Power (Max of 15/30/45/60 min)')
plt.legend()
plt.savefig('plots/peak_power_dist.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Peak'])
plt.axvline(p99, color='red', linestyle='dashed', linewidth=2, label=f'99th percentile: {p99:.2f}')
plt.title('Boxplot of Peak Power')
plt.legend()
plt.savefig('plots/peak_power_boxplot.png')
plt.close()

# 2. Correlation Analysis
# Relevant columns: Peak, 평균, 생산량, 기온, 풍속, 습도, 강수량, 공장인원
cols_for_corr = ['Peak', '평균', '생산량', '기온', '풍속', '습도', '강수량', '공장인원']
corr = df[cols_for_corr].corr(method='pearson')
print("\nPearson Correlation with Peak Power:")
print(corr['Peak'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('plots/correlation_matrix.png')
plt.close()

print("\nEDA completed. Plots saved to 'plots/' directory.")
