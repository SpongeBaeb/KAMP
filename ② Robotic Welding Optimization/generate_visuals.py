import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('plots', exist_ok=True)
OUT_DIR = "plots"

def plot_daily_comparison():
    df_raw = pd.read_csv('pseudo_labeled_data.csv')
    true_defects = pd.read_csv('true_daily_defects.csv')
    
    daily_pred = df_raw.groupby('working time')['is_anomaly'].sum().reset_index()
    daily_pred.columns = ['date', 'pred_defects']
    daily_pred['date'] = daily_pred['date'].astype(str)
    true_defects['date'] = true_defects['date'].astype(str)
    
    merged = pd.merge(daily_pred, true_defects, on='date', how='left').fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(merged['date']))
    width = 0.35
    
    ax.bar(x - width/2, merged['true_defects'], width, label='실제 QC 육안 검사 불량 (True Defects)', color='gray')
    ax.bar(x + width/2, merged['pred_defects'], width, label='AI 예측 기계적 이상치 (Predicted Anomalies)', color='red')
    
    ax.set_ylabel('건수 (Count)')
    ax.set_title('AI 모델 실패 원인 분석: QC 육안 검사와 기계 센서 이상의 불일치')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['date'], rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'daily_comparison.png'), dpi=300)
    plt.close()

def plot_safe_zone():
    df = pd.read_csv('pseudo_labeled_data.csv')
    
    normal = df[df['is_anomaly'] == 0]
    anomaly = df[df['is_anomaly'] == 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(normal['weld Voltage(v)'], normal['weld force(bar)'], alpha=0.3, color='blue', label='정상 (Normal)')
    ax.scatter(anomaly['weld Voltage(v)'], anomaly['weld force(bar)'], alpha=1.0, color='red', marker='x', label='이상치 (Anomaly)')
    
    ax.axhline(y=8.96, color='red', linestyle='--', label='위험 임계치 (8.96 bar)')
    ax.axvspan(1.5, 3.5, ymin=0.7, ymax=1, alpha=0.1, color='red', label='High Risk Zone')
    
    ax.set_xlabel('용접 전압 (Voltage, v)')
    ax.set_ylabel('가압력 (Weld Force, bar)')
    ax.set_title('안정 공정조건 범위 (Safe Zone) 도출')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'safe_zone.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_daily_comparison()
    plot_safe_zone()
    print("Visualizations generated in plots directory.")
