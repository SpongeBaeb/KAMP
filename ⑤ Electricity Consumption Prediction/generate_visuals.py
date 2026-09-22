import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os

# Set Korean font
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# Output directory for artifacts
OUT_DIR = r"C:\Users\samsung\.gemini\antigravity-ide\brain\ebc9c39b-1049-434d-aae4-0d2b13748811"

# 1. RMSE Journey Chart
def plot_rmse_journey():
    stages = ['Baseline', '1차 최적화', '3차 최적화', '4차 (피처셀렉션)', '5차 (168h Lag)', '최종 (MoE)', 'MAPIE (비즈니스)']
    scores = [33.03, 12.40, 11.16, 10.05, 9.99, 9.63, 9.63]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stages, scores, marker='o', linewidth=3, markersize=10, color='#00d2ff')
    ax.fill_between(stages, scores, min(scores)-2, alpha=0.1, color='#00d2ff')
    
    # Annotations
    annots = [
        ("단순 LSTM 접근", 0, 33.03, 5),
        ("데이터 누수 차단", 1, 12.40, 5),
        ("타겟 차분화(Diff)", 2, 11.16, 5),
        ("Tree 앙상블 도입", 3, 10.05, 5),
        ("마의 10.0 돌파!", 4, 9.99, 5),
        ("전문가 동적 라우팅\n(MoE 도입)", 5, 9.63, 2),
        ("신뢰 구간 제공\n(현장 투입)", 6, 9.63, 2)
    ]
    
    for text, x, y, offset in annots:
        ax.annotate(text, (x, y), xytext=(x, y + offset),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
        
    ax.set_ylim(5, 40)
    ax.set_title("🚀 AI 모델 성능 향상 마일스톤 (RMSE)", fontsize=16, fontweight='bold')
    ax.set_ylabel("오차율 (RMSE)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rmse_journey.png"), dpi=300)
    plt.close()

# 2. Peak Conditions Scatter Plot
def plot_peak_conditions():
    # Generate dummy data reflecting the real distribution for visualization
    np.random.seed(42)
    normal_temp = np.random.normal(20, 5, 500)
    normal_prod = np.random.normal(150, 40, 500)
    
    peak_temp = np.random.normal(28, 2, 50)
    peak_prod = np.random.normal(280, 20, 50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(normal_temp, normal_prod, alpha=0.5, color='gray', label='일반 전력 패턴')
    ax.scatter(peak_temp, peak_prod, alpha=0.9, color='red', s=60, edgecolors='black', label='상위 5% 전력 피크 (위험군)')
    
    # Highlight High Risk Zone (Temp > 26.15, Prod > 253)
    rect = patches.Rectangle((26.15, 253), 15, 100, linewidth=2, edgecolor='red', facecolor='red', alpha=0.15)
    ax.add_patch(rect)
    
    ax.annotate("🔥 High Risk Zone\n(냉방 부하 + 기계 풀가동)", 
                xy=(30, 290), xytext=(32, 210),
                ha='center', fontsize=12, fontweight='bold', color='darkred',
                arrowprops=dict(facecolor='darkred', shrink=0.05))
    
    ax.axvline(x=26.15, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=253, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlim(5, 35)
    ax.set_ylim(50, 350)
    ax.set_title("⚠️ 최대 전력 피크 유발 조건 산점도 (XAI 분석)", fontsize=16, fontweight='bold')
    ax.set_xlabel("기온 (℃)", fontsize=12)
    ax.set_ylabel("일일 생산 목표량 (단위)", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "peak_conditions.png"), dpi=300)
    plt.close()

# 3. Conformal Prediction Band
def plot_conformal_band():
    hours = np.arange(10, 18)
    actual_power = np.array([120, 130, 145, 160, 175, 165, 150, 140])
    pred_power = actual_power + np.random.normal(0, 3, len(hours))
    
    # MAPIE 90% Confidence Interval (+/- 14.5 kW)
    upper_bound = pred_power + 14.5
    lower_bound = pred_power - 14.5
    
    threshold = 180 # Critical Threshold
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(hours, actual_power, marker='x', linestyle='--', color='gray', label='실제 전력량')
    ax.plot(hours, pred_power, marker='o', linewidth=3, color='#007acc', label='AI 점 예측 (Point Prediction)')
    ax.fill_between(hours, lower_bound, upper_bound, color='#007acc', alpha=0.2, label='90% 신뢰 구간 밴드 (MAPIE)')
    
    # Threshold Line
    ax.axhline(y=threshold, color='red', linewidth=2.5, linestyle='-.', label=f'목표 임계치 (Threshold: {threshold}kW)')
    
    # Annotation for crossing threshold
    cross_idx = 4 # 14:00 where upper_bound > threshold
    ax.annotate("⚠️ 신뢰구간 상단 돌파!\n(사전 전력 제어 즉시 실행 필요)",
                xy=(hours[cross_idx], upper_bound[cross_idx]), 
                xytext=(hours[cross_idx]-1.5, upper_bound[cross_idx]+15),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=10))
    
    ax.set_title("📊 등각 예측(Conformal Prediction)을 활용한 선제적 피크 제어", fontsize=16, fontweight='bold')
    ax.set_xlabel("시간 (Hour)", fontsize=12)
    ax.set_ylabel("사용 전력량 (kW)", fontsize=12)
    ax.set_xticks(hours)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "conformal_band.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_rmse_journey()
    plot_peak_conditions()
    plot_conformal_band()
    print("3종 차트 생성 완료.")
