import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import lightgbm as lgb
import json
import os

# Set Korean font for matplotlib (Windows)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# Set page config
st.set_page_config(page_title="PowerPeak Simulator", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for Premium Dark Mode & Glassmorphism
st.markdown("""
<style>
    :root {
        --primary-color: #00d2ff;
        --secondary-color: #3a7bd5;
        --bg-color: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --text-color: #f1f5f9;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Cards */
    div.css-1r6slb0, div.css-12oz5g7 {
        background: var(--card-bg) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div.css-1r6slb0:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.2);
    }

    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px var(--primary-color);
    }
    
    /* Metric Cards */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ PowerPeak AI Simulator")
st.markdown("제조 현장의 전력사용량(24시간) 예측 및 최적화 시나리오 시뮬레이터")

# Sidebar for inputs
with st.sidebar:
    st.header("🎛️ 시나리오 파라미터 입력")
    
    prod_target = st.slider("일일 생산 목표량 (단위)", min_value=0, max_value=500, value=250)
    workers = st.slider("투입 공장인원 (명)", min_value=10, max_value=200, value=80)
    temp = st.slider("예상 기온 (℃)", min_value=-15.0, max_value=40.0, value=22.0)
    wind = st.slider("예상 풍속 (m/s)", min_value=0.0, max_value=20.0, value=3.5)
    
    st.markdown("---")
    st.markdown("### 💡 가이드라인 제안")
    st.info("- 경부하 시간대(야간)에 에너지 집약 공정 배치\n- 대형 설비 기동 시차 분산 배치")
    
    run_sim = st.button("🚀 시뮬레이션 실행")

# Main content
col1, col2 = st.columns([2, 1])

if run_sim:
    # Generate dummy simulated predictions for 24 hours based on inputs
    # In a real scenario, we would load the trained LSTM + LGBM models and run inference
    hours = np.arange(24)
    base_load = 50 + (workers * 0.2) + (abs(temp - 20) * 1.5)
    
    # Simulate a daily curve (higher during day, lower at night)
    time_multiplier = np.sin((hours - 6) * np.pi / 12) * 0.4 + 1.0
    
    pred_power = base_load * time_multiplier + np.random.normal(0, 5, 24)
    pred_power = np.clip(pred_power, 20, 300) # clip to realistic bounds
    
    # Calculate Risk
    peak_val = np.max(pred_power)
    peak_hour = np.argmax(pred_power)
    threshold = 187.0 # from XAI analysis
    
    with col1:
        st.subheader("📈 24시간 예측 부하 곡선")
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Conformal Prediction Band (MAPIE)
        q_90 = 14.5 # 90% 신뢰 구간 밴드 폭 (실험 4 캘리브레이션 결과)
        upper_bound = pred_power + q_90
        lower_bound = pred_power - q_90
        
        # Plot curve and MAPIE band
        ax.plot(hours, pred_power, color='#00d2ff', linewidth=3, marker='o', markersize=6, label='점 추정 (Point Prediction)')
        ax.fill_between(hours, lower_bound, upper_bound, alpha=0.3, color='#00d2ff', label='90% 등각 예측 신뢰구간 (MAPIE)')
        
        ax.axhline(y=threshold, color='#ff4b4b', linestyle='--', linewidth=2, label='상위 5% 위험 임계값 (187.0)')
        
        ax.set_xlabel("시간 (Hour)")
        ax.set_ylabel("최대수요전력 (Peak)")
        ax.set_xticks(hours)
        ax.legend()
        ax.grid(color='#333333', linestyle=':', linewidth=1)
        
        # Remove borders for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        st.pyplot(fig)

    with col2:
        st.subheader("📊 리스크 분석 및 최적화")
        
        st.markdown(f"**점 추정 피크치:** <span class='metric-value'>{peak_val:.1f}</span> kW", unsafe_allow_html=True)
        st.markdown(f"**90% 신뢰 구간 밴드:** {peak_val-14.5:.1f} kW ~ {peak_val+14.5:.1f} kW (MAPIE)")
        st.markdown(f"**피크 발생 예상 시간:** {peak_hour}시")
        
        if peak_val > threshold:
            st.error(f"⚠️ 임계값 초과 예상! {peak_hour}시에 전력 초과 과금 리스크가 높습니다.")
            
            # Recommendation
            st.markdown("""
            **권장 스케줄링 조치:**
            1. 인력 투입을 조절하여 공정 속도 완화
            2. 주요 설비 A라인 기동을 2시간 지연
            3. 공조 설비 온도 설정 완화
            """)
            
            # Calculate dummy savings
            savings = (peak_val - threshold) * 109.8 * 1.5 * 30 # monthly estimation
            st.success(f"💡 권장 스케줄 적용 시 월간 예상 전기요금 절감액: **약 {savings:,.0f}원**")
        else:
            st.success("✅ 안전 구역입니다. 현재 스케줄대로 진행해도 무방합니다.")
            
        st.markdown("---")
        st.markdown("### 추출된 위험 조건 (XAI)")
        st.code("""
If Peak_MA3 <= 149.83 and 생산량 > 179.50 and Peak_MA24 <= 37.73:
    => 피크 위험군 (Class 1)
        """)
else:
    with col1:
        st.info("좌측 사이드바에서 시나리오 파라미터를 설정하고 '시뮬레이션 실행' 버튼을 눌러주세요.")
