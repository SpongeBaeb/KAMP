import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def run_eda():
    print("Loading labeled CN7 data...")
    df = pd.read_csv('data/moldset_labeled_cn7.csv')
    
    # 1. Data Cleaning
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Separate features and target
    target = 'PassOrFail'
    features = [c for c in df.columns if c != target]
    
    print(f"Total rows: {len(df)}, Defects: {df[target].sum()} ({(df[target].sum()/len(df))*100:.2f}%)")
    
    # 2. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(16, 12))
    corr = df[features].corr()
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    plt.title("Sensor Correlation Heatmap (CN7 Molding)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/01_correlation_heatmap.png', dpi=300)
    plt.close()
    
    # 3. Defect vs Normal Distribution (Boxplots)
    print("Generating Distribution Plots...")
    
    # We have 23 features left. Let's plot them in a 6x4 grid.
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.boxplot(x=target, y=feature, data=df, ax=axes[i], palette={0: 'steelblue', 1: 'red'}, hue=target, legend=False, showfliers=False)
            axes[i].set_title(feature, fontsize=10)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            
            # Highlight differences visually
            axes[i].grid(True, linestyle='--', alpha=0.6)
            
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.suptitle("Sensor Distributions: Normal (0) vs Defect (1)", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/01_defect_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Point-Biserial Correlation (Correlation between continuous features and binary target)
    print("Calculating Feature vs Target correlation...")
    target_corr = df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if val < 0 else 'steelblue' for val in target_corr.values]
    sns.barplot(x=target_corr.values, y=target_corr.index, palette=colors)
    plt.title("Feature Correlation with Defect (PassOrFail)", fontsize=14, fontweight='bold')
    plt.xlabel("Pearson Correlation Coefficient")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/01_target_correlation.png', dpi=300)
    plt.close()
    
    print("\nTop 5 Most Correlated Features with Defect:")
    print(target_corr.head(5))
    
    print("\nEDA complete. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    run_eda()
