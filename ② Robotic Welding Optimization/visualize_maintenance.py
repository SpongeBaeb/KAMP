import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_maintenance_visualization():
    print("Loading original raw data...")
    # Load the full data before isolation
    df = pd.read_csv('preprocessed_raw.csv')
    df['date'] = pd.to_datetime(df['working time']).dt.date
    
    # Define maintenance days
    maintenance_days = ['2020-03-25', '2020-03-27', '2020-03-31', '2020-04-03']
    maintenance_dates = pd.to_datetime(maintenance_days).date
    
    # Mark the data
    df['is_maintenance'] = df['date'].isin(maintenance_dates)
    
    # We will plot Weld Force and Weld Current as they are the most indicative
    plt.figure(figsize=(15, 10))
    
    # 1. Weld Force Plot
    plt.subplot(2, 1, 1)
    sns.scatterplot(x='date', y='weld force(bar)', hue='is_maintenance', 
                    palette={True: 'red', False: 'blue'}, data=df, alpha=0.5, s=20)
    plt.title("Weld Force(bar) over Time: Why we dropped Maintenance Days", fontsize=14, fontweight='bold')
    plt.ylabel("Weld Force (bar)", fontsize=12)
    plt.xlabel("")
    plt.axhline(y=df[~df['is_maintenance']]['weld force(bar)'].mean(), color='black', linestyle='--', label='Normal Mean')
    plt.legend(title='Is Maintenance/Calibration?', loc='upper right')
    
    # Highlight the maintenance days with shaded background
    for md in maintenance_dates:
        plt.axvspan(md, md, color='red', alpha=0.2)
        
    # 2. Weld Current Plot
    plt.subplot(2, 1, 2)
    sns.scatterplot(x='date', y='weld current(kA)', hue='is_maintenance', 
                    palette={True: 'red', False: 'blue'}, data=df, alpha=0.5, s=20)
    plt.title("Weld Current(kA) over Time: Extreme Testing Anomalies", fontsize=14, fontweight='bold')
    plt.ylabel("Weld Current (kA)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(title='Is Maintenance/Calibration?', loc='upper right')
    
    for md in maintenance_dates:
        plt.axvspan(md, md, color='red', alpha=0.2)
        
    plt.tight_layout()
    plt.savefig('plots/maintenance_justification.png', dpi=300)
    plt.close()
    print("Visualization saved to 'plots/maintenance_justification.png'")

if __name__ == "__main__":
    create_maintenance_visualization()
