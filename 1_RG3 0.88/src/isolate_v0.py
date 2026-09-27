import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT.parent / "\u2460Molding"

def main():
    df_rg3 = pd.read_csv(data_dir / "moldset_labeled_rg3.csv", index_col=0)
    df_rg3['machine'] = 1
    
    sensor_cols = [c for c in df_rg3.columns if c not in ('PassOrFail', 'machine', 'Clamp_Open_Position')]
    df_rg3['Virtual_ID'] = df_rg3.groupby(sensor_cols, sort=False).cumcount()
    
    # Filter to Virtual_ID == 0
    v0 = df_rg3[df_rg3['Virtual_ID'] == 0].copy()
    print(f"Total Virtual_ID=0: {len(v0)}")
    print(f"Defects in Virtual_ID=0: {v0['PassOrFail'].sum()}")
    
    X = v0[sensor_cols]
    y = v0['PassOrFail']
    
    # Fit a decision tree with limited depth to see if any simple rule isolates them
    dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    dt.fit(X, y)
    
    print("\nDecision Tree Rules:")
    print(export_text(dt, feature_names=sensor_cols))
    
    # How well does it separate?
    preds = dt.predict(X)
    print(f"\nConfusion Matrix on Training Data (Depth 3):")
    print(pd.crosstab(y, preds, rownames=['Actual'], colnames=['Predicted']))
    
if __name__ == "__main__":
    main()
