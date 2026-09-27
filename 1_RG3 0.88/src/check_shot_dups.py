import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT.parent / "\u2460Molding"

df_rg3 = pd.read_csv(data_dir / "moldset_labeled_rg3.csv", index_col=0)
df_cn7 = pd.read_csv(data_dir / "moldset_labeled_cn7.csv", index_col=0)
df_rg3['machine'] = 1
df_cn7['machine'] = 0
combined = pd.concat([df_rg3, df_cn7], ignore_index=True)
combined = combined.drop(columns=['Clamp_Open_Position'], errors='ignore')
sensor_cols = [c for c in combined.columns if c not in ('PassOrFail', 'machine')]

shots = combined.drop_duplicates(subset=sensor_cols + ['machine']).copy()

# Are there any duplicate shots?
dups = shots.duplicated(subset=sensor_cols, keep=False)
print(f"Total unique shots: {len(shots)}")
print(f"Number of shots that share exact sensor readings with OTHER shots: {dups.sum()}")
