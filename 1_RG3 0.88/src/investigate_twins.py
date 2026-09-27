import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT.parent / "①Molding"

df_rg3 = pd.read_csv(data_dir / "moldset_labeled_rg3.csv", index_col=0)
df_cn7 = pd.read_csv(data_dir / "moldset_labeled_cn7.csv", index_col=0)

df_rg3['machine'] = 1
df_cn7['machine'] = 0
combined = pd.concat([df_rg3, df_cn7], ignore_index=True)
combined = combined.drop(columns=['Clamp_Open_Position'], errors='ignore')

feat_cols = [c for c in combined.columns if c not in ('PassOrFail', 'machine')]

combined['is_dup'] = combined.duplicated(subset=feat_cols, keep=False)
twins = combined[combined['is_dup']]

print(f"Total twins: {len(twins)}")
print("Twin indices:")
print(twins.index.tolist()[:50])

# Are twins consecutive?
# Let's sort by features and print the indices to see their original row numbers.
twins_sorted = twins.sort_values(by=feat_cols)
print("\nFirst 10 twin pairs (original indices):")
current_pair = []
for idx in twins_sorted.index[:20]:
    print(idx, twins.loc[idx, 'PassOrFail'])
