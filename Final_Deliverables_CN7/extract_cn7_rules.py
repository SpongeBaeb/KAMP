import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings
warnings.filterwarnings('ignore')

file_path = 'c:/Hackerton/Final_Deliverables_CN7/2_Preprocessed_Data_CN7.csv'
df = pd.read_csv(file_path)

# 캐비티 0번만 추출 (불량이 주로 발생하는 곳)
df_v0 = df[df['Virtual_ID'] == 0]

# 주요 특징 변수
features = ['EMA_Injection_Time', 'EMA_Mold_Temp_3', 'Average_Back_Pressure', 'Max_Injection_Pressure', 'Mold_Temperature_3']
X = df_v0[features]
y = df_v0['PassOrFail']

dt = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
dt.fit(X, y)

print("="*60)
print(" 🤖 [CN7] AI가 불량을 걸러내는 실제 판단 기준 (Decision Tree)")
print("="*60)
tree_rules = export_text(dt, feature_names=features)
print(tree_rules)
