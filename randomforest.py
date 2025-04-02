import numpy as np
from sklearn.ensemble import RandomForestRegressor
from utils import load_data
from feature_extractor import FeatureExtractor

# Load and prepare data
data = load_data()
extractor = FeatureExtractor()
X, y = [], []

for state, utility in data:
    if state.is_terminal():
        continue
    features = extractor.extract_features(state)
    X.append(features)
    y.append(utility)

X = np.array(X)
y = np.array(y)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X, y)
importances = rf_model.feature_importances_

# Feature names for readability
feature_names = [
    "Board win diff",
    "Valid actions",
    "Global contrib",
    "Win in 1",
    "Opponent win in 1",
    "Available boards",
    "2-in-a-line global"
]

# Normalize importances to sum to 1
normalized = importances / np.sum(importances)
scaled_weights = normalized 

# Combine and sort by magnitude
final_weights = list(zip(feature_names, scaled_weights))
final_weights.sort(key=lambda x: -abs(x[1]))
print(final_weights)
