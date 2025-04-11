import numpy as np
from sklearn.linear_model import LinearRegression
from utils import load_data
from feature_extractor_defense import DefensiveFeatureExtractor

# Load data and extract features
data = load_data()
extractor = DefensiveFeatureExtractor()

X, y = [], []
for state, utility in data:
    if state.is_terminal():
        continue
    X.append(extractor.extract_features(state))
    y.append(utility) 

X = np.array(X)
y = np.array(y)

model = LinearRegression()
model.fit(X, y)

weights = model.coef_
intercept = model.intercept_

# Feature names
feature_names = [
    "Board win diff",
    "Central control",
    "Global contrib",
    "Win in 1",
    "Filled ratio",
    "Player turn advantage",
    "Two in a line",
    "Opponent two in a line",
    "Meta opp two in a line",
    "Risky ratio"
]

print("Linear Regression Weights:")
for name, w in zip(feature_names, weights):
    print(f"{name:25s}: {w:+.4f}")
print(f"Intercept (bias term): {intercept:+.4f}")
