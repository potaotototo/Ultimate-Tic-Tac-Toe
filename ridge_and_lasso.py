from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import load_data
from feature_extractor import FeatureExtractor
import numpy as np

# Load and prepare data
data = load_data()
extractor = FeatureExtractor()
X, y = [], []

for state, utility in data:
    if state.is_terminal():
        continue
    X.append(extractor.extract_features(state))
    y.append(utility)

X = np.array(X)
y = np.array(y)

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.001)

ridge.fit(X, y)
lasso.fit(X, y)

ridge_pred = ridge.predict(X)
lasso_pred = lasso.predict(X)

ridge_mae = mean_absolute_error(y, ridge_pred)
lasso_mae = mean_absolute_error(y, lasso_pred)

ridge_mse = mean_squared_error(y, ridge_pred)
lasso_mse = mean_squared_error(y, lasso_pred)

feature_names = [
    "Board win diff",
    "Valid actions",
    "Global contrib",
    "Win in 1",
    "Opponent win in 1",
    "Filled ratio",
    "2-in-a-line",
    "Freedom move"
]

print(f"Ridge MAE: {ridge_mae:.4f}, MSE: {ridge_mse:.4f}")
print(f"Lasso MAE: {lasso_mae:.4f}, MSE: {lasso_mse:.4f}")

print("\nRidge Coefficients:")
for name, coef in zip(feature_names, ridge.coef_):
    print(f"{name:25s}: {coef:+.4f}")
print(f"Intercept: {ridge.intercept_:+.4f}")

print("\nLasso Coefficients:")
for name, coef in zip(feature_names, lasso.coef_):
    print(f"{name:25s}: {coef:+.4f}")
print(f"Intercept: {lasso.intercept_:+.4f}")
