import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import load_data
from feature_extractor import FeatureExtractor  

# Load data
data = load_data()

# Extract features
extractor = FeatureExtractor()
X = []
y = []

for state, utility in data:
    if state.is_terminal():
        continue
    features = extractor.extract_features(state)
    X.append(features)
    y.append(utility) 

X = np.array(X)
y = np.array(y)

# Models to test
models = {
    "LinearRegression": LinearRegression(),
    "HuberRegressor": HuberRegressor(),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(32,), activation='relu', max_iter=200, random_state=42)
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    results[name] = {"model": model, "mae": mae, "mse": mse, "pred": y_pred}

# # Plot predictions
# plt.figure(figsize=(14, 10))
# for idx, (name, result) in enumerate(results.items(), 1):
#     plt.subplot(2, 2, idx)
#     plt.scatter(y, result["pred"], alpha=0.1, s=5)
#     plt.plot([-1, 1], [-1, 1], 'r--')
#     plt.title(f"{name}\nMAE: {result['mae']:.4f}, MSE: {result['mse']:.4f}")
#     plt.xlabel("True Utility")
#     plt.ylabel("Predicted")

# plt.tight_layout()
# plt.suptitle("Model Comparison: Utility Prediction", fontsize=16, y=1.02)
# plt.show()

# Plot residuals
plt.figure(figsize=(14, 10))
for idx, (name, result) in enumerate(results.items(), 1):
    residuals = result["pred"] - y
    plt.subplot(2, 2, idx)
    plt.scatter(y, residuals, alpha=0.1, s=5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{name} Residuals")
    plt.xlabel("True Utility")
    plt.ylabel("Prediction Error (Residual)")

plt.tight_layout()
plt.suptitle("Residual Plots for Utility Prediction Models", fontsize=16, y=1.02)
plt.show()
