import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from utils import load_data
from more_meta_feature_extractor import MoreMetaFeatureExtractor

data = load_data()
extractor = MoreMetaFeatureExtractor()
X, y = [], []

for state, utility in data:
    if state.is_terminal():
        continue
    X.append(extractor.extract_features(state))
    y.append(utility)

X = np.array(X)
y = np.array(y)

# Train models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.001)
linear = LinearRegression()

ridge.fit(X, y)
lasso.fit(X, y)
linear.fit(X, y)

# Predictions
ridge_pred = ridge.predict(X)
lasso_pred = lasso.predict(X)
linear_pred = linear.predict(X)

# Loss plotting
losses_ridge = [mean_squared_error(y[:i+1], ridge_pred[:i+1]) for i in range(len(y))]
losses_lasso = [mean_squared_error(y[:i+1], lasso_pred[:i+1]) for i in range(len(y))]
losses_linear = [mean_squared_error(y[:i+1], linear_pred[:i+1]) for i in range(len(y))]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(losses_ridge, label="Ridge MSE", alpha=0.8)
plt.plot(losses_lasso, label="Lasso MSE", alpha=0.8)
plt.plot(losses_linear, label="Linear MSE", alpha=0.8)
plt.xlabel("Training Data Points")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss vs Data Points")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
