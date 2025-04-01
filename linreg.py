import pickle
import numpy as np
from utils import load_data
from sklearn.linear_model import LinearRegression

# Load data
data = load_data()

# Extract X (features) and y (target)
X = np.array([state.board.flatten() for state, utility in data])  # Flatten board
y = np.array([utility for state, utility in data])  # Utility as target

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save learned weights
model_weights = model.coef_
model_intercept = model.intercept_

print("Learned Weights:", model_weights)
print("Intercept:", model_intercept)
