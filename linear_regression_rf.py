import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from utils import load_data
from feature_extractor_rf import RFFeatureExtractor

# Load data and extract features
data = load_data()
extractor = RFFeatureExtractor()

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

# Predictions and loss calculation
predictions = model.predict(X)

# Calculate Mean Squared Error and Mean Absolute Error
mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)

# Feature names
feature_names = [
    "Board win diff",
    "Central board"
    "Global contrib",
    "Win in 1",
    "Filled ratio",
    "Player turn",
    "Two in a line",
    "Opponent two in a line",
    "Restricted and blocked"
]

print("Linear Regression Weights:")
for name, w in zip(feature_names, weights):
    print(f"{name:25s}: {w:+.4f}")
print(f"Intercept (bias term): {intercept:+.4f}")

# Calculate loss at each iteration (training loss) for plotting
losses = []
for i in range(len(y)):
    mse_loss = mean_squared_error(y[:i+1], model.predict(X[:i+1]))
    losses.append(mse_loss)

# Plotting the loss
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Mean Squared Error')
plt.xlabel('Training Data Points')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss vs Data Points')
plt.legend()
plt.grid(True)
plt.show()
