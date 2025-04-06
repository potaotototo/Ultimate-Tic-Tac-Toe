from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import load_data
from feature_extractor_rf import RFFeatureExtractor
from more_meta_feature_extractor import MoreMetaFeatureExtractor
import numpy as np

# Load and prepare data
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
    "Board win diff",             # my_won - opp_won
    "Center control",             # center_control
    "Global contribution",        # global_contrib
    "Win in one",                 # win_in_one
    "Opponent win in one",        # opp_win_in_one
    "Filled ratio",               # filled_ratio
    "Player turn advantage",      # player_turn_advantage
    "Two in a line",              # two_in_line
    "Opponent two in a line",     # opponent_two_in_line
    "Restricted freedom",         # restricted_freedom
    "Blocking opportunities",     # blocking_opportunities
    "Restricted and blocked",     # restricted_and_blocked
    "Meta two in a line",         # meta_two_in_line
    "Meta opponent two in a line" # meta_opp_two_in_line
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
