# Re-run the previous cell since the code environment was reset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from utils import load_data

class FeatureExtractor:
    def extract_features(self, state):
        if state.is_terminal():
            return np.zeros(8)

        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        # Feature 1: Local boards won
        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)

        # Feature 3: Local threats
        local_threats = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] != 0:
                    continue
                local_threats += self.quick_local_threat(board[i][j], my_fill)
                local_threats -= self.quick_local_threat(board[i][j], opp_fill)

        # Feature 4: Number of valid actions (normalized)
        num_valid_actions = len(state.get_all_valid_actions()) / 81

        # Feature 5: Opponent forced into a bad board
        next_i, next_j = state.prev_local_action if state.prev_local_action else (None, None)
        opp_forced_bad_board = 0
        if next_i is not None:
            next_status = lbs[next_i][next_j]
            if next_status == 0:
                next_board = board[next_i][next_j]
                if np.count_nonzero(next_board) >= 7:
                    opp_forced_bad_board = 1

        # Feature 6: Number of tied boards (normalized)
        num_tied = np.sum(lbs == 3) / 9

        # Feature 7: Partial (unfinished) boards in progress (normalized)
        partial_boards = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:
                    filled = np.count_nonzero(board[i][j])
                    if 0 < filled < 9:
                        partial_boards += 1
        partial_boards /= 9

        # Feature 8: Global win contribution (weighted board control)
        global_weights = np.array([
            [2, 1, 2],
            [1, 3, 1],
            [2, 1, 2]
        ])
        global_contrib = np.sum(global_weights * (lbs == my_fill)) - np.sum(global_weights * (lbs == opp_fill))

        # Feature 9: Can win a local board in one move
        win_in_one = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0 and self.has_local_win_in_one(board[i][j], my_fill):
                    win_in_one += 1
        win_in_one = win_in_one / 9  # normalize

        return np.array([
            my_won - opp_won,
            local_threats,
            num_valid_actions,
            opp_forced_bad_board,
            num_tied,
            partial_boards,
            global_contrib,
            win_in_one
        ])

    def quick_local_threat(self, sub_board, fill):
        threats = 0
        for i in range(3):
            row = sub_board[i, :]
            col = sub_board[:, i]
            if np.count_nonzero(row == fill) == 2 and np.count_nonzero(row == 0) == 1:
                threats += 1
            if np.count_nonzero(col == fill) == 2 and np.count_nonzero(col == 0) == 1:
                threats += 1
        diag1 = [sub_board[i, i] for i in range(3)]
        diag2 = [sub_board[i, 2 - i] for i in range(3)]
        if diag1.count(fill) == 2 and diag1.count(0) == 1:
            threats += 1
        if diag2.count(fill) == 2 and diag2.count(0) == 1:
            threats += 1
        return threats
    
    def has_local_win_in_one(self, sub_board, fill):
        for i in range(3):
            row = sub_board[i, :]
            col = sub_board[:, i]
            if np.count_nonzero(row == fill) == 2 and np.count_nonzero(row == 0) == 1:
                return True
            if np.count_nonzero(col == fill) == 2 and np.count_nonzero(col == 0) == 1:
                return True
        diag1 = [sub_board[i, i] for i in range(3)]
        diag2 = [sub_board[i, 2 - i] for i in range(3)]
        if diag1.count(fill) == 2 and diag1.count(0) == 1:
            return True
        if diag2.count(fill) == 2 and diag2.count(0) == 1:
            return True
        return False

# Load data and extract features
data = load_data()
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

# Use Huber Loss
model = HuberRegressor()
model.fit(X, y)

# Output learned weights
weights = model.coef_
intercept = model.intercept_
print(f"Learned weights: {weights.round(4)}, intercept: {intercept:.4f}")

# Predict and plot
y_pred = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.1, s=5)
plt.plot([-1, 1], [-1, 1], color='red', linestyle='--', linewidth=1)
plt.xlabel("True Utility (scaled to [-1, 1])")
plt.ylabel("Predicted Heuristic (linear model)")
plt.title("Regression Heuristic vs True Utility (8 features)")
plt.grid(True)
plt.show()
