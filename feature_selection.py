import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import State, load_data 

# This should contain more features than what we use to train our weights later on
class ExtendedFeatureExtractor:
    def extract_features(self, state):
        # Extract features based on the state (use all available features here)
        
        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)

        center_control = 0
        if board[1][1][1, 1] == my_fill:
            center_control = 1
        elif board[1][1][1, 1] == opp_fill:
            center_control = -1

        global_weights = np.array([
            [2, 1, 2],
            [1, 3, 1],
            [2, 1, 2]
        ])
        global_contrib = np.sum(global_weights * (lbs == my_fill)) - np.sum(global_weights * (lbs == opp_fill))

        win_in_one = 0
        opp_win_in_one = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] != 0:
                    continue
                local = board[i][j]
                if self.has_local_win_in_one(local, my_fill):
                    win_in_one += 1
                if self.has_local_win_in_one(local, opp_fill):
                    opp_win_in_one += 1
        win_in_one /= 9
        opp_win_in_one /= 9

        filled_ratio = np.sum(state.board != 0) / 81 / 9

        freedom = 0.0
        if state.prev_local_action is not None:
            i, j = state.prev_local_action
            if state.local_board_status[i][j] != 0:
                freedom = 1.0

        blocking_opportunities = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:  # Empty spot
                    local_board = board[i][j]
                    if self.has_local_win_in_one(local_board, opp_fill):
                        blocking_opportunities += 1
        blocking_opportunities /= 9

        safe_opponent = (1 - opp_win_in_one) * blocking_opportunities

        player_turn_advantage = 0.1 if state.fill_num % 2 == 0 else -0.1

        two_in_line = 0
        for line in [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1

        opponent_two_in_line = 0
        for line in [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(opp_fill) == 2 and cells.count(0) == 1:
                opponent_two_in_line += 1

        available_subboards = np.sum(lbs == 0) / 9  
        boards_ahead = np.sum(lbs == my_fill) / 9  

        restricted_freedom = 0.0
        total_available_moves = 9  

        opponent_valid_moves = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:  
                    opponent_valid_moves += 1

        restricted_freedom = 1 - (opponent_valid_moves / total_available_moves)

        restricted_and_blocked = restricted_freedom + blocking_opportunities


        features = np.array([
            my_won - opp_won, 
            center_control, 
            global_contrib, 
            win_in_one, 
            opp_win_in_one, 
            filled_ratio, 
            freedom, 
            blocking_opportunities, 
            safe_opponent,
            player_turn_advantage, 
            two_in_line, 
            opponent_two_in_line, 
            available_subboards, 
            boards_ahead,
            restricted_freedom,
            restricted_and_blocked
        ])

        return features

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


# Function to collect data (extract features and corresponding labels)
def get_data():
    feature_extractor = ExtendedFeatureExtractor()
    data = [] 
    for state, true_utility in load_data(): 
        if state.is_terminal():  # skip terminal states
            continue
        features = feature_extractor.extract_features(state)
        data.append((features, true_utility))

    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    return features, labels

features, labels = get_data()

# Random forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(features, labels)

# Feature importances
feature_importances = rf.feature_importances_

threshold = 0.04
indices = np.where(feature_importances > threshold)[0]  # Indices where importance is greater than threshold

# Select the features with indices above the threshold
selected_features = features[:, indices]

# Train with linear regression using the selected features
model = LinearRegression()
model.fit(selected_features, labels)

# Evaluate the new model
predictions = model.predict(selected_features)
mae = mean_absolute_error(labels, predictions)
mse = mean_squared_error(labels, predictions)

# Output the evaluation metrics
print(f'Mean Absolute Error: {mae:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

print("Trained Model Coefficients:")
for idx, coef in zip(indices, model.coef_):
    print(f"Feature {idx} - Coefficient: {coef:.4f}")

print(f"Intercept: {model.intercept_:.4f}")

feature_names = [
    "Local boards won", "Center control", "Global contribution", 
    "Win in one", "Opponent win in one", "Filled ratio", 
    "Freedom of next move", "Blocking opportunities", "Safe opponent move", 
    "Player turn advantage", "Two in a line", "Opponent two in a line",
    "Available local boards", "Boards ahead", "Restricted opponent", 
    "Restricted and blocked"
]

plt.figure(figsize=(14, 6))
plt.barh(range(len(feature_importances)), feature_importances, color='skyblue')
plt.yticks(range(len(feature_importances)), feature_names)

plt.xlabel("Feature Importance")
plt.ylabel("Feature Index")
plt.title("Feature Importances from Random Forest")

plt.show()
