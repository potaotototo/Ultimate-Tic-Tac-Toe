import numpy as np
import matplotlib.pyplot as plt
from utils import load_data

class StudentAgent:
    def __init__(self, model="huber"):
        self.model = model.lower()

    def evaluate(self, state):
        if state.is_terminal():
            return state.terminal_utility()
        
        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        # Feature 1: Local boards won
        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)

        # Feature 2: Centre priority (revived)
        center_control = 0
        if board[1][1][1, 1] == my_fill:  
            center_control = 1
        elif board[1][1][1, 1] == opp_fill:  
            center_control = -1

        # Feature 8: Global win contribution (weighted board control)
        global_weights = np.array([
            [2, 1, 2],
            [1, 3, 1],
            [2, 1, 2]
        ])
        global_contrib = np.sum(global_weights * (lbs == my_fill)) - np.sum(global_weights * (lbs == opp_fill))

        # Feature 9: Can win a local board in one move
        # Feature 10: Opponent can win a local board in one move
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

        # Feature 13: Number of filled cells
        filled_ratio = np.sum(state.board != 0) / 81 / 9

        # Feature 14: Freedom of next move
        freedom = 0.0
        if state.prev_local_action is not None:
            i, j = state.prev_local_action
            if state.local_board_status[i][j] != 0:
                freedom = 1.0

        # Feature 15: Track if the opponent is about to win local board
        blocking_opportunities = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:  # Empty spot
                    local_board = board[i][j]
                    if self.has_local_win_in_one(local_board, opp_fill):
                        blocking_opportunities += 1
        blocking_opportunities /= 9

        # Feature 16: Prioritise player turn
        player_turn_advantage = 0.1 if state.fill_num % 2 == 0 else -0.1

        features = np.array([
            my_won - opp_won,
            center_control,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            filled_ratio,
            freedom,
            blocking_opportunities,
            player_turn_advantage
        ])

        if self.model == "huber":
            weights = np.array([
                +0.0800,
                +0.0448,
                +0.1467,
                +1.1254,
                -0.5692,
                +0.0229,
                +0.0000,
                -0.5692,
                -1.5228
            ])
            intercept = -0.0004
        elif self.model == "linear":
            weights = np.array([
                +0.0852,
                +0.0393,
                +0.1201,
                +1.1178,
                -0.5653,
                -0.0377,
                +0.0000,
                -0.5653,
                -1.9424
            ])
            intercept = +0.0033
        elif self.model == "ridge":
            weights = np.array([
                +0.0851,
                +0.0393,
                +0.1201,
                +1.1165,
                -0.5649,
                -0.0356,
                +0.0000,
                -0.5649,
                -1.9398
            ])
            intercept = +0.0033
        elif self.model == "lasso":
            weights = np.array([
                +0.0803,
                +0.0374,
                +0.1190,
                +1.0240,
                -1.0400,
                -0.0000,
                +0.0000,
                -0.0000,
                -1.8309
            ])
            intercept = +0.0024
        else:
            raise ValueError("Unknown model type specified.")
        
        return float(np.dot(weights, features) + intercept)

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

class TestEval:
    def __init__(self, agent):
        self.agent = agent
        self.data = load_data()

    def evaluate_heuristic_accuracy(self):
        predictions = []
        true_values = []

        for state, true_utility in self.data:
            # Skip terminal states for evaluation
            if state.is_terminal():
                continue

            # Predict heuristic for non-terminal states
            pred = self.agent.evaluate(state)
            predictions.append(pred)
            true_values.append(true_utility)

        predictions = np.array(predictions)
        true_values = np.array(true_values)

        # Calculate and print Mean Absolute Error (MAE)
        mae = np.mean(np.abs(predictions - true_values))
        print(f"Mean Absolute Error (no normalization): {mae:.4f}")

        # Scatter plot of true vs predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predictions, alpha=0.1, s=5)
        plt.xlabel("True Utility (scaled to [-1, 1])")
        plt.ylabel("Predicted Heuristic (linear model)")
        plt.title(f"Hardcoded Heuristic vs True Utility")
        plt.plot([-1, 1], [-1, 1], 'r--')  # Identity line
        plt.grid(True)
        plt.show()

# Run the test
agent = StudentAgent(model="lasso")  
tester = TestEval(agent)
tester.evaluate_heuristic_accuracy()
