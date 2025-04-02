import numpy as np
import matplotlib.pyplot as plt
from utils import load_data

class StudentAgent:
    def evaluate(self, state):
        if state.is_terminal():
            return state.terminal_utility()

        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        # Feature: Board win diff
        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)
        board_win_diff = my_won - opp_won

        # Feature: Valid actions
        valid_actions = len(state.get_all_valid_actions()) / 81

        # Feature: Global contrib
        global_weights = np.array([
            [2, 1, 2],
            [1, 3, 1],
            [2, 1, 2]
        ])
        global_contrib = np.sum(global_weights * (lbs == my_fill)) - np.sum(global_weights * (lbs == opp_fill))

        # Features: Win-in-one
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

        # Feature: Available boards
        available_boards = np.sum(lbs == 0) / 9

        # Feature: Control 2-in-a-line globally
        win_lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]
        control_2_in_line = 0
        for line in win_lines:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                control_2_in_line += 1

        # Hardcoded weights from forest importances
        weights = np.array([
            0.03899147163501777,   # board_win_diff
            0.07640277836194594,   # valid_actions
            0.08444654699233495,   # global_contrib
            0.08510008176362917,   # win_in_one
            -0.12516325866018674,  # opp_win_in_one
            0.13267551517748227,   # available_boards (not in final importance list, keep 0)
            0.45722034740940315    # control_2_in_line
        ])

        features = np.array([
            board_win_diff,
            valid_actions,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            available_boards,
            control_2_in_line
        ])

        return float(np.dot(weights, features))

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
            if state.is_terminal():
                continue
            pred = self.agent.evaluate(state)
            predictions.append(pred)
            true_values.append(true_utility)

        predictions = np.array(predictions)
        true_values = np.array(true_values)

        pred_min, pred_max = predictions.min(), predictions.max()
        norm_predictions = 2 * (predictions - pred_min) / (pred_max - pred_min) - 1

        mae = np.mean(np.abs(norm_predictions - true_values))
        print(f"Mean Absolute Error (normalized): {mae:.4f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, norm_predictions, alpha=0.1, s=5)
        plt.xlabel("True Utility")
        plt.ylabel("Predicted (Normalized) Heuristic")
        plt.title("Hardcoded Heuristic vs True Utility")
        plt.grid(True)
        plt.show()

# Run evaluation
agent = StudentAgent()
tester = TestEval(agent)
tester.evaluate_heuristic_accuracy()
