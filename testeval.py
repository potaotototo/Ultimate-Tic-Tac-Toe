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

        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)
        board_win_diff = my_won - opp_won

        valid_actions = len(state.get_all_valid_actions()) / 81

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

        available_boards = np.sum(lbs == 0) / 9
        filled_ratio = np.sum(state.board != 0) / 729

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
        two_in_line = 0
        opp_two_in_line = 0
        for line in win_lines:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1
            if cells.count(opp_fill) == 2 and cells.count(0) == 1:
                opp_two_in_line += 1

        features = np.array([
            board_win_diff,
            valid_actions,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            available_boards,
            filled_ratio,
            two_in_line,
            opp_two_in_line
        ])

        if self.model == "huber":
            weights = np.array([
                +0.0381,
                +0.0260,
                +0.0988,
                +0.9838,
                -0.9684,
                -0.0594,
                +0.2225,
                -0.2416,
                -0.4433
            ])
            intercept = 0.0629 + 0.2
        elif self.model == "linear":
            weights = np.array([
                +0.0333,
                +0.0191,
                +0.0822,
                +0.9999,
                -0.9955,
                -0.0162,
                +0.1776,
                -0.1737,
                -0.3151
            ])
            intercept = 0.0217
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

class LassoSlimStudentAgent:
    def evaluate(self, state):
        if state.is_terminal():
            return state.terminal_utility()

        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        # Selected features
        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)
        board_win_diff = my_won - opp_won

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

        filled_ratio = np.sum(state.board != 0) / 729

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
        two_in_line = 0
        for line in win_lines:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1

        # Lasso-selected features and weights
        features = np.array([
            board_win_diff,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            filled_ratio,
            two_in_line
        ])

        weights = np.array([
            +0.0397,   # board_win_diff
            +0.1920,   # global_contrib
            +0.9511,   # win_in_1
            -0.9912,   # opp_win_in_1
            +0.1723,   # filled_ratio
            +0.05    # two_in_line
        ])
        intercept = 0.0034

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
            if state.is_terminal():
                continue
            pred = self.agent.evaluate(state)
            predictions.append(pred)
            true_values.append(true_utility)

        predictions = np.array(predictions)
        true_values = np.array(true_values)

        mae = np.mean(np.abs(predictions - true_values))
        print(f"Mean Absolute Error (no normalization): {mae:.4f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predictions, alpha=0.1, s=5)
        plt.xlabel("True Utility (scaled to [-1, 1])")
        plt.ylabel("Predicted Heuristic (linear model)")
        plt.title(f"Hardcoded Heuristic vs True Utility")
        plt.plot([-1, 1], [-1, 1], 'r--')
        plt.grid(True)
        plt.show()

# agent = StudentAgent(model="linear")
agent = LassoSlimStudentAgent()
tester = TestEval(agent)
tester.evaluate_heuristic_accuracy()
