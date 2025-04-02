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

        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)

        center = lbs[1][1]
        center_bonus = 1 if center == my_fill else -1 if center == opp_fill else 0

        local_threats = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] != 0:
                    continue
                local_threats += self.quick_local_threat(board[i][j], my_fill)
                local_threats -= self.quick_local_threat(board[i][j], opp_fill)

        score = (
            10 * (my_won - opp_won) +
             2 * center_bonus +
             1 * local_threats
        )
        return score

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


class TestEval:
    def __init__(self, agent):
        self.agent = agent
        self.data = load_data() 

    def evaluate_heuristic_accuracy(self):
        predictions = []
        true_values = []

        for state, true_utility in self.data:
            pred = self.agent.evaluate(state)
            predictions.append(pred)
            true_values.append(true_utility)

        predictions = np.array(predictions)
        true_values = np.array(true_values)

        # Normalize predictions to [0, 1]
        pred_min, pred_max = predictions.min(), predictions.max()
        norm_predictions = 2 * (predictions - pred_min) / (pred_max - pred_min) - 1

        mae = np.mean(np.abs(norm_predictions - true_values))
        print(f"Mean Absolute Error (normalized): {mae:.4f}")

        # Plot prediction vs true utility
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, norm_predictions, alpha=0.1, s=5)
        plt.xlabel("True Utility")
        plt.ylabel("Predicted (Normalized) Heuristic")
        plt.title("Heuristic vs True Utility (n = {})".format(len(true_values)))
        plt.grid(True)
        plt.show()

agent = StudentAgent()
tester = TestEval(agent)
tester.evaluate_heuristic_accuracy()