import numpy as np
from utils import State, Action
from killer_more_depth_agent import KillerAgent

def generate_win_in_one_table():
        table = {}
        all_cells = [0, 1, 2]
        from itertools import product

        for config in product(all_cells, repeat=9):
            board = np.array(config).reshape(3, 3)
            for fill in [1, 2]:
                count = 0
                for i in range(3):
                    row = board[i, :]
                    col = board[:, i]
                    if np.count_nonzero(row == fill) == 2 and np.count_nonzero(row == 0) == 1:
                        count += 1
                    if np.count_nonzero(col == fill) == 2 and np.count_nonzero(col == 0) == 1:
                        count += 1
                diag1 = [board[i, i] for i in range(3)]
                diag2 = [board[i, 2 - i] for i in range(3)]
                if diag1.count(fill) == 2 and diag1.count(0) == 1:
                    count += 1
                if diag2.count(fill) == 2 and diag2.count(0) == 1:
                    count += 1

                table[(tuple(config), fill)] = count > 0
        return table

WIN_IN_ONE_LOOKUP = generate_win_in_one_table()

# Modified based on the FourthAgent with addition of freedom as a feature
class SeventeenthAgent(KillerAgent):
    def __init__(self):
        super().__init__()

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

        central_pressure = int(lbs[1, 1] == opp_fill)
        if central_pressure:
            global_weights = np.array([[3, 2, 3], [2, 1, 2], [3, 2, 3]])
        else:
            global_weights = np.array([[2, 1, 2], [1, 3, 1], [2, 1, 2]])

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

        two_in_line = 0
        for line in [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1

        # Freedom (if previous target board is completed)
        freedom = 0.0
        if state.prev_local_action is not None:
            i, j = state.prev_local_action
            if lbs[i][j] != 0:
                freedom = 1.0

        weights = np.array([
            +0.0397,    # board_win_diff
            +0.1920,    # global_contrib
            +0.9511,    # win_in_one
            -0.9912,    # opp_win_in_one
            +0.1723,    # filled_ratio
            +0.05,      # two_in_line
            +0.20       # freedom 
        ])
        intercept = 0.0034

        if filled_ratio < 0.20:
            weights[1] *= 5.0  # global_contrib
        elif filled_ratio < 0.40:
            weights[1] /= 3.0
            weights[2] *= 3.0  # win_in_one
            weights[3] *= 2.5  # opp_win_in_one
            weights[6] *= 2.0
        else:
            weights[2] *= 2.5

        if board_win_diff < 0: # opp leading, play more defensively
            weights[3] *= 3.0
            weights[2] *= 0.8

        features = np.array([
            board_win_diff,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            filled_ratio,
            two_in_line,
            freedom
        ])

        return float(np.dot(weights, features) + intercept)

    def has_local_win_in_one(self, sub_board, fill):
        return WIN_IN_ONE_LOOKUP[(tuple(sub_board.flatten()), fill)]
