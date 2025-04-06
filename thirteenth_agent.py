import numpy as np
from utils import State, Action
from killer_agent import KillerAgent

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

# Thirteenth: Twelve with new computation
class ThirteenthAgent(KillerAgent):
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

        center_control = 0
        if board[1][1][1, 1] == my_fill:
            center_control = 1
        elif board[1][1][1, 1] == opp_fill:
            center_control = -1

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
        opp_two_in_line = 0
        for line in [
            [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]
        ]:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1
            if cells.count(opp_fill) == 2 and cells.count(0) == 1:
                opp_two_in_line += 1

        blocking_opportunities = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:
                    local_board = board[i][j]
                    if self.has_local_win_in_one(local_board, opp_fill):
                        blocking_opportunities += 1
        blocking_opportunities /= 9

        restricted_freedom = 1 - (np.sum(lbs == 0) / 9)
        restricted_and_blocked = restricted_freedom + blocking_opportunities

        meta_two_in_line = 1 if two_in_line >= 1 else 0
        meta_opp_two_in_line = 1 if opp_two_in_line >= 1 else 0

        features = np.array([
            board_win_diff,
            center_control,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            filled_ratio,
            two_in_line,
            opp_two_in_line,
            restricted_freedom,
            blocking_opportunities,
            restricted_and_blocked,
            meta_two_in_line,
            meta_opp_two_in_line
        ])

        # Lasso weights (early game)
        lasso_weights = np.array([
            +0.0445, +0.0382, +0.0814, +0.9902, -0.9982, -0.0000,
            -0.0000, -1.8251, +0.1544, -0.1651, -0.0084, +0.0190, 
            -0.0020
        ])
        lasso_intercept = +0.0025

        # Ridge weights (late game)
        ridge_weights = np.array([
            +0.0487, +0.0401, +0.0817, +1.0960, -0.5526, -0.4054,
            -0.5526, -1.9342, +0.0885, -0.0854, +0.0226, +0.0885, 
            -0.0854
        ])
        ridge_intercept = +0.0069

        if filled_ratio < 0.4:
            return float(np.dot(lasso_weights, features) + lasso_intercept)
        else:
            return float(np.dot(ridge_weights, features) + ridge_intercept)

    def has_local_win_in_one(self, sub_board, fill):
        return WIN_IN_ONE_LOOKUP[(tuple(sub_board.flatten()), fill)]
