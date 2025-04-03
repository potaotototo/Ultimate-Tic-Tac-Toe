import numpy as np

class FeatureExtractor:
    def extract_features(self, state):
        num_features = 7

        if state.is_terminal():
            return np.zeros(num_features)

        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        # Feature 1: Local boards won
        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)

        # Feature 2: Centre priority (removed)

        # Feature 3: Local threats (removed)

        # Feature 4: Number of valid actions (normalized) (removed)

        # Feature 5: Opponent forced into a bad board (removed)

        # Feature 6: Number of tied boards (normalized) (removed)

        # Feature 7: Partial (unfinished) boards in progress (normalized) (removed)

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

        # Feature 11: Boards where we are ahead (normalized) (removed)

        # Feature 12: Available sub-boards (normalized) (removed)

        # Feature 13: Number of filled cells
        filled_ratio = np.sum(state.board != 0) / 81 / 9

        # Feature 13: Consider 2 in a line
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

        # Feature 14: Freedom of next move
        freedom = 0.0
        if state.prev_local_action is not None:
            i, j = state.prev_local_action
            if state.local_board_status[i][j] != 0:
                freedom = 1.0


        features = np.array([
            my_won - opp_won,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            control_2_in_line,
            filled_ratio,
            freedom
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
