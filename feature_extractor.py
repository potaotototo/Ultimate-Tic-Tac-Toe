import numpy as np

class FeatureExtractor:
    def extract_features(self, state):
        num_features = 10

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

        # Feature 4: Number of valid actions (normalized)
        num_valid_actions = len(state.get_all_valid_actions()) / 81

        # Feature 5: Opponent forced into a bad board (removed)

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

        # Feature 11: Boards where you’re ahead (normalized)
        leading_boards = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] != 0:
                    continue
                local = board[i][j]
                my_count = np.sum(local == my_fill)
                opp_count = np.sum(local == opp_fill)
                if my_count > opp_count:
                    leading_boards += 1
        leading_boards /= 9  

        # Feature 12: Available sub-boards (normalized)
        available_boards = np.sum(lbs == 0) / 9

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

        return np.array([
            my_won - opp_won,
            num_valid_actions,
            num_tied,
            partial_boards,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            leading_boards,
            available_boards,
            control_2_in_line
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
