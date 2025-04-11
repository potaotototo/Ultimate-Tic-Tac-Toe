import numpy as np

class MoreMetaFeatureExtractor:
    def extract_features(self, state):
        lbs = state.local_board_status
        board = state.board
        my_fill = state.fill_num
        opp_fill = 3 - my_fill

        # 1) Basic local boards won
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

        # 2) "Win in one" for local boards
        win_in_one = 0
        opp_win_in_one = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] != 0:    # sub-board is already won or lost
                    continue
                local_board = board[i][j]
                if self.has_local_win_in_one(local_board, my_fill):
                    win_in_one += 1
                if self.has_local_win_in_one(local_board, opp_fill):
                    opp_win_in_one += 1
        win_in_one /= 9
        opp_win_in_one /= 9

        # 3) Filled ratio (9×9 board)
        filled_ratio = np.sum(state.board != 0) / 81 / 9

        # 4) Freedoms & blocking
        freedom = 0.0
        if state.prev_local_action is not None:
            i, j = state.prev_local_action
            if lbs[i][j] != 0:
                freedom = 1.0

        blocking_opportunities = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:
                    local_board = board[i][j]
                    if self.has_local_win_in_one(local_board, opp_fill):
                        blocking_opportunities += 1
        blocking_opportunities /= 9

        player_turn_advantage = 0.1 if state.fill_num % 2 == 0 else -0.1

        # 5) Agent "two_in_line" in local boards
        two_in_line = 0
        
        # 6) Opponent "two_in_line" in local boards
        opponent_two_in_line = 0
        local_win_lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]
        for line in local_win_lines:
            cells = [lbs[i][j] for i, j in line]
            # agent near-completion
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1
            # opponent near-completion
            if cells.count(opp_fill) == 2 and cells.count(0) == 1:
                opponent_two_in_line += 1

        # 7) restricted freedom overall
        total_available_moves = 9
        opponent_valid_moves = sum(lbs[i][j] == 0 for i in range(3) for j in range(3))
        restricted_freedom = 1 - (opponent_valid_moves / total_available_moves)
        restricted_and_blocked = restricted_freedom + blocking_opportunities

        # 8) *** New synergy features ***: meta_two_in_a_line
        # (lines in the meta board (lbs) that have 2 of my_fill or opp_fill + 1 empty)
        meta_two_in_a_line = self.count_meta_two_in_line(lbs, my_fill)
        meta_opp_two_in_a_line = self.count_meta_two_in_line(lbs, opp_fill)

        # Return the feature vector (11 features now, or more if you prefer).
        return np.array([
            my_won - opp_won,             # 0
            center_control,               # 1
            global_contrib,               # 2
            win_in_one,                   # 3
            opp_win_in_one,               # 4 (optional if you want it)
            filled_ratio,                 # 5
            freedom,                      # 6
            blocking_opportunities,       # 7
            player_turn_advantage,        # 8
            two_in_line,                  # 9
            opponent_two_in_line,         # 10
            restricted_and_blocked,       # 11
            meta_two_in_a_line,           # 12
            meta_opp_two_in_a_line        # 13
        ])

    def has_local_win_in_one(self, sub_board, fill):
        """Check if sub_board can be won in one move by 'fill'."""
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

    def count_meta_two_in_line(self, lbs, fill):
        """Count how many lines in the 3x3 meta-board (lbs) 
           have exactly 2 'fill' and 1 empty."""
        lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]
        count = 0
        for line in lines:
            cells = [lbs[r, c] for (r, c) in line]
            if cells.count(fill) == 2 and cells.count(0) == 1:
                count += 1
        return count
