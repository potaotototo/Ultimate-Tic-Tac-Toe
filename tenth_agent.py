import numpy as np
from utils import State, Action
from student_agent import StudentAgentWithCache

"""
This evaluation extends EighthAgent with extra synergy for the meta-board 
(the local_board_status 3x3). We also add more dynamic weighting to emphasize 
synergy in mid/late game scenarios.
 """
class TenthAgent(StudentAgentWithCache):
    def __init__(self):
        super().__init__()

    def evaluate(self, state):
        
        if state.is_terminal():
            return state.terminal_utility()

        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        # Basic scoreboard of local boards won
        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)
        board_win_diff = my_won - opp_won

        # Center control on meta-board (the middle local board)
        center_control = 0
        if board[1][1][1, 1] == my_fill:
            center_control = 1
        elif board[1][1][1, 1] == opp_fill:
            center_control = -1

        # Weighted global contribution for partial local boards
        global_weights = np.array([
            [2, 1, 2],
            [1, 3, 1],
            [2, 1, 2]
        ])
        global_contrib = np.sum(global_weights * (lbs == my_fill)) - np.sum(global_weights * (lbs == opp_fill))

        # "Win in one" and "Opponent win in one" checks on local boards
        win_in_one = 0
        opp_win_in_one = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] != 0:
                    continue  # sub-board is already won
                local = board[i][j]
                if self.has_local_win_in_one(local, my_fill):
                    win_in_one += 1
                if self.has_local_win_in_one(local, opp_fill):
                    opp_win_in_one += 1
        # Normalize
        win_in_one /= 9
        opp_win_in_one /= 9

        # Filled ratio across the entire 9x9
        filled_ratio = np.sum(state.board != 0) / 81 / 9  

        # Additional checks: blocking opportunities in local boards
        blocking_opportunities = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:
                    local_board = board[i][j]
                    if self.has_local_win_in_one(local_board, opp_fill):
                        blocking_opportunities += 1
        blocking_opportunities /= 9

        # Slight advantage if we are the second player or not
        # (or your logic: 0.1 if even fill_num, else -0.1)
        player_turn_advantage = 0.1 if state.fill_num % 2 == 0 else -0.1

        # Two-in-a-line on local boards (agent’s near-complete lines)
        two_in_line = self.count_near_completions(lbs, my_fill)
        opp_two_in_line = self.count_near_completions(lbs, opp_fill)

        # Freedoms or restricted
        total_available_moves = 9
        opponent_valid_moves = sum(lbs[i][j] == 0 for i in range(3) for j in range(3))
        restricted_freedom = 1 - (opponent_valid_moves / total_available_moves)
        restricted_and_blocked = restricted_freedom + blocking_opportunities

        # New feature: meta-board synergy
        # Count how many lines in lbs (the 3x3 meta board) are 2 in a row for agent or opp
        meta_two_in_a_line = self.count_near_completions(lbs, my_fill)
        meta_opp_two_in_a_line = self.count_near_completions(lbs, opp_fill)

        freedom = 0.0
        if state.prev_local_action is not None:
            i, j = state.prev_local_action
            if state.local_board_status[i][j] != 0:
                freedom = 1.0

        features = np.array([
            board_win_diff,
            center_control,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            player_turn_advantage,
            two_in_line,
            opp_two_in_line,
            restricted_and_blocked,
            meta_two_in_a_line,
            meta_opp_two_in_a_line,
            freedom
        ])

        weights = np.array([
            0.0369,    # board_win_diff
            0.0407,    # center_control
            0.0725,    # global_contrib
            0.7170,    # win_in_one
            -0.700,    # opp_win_in_one
            -1.700,    # player_turn_advantage
            0.2682,    # two_in_line
            -0.0822,   # opp_two_in_line
            -0.4630,   # restricted_and_blocked
            0.5000,    # meta_two_in_a_line (synergy)
            -0.600,    # meta_opp_two_in_a_line
            0.5000     # freedom
        ])
        intercept = 0.01

        # Dynamic weighting
        if filled_ratio < 0.10:
            # Early game: emphasize global board approach
            weights[2] *= 2.0     # global_contrib
            weights[10] *= 1.2    # meta synergy (agent)
        elif filled_ratio < 0.40:
            # Mid game: bigger emphasis on immediate local wins and blocking
            weights[3] *= 1.8     # win_in_one
            weights[4] *= 2.0     # opp_win_in_one (defense)
            weights[9] *= 1.5     # restricted_and_blocked
            weights[10] *= 1.2    # meta synergy
        else:
            # Late game: strong finishing and heavy block
            weights[3] *= 2.2
            weights[4] *= 2.5
            weights[10] *= 1.5    # finishing synergy on meta
            weights[11] *= 2.0    # heavily penalize opponent synergy

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

    def count_near_completions(self, lbs, fill):
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
