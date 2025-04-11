import numpy as np

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

# Features from feature_extractor_rf with new defensive features
# Threshold importance 0.025
class DefensiveFeatureExtractor:
    def extract_features(self, state):
        
        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

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

        filled_ratio = np.sum(state.board != 0) / 81 / 9

        blocking_opportunities = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:  
                    local_board = board[i][j]
                    if self.has_local_win_in_one(local_board, opp_fill):
                        blocking_opportunities += 1
        blocking_opportunities /= 9

        player_turn_advantage = 0.1 if state.fill_num % 2 == 0 else -0.1

        two_in_line = 0
        for line in [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1

        opponent_two_in_line = 0
        for line in [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(opp_fill) == 2 and cells.count(0) == 1:
                opponent_two_in_line += 1

        opponent_valid_moves = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:  
                    opponent_valid_moves += 1

        meta_opp_two_in_line = 0
        for line in [
            [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)],
        ]:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(opp_fill) == 2 and cells.count(0) == 1:
                meta_opp_two_in_line += 1

        opp_win_potential = opp_win_in_one * 9
        total_free_boards = np.sum(lbs == 0)
        risky_ratio = (opp_win_potential / total_free_boards) if total_free_boards else 0.0

        features = np.array([
            my_won - opp_won, 
            center_control, 
            global_contrib, 
            win_in_one, 
            filled_ratio, 
            player_turn_advantage, 
            two_in_line, 
            opponent_two_in_line, 
            meta_opp_two_in_line,
            risky_ratio
        ])
        
        return features

    def has_local_win_in_one(self, sub_board, fill):
        return WIN_IN_ONE_LOOKUP[(tuple(sub_board.flatten()), fill)]
